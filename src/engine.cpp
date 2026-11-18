// LLMServe-Lite 本地推理服务 | src/engine.cpp
// 功能：Scheduler（调度器）通过 Engine 接口调用这里的生成逻辑。
//       MockEngine 按固定规则吐出预设文字，用来测试；
//       LlamaEngine 加载 GGUF 模型文件，先处理提示词，再逐个预测新 token，
//       每凑出一段完整文字就通过 emit 回调交给 Job（一次请求对应的任务对象）。
// 目的：在同一个生成接口下同时支持“模拟输出”和“真实模型”，
//       并把完整文字、停止状态和各阶段耗时交回给调度器。
//
// 先弄懂两个词：
//   token：模型处理文字的最小单位，可能是一个词、半个词，甚至一个汉字的一部分字节。
//   KV 缓存：模型处理过的 token 留下的“中间记忆”，复用它就不用把相同的开头重新算一遍。
//
// 阅读顺序：
//   1. UTF-8 拼字（complete_utf8）与字符计数、模拟引擎（MockEngine）
//   2. 真实模型的初始化（Backend、LlamaEngine 构造函数）
//   3. generate：套聊天模板 → 分词 → 检查容量 → 复用前缀 → 预填充（prefill）
//   4. generate：逐 token 解码（decode）→ 保存缓存
//   5. 引擎工厂（make_engine）：按配置选择用哪个引擎


#include "serve.hpp"  // HTTP、调度器与生成引擎共用的配置、请求和任务类型
#include <algorithm>
#include <stdexcept>
#include <vector>
#ifdef LLMSERVE_WITH_LLAMA
#include "llama.h"  // llama.cpp 的模型、上下文、分词和执行接口
#endif

namespace serve {


// 模块：UTF-8 文本边界处理（complete_utf8）
// 功能：缓存跨 token 的 UTF-8 字节，只吐出完整字符，并把非法序列替换为 U+FFFD（�）。
// 目的：保证发给客户端的每段文字都是完整、合法的 UTF-8，不出现乱码或半个字。
// 通俗讲：一个汉字占 3 个字节，而模型的一个 token 可能只输出了其中 1~2 个字节。
//         如果直接发出去，客户端会看到乱码。所以没凑齐的字节先留在 buffer 里，下次再拼。
// 规则：首字节决定该字符占 1~4 字节；后续字节必须以二进制 10 开头；
//       非法序列，或结束时仍残缺的序列，用 U+FFFD 替换。
// 参数 buffer：待处理的字节缓冲；函数会把已处理的部分删掉，只留下残缺的尾巴。
// 参数 flush：为 true 表示生成已结束，不会再有后续字节，残缺部分也要处理掉。
// 返回值：本次可以安全发给客户端的完整文字。
std::string complete_utf8(std::string& buffer, bool flush) {
    size_t pos = 0;
    std::string out;
    while (pos < buffer.size()) {
        auto c = static_cast<unsigned char>(buffer[pos]);
        // 根据首字节的高位判断字符宽度：
        // 0xxxxxxx 为 1 字节（ASCII）；110xxxxx 为 2 字节；1110xxxx 为 3 字节；11110xxx 为 4 字节；
        // 其余（如孤立的 10xxxxxx）不是合法首字节，宽度记 0
        size_t width = c < 0x80 ? 1
                                : ((c & 0xe0) == 0xc0
                                       ? 2
                                       : ((c & 0xf0) == 0xe0 ? 3 : ((c & 0xf8) == 0xf0 ? 4 : 0)));
        // 非法首字节：输出替换字符，跳过这 1 个字节继续
        if (!width) {
            out += "\xef\xbf\xbd";
            ++pos;
            continue;
        }


        // 模块：尾部字节不足（UTF-8 残缺尾巴）
        // 功能：字节数不够一个完整字符时，决定是“先留着”还是“当作残缺丢弃”。
        // 生成还没结束（flush 为 false）：先留着，等下一个 token 带来剩下的字节。
        // 生成已结束（flush 为 true）：再也等不到了，输出一个替换字符并清掉剩余字节。
        // 例：“你”的 UTF-8 是 E4 BD A0。这次只拿到 E4 BD，就先留在 buffer 里；
        //     下次 A0 到了，再把完整的“你”交给 emit。
        //     可见一个 token 的边界和一个字的边界可以不一致。
        if (pos + width > buffer.size()) {
            if (flush) {
                out += "\xef\xbf\xbd";
                pos = buffer.size();
            }
            break;
        }


        // 检查后续字节是否都以 10 开头
        bool valid = true;
        for (size_t k = 1; k < width; ++k)
            if ((static_cast<unsigned char>(buffer[pos + k]) & 0xc0) != 0x80)
                valid = false;
        // 额外排除“过长编码”“代理区”“超出 Unicode 上限”等非法情形：
        // 过长编码指本来 1 字节能表示的字符却用了 2~4 字节；
        // 代理区（U+D800~U+DFFF）是 UTF-16 专用，不能出现在 UTF-8 中；
        // Unicode 的最大码点是 U+10FFFF，更大的都不合法。
        if (valid && width > 1) {
            const auto b = static_cast<unsigned char>(buffer[pos + 1]);
            if ((width == 2 && c < 0xc2) ||                                    // 2 字节过长编码
                (width == 3 && ((c == 0xe0 && b < 0xa0) || (c == 0xed && b >= 0xa0))) ||  // 3 字节过长/代理区
                (width == 4 && (c > 0xf4 || (c == 0xf0 && b < 0x90) || (c == 0xf4 && b >= 0x90))))  // 4 字节过长/超上限
                valid = false;
        }
        if (!valid) {
            out += "\xef\xbf\xbd";
            ++pos;
            continue;
        }
        // 合法：整个字符原样输出
        out.append(buffer, pos, width);
        pos += width;
    }
    // 已处理的部分从 buffer 删除，剩下的（残缺字节）留待下次
    buffer.erase(0, pos);
    return out;
}

namespace {


// 模块：模拟计数（codepoints）
// 功能：数出字符串里有多少个字符（码点），仅供 Mock 引擎估算输入长度。
// 目的：用一个简单、可预测的规则代替真实分词，方便测试容量检查。
// 原理：UTF-8 的后续字节都以 10 开头，不以 10 开头的字节才是一个字符的开头，数它们就是字符数。
size_t codepoints(const std::string& s) {
    return std::count_if(s.begin(), s.end(), [](unsigned char c) {
        return (c & 0xc0) != 0x80;
    });
}


// 模块：模拟引擎（MockEngine）
// 功能：不加载任何模型，循环输出固定的几个汉字。
// 目的：让测试可以精确控制延迟、排队和错误发生的时机，不用等真实模型。
// 可控的行为：
//   - mock_delay_ms：每个 token 之间等多久，用来模拟“生成很慢”；
//   - prompt 为 "__error__"：开始就抛异常，测试失败路径；
//   - prompt 为 "__error_after__"：生成到第 4 个 token 时抛异常，测试“输出到一半失败”。
class MockEngine final : public Engine {
    Config c_;

public:
    explicit MockEngine(Config c) : c_(std::move(c)) {
    }

    std::string name() const override {
        return "mock";
    }

    // 参数 emit：每产生一段文字就调用；返回 false 表示接收方不要了（如客户端断开），应立刻停下。
    // 参数 stop：每隔一小段时间检查；返回 true 表示请求被取消或超时，应立刻停下。
    Generation generate(const Request& r,
                        const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
        // 字符数 + 8 只是 Mock 的上下文估算，真实引擎使用聊天模板后的实际分词结果。
        const int n = int(codepoints(r.prompt)) + 8;


        // 模块：模拟容量与错误检查
        // 功能：用可预测的字符数检查“输入 + 预留输出”是否超过上下文，并按特殊提示词注入错误。
        // 注意：这里的计数方式与真实模型的词表分词不同，两者的临界值不会一致。
        if (n + r.max_tokens > c_.context_tokens)
            throw std::invalid_argument(
                "input plus output reservation exceeds context (mock codepoint estimate)");
        if (r.prompt == "__error__")
            throw std::runtime_error("injected mock failure");
        const std::vector<std::string> pieces = {"你", "好", "，", "这", "是", "测", "试", "。"};
        Generation g;
        g.prompt_tokens = n;
        const auto begin = Clock::now();
        for (int i = 0; i < r.max_tokens; ++i) {
            // 把总延迟切成每 5ms 一小段，每段之间都检查一次 stop，这样取消请求能很快生效，
            // 而不是傻等完整的延迟。
            for (int waited = 0; waited < c_.mock_delay_ms; waited += 5) {
                if (stop())
                    return g;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(std::min(5, c_.mock_delay_ms - waited)));
            }
            if (stop())
                return g;
            if (r.prompt == "__error_after__" && i == 3)
                throw std::runtime_error("injected midstream mock failure");
            // 片段数量只有 8 个，用取余循环使用
            if (!emit(pieces[size_t(i) % pieces.size()]))
                return g;
            ++g.output_tokens;
        }
        // 只有完整生成完才会记录耗时；中途因取消或 emit 失败返回时，decode_ms 保持默认值
        g.decode_ms = millis(begin, Clock::now());
        return g;
    }
};
#ifdef LLMSERVE_WITH_LLAMA


// 模块：后端生命周期（Backend）
// 功能：初始化和释放 llama.cpp 的全局后端。
// 目的：用 RAII（构造时初始化、析构时释放）保证后端一定会被释放，不用手动记着调用。
struct Backend {
    Backend() {
        llama_backend_init();
    }

    ~Backend() {
        llama_backend_free();
    }
};


// 模块：CPU 模型引擎（LlamaEngine）
// 功能：持有唯一的模型和一个可复用的 context（上下文，保存 KV 缓存和计算状态）。
// 目的：用真实的 GGUF 模型生成文字，并通过前缀缓存减少重复计算。
// RAII 管理资源：成员按声明顺序的反序销毁，所以先释放 context、再释放 model、最后释放 backend。
//     这个顺序不能乱：context 依赖 model，model 依赖 backend。
// 模型和 context 在启动时创建一次；后续请求都复用它们，但不会共享对话历史。
class LlamaEngine final : public Engine {
    Backend backend_;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_{nullptr, llama_free};
    Config c_;


    // 模块：前缀缓存身份（previous_prompt_）
    // 功能：只保存上一次成功请求的输入 token，用来和这次的输入比较，找出相同的开头。
    // 为什么只存 token 就够：权重和聊天模板在同一个引擎里始终不变，
    //     所以“token 相同”就等于“模型看到的输入相同”。
    std::vector<llama_token> previous_prompt_;


public:
    explicit LlamaEngine(Config c) : c_(std::move(c)) {
        // CPU 首版不把任何模型层放到 GPU 计算；GGUF 中的量化权重由 llama.cpp 加载执行。


        // 模块：模型加载与上下文创建
        // 功能：构造函数加载模型文件，并按配置创建 context。
        // 提示：引擎会被多次请求复用，所以内存一直被占用是预期行为，不是内存泄漏。
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model_.reset(llama_model_load_from_file(c_.model_path.c_str(), mp));
        if (!model_)
            throw std::runtime_error("cannot load GGUF model: " + c_.model_path);
        auto cp = llama_context_default_params();
        // n_ctx 是上下文容量（输入加输出一共最多多少个 token）；
        // n_batch = 128 是一次最多提交的 token 数，n_ubatch = 64 是内部再切的小批大小。
        // 这两个是“批大小”，不代表同时能处理 128 个请求。
        cp.n_ctx = c_.context_tokens;
        cp.n_batch = 128;
        cp.n_ubatch = 64;
        cp.n_threads = c_.threads;
        cp.n_threads_batch = c_.threads;
        ctx_.reset(llama_init_from_model(model_.get(), cp));
        if (!ctx_)
            throw std::runtime_error("cannot allocate llama context");
        // 聊天模型要先把用户文字包装成训练时约定的对话格式（聊天模板）；
        // 缺少模板时不能随便拼一串字符冒充模型原本的输入，所以直接报错。
        if (!llama_model_chat_template(model_.get(), nullptr))
            throw std::runtime_error("model has no chat template");
    }

    std::string name() const override {
        return "llama";
    }

    // 参数和返回值的含义与 MockEngine::generate 相同。
    // 返回的 Generation 里记录了输入/输出 token 数、复用了多少缓存、各阶段耗时和结束原因。
    Generation generate(const Request& r,
                        const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
        const auto begin = Clock::now();


        // 模块：请求准备
        // 功能：取出上次的前缀记录，并把用户文字套上聊天模板。
        // 隔离原则：不同请求之间不共享对话历史。
        //     缓存开关默认关闭；开启后，旧请求的 KV 缓存也只会在分词后按“共同前缀”保留，
        //     共同前缀之外的内容一律删除，绝不直接沿用全部历史。
        // 这里先把 previous_prompt_ 清空：只要中途任何一步失败，下一次请求就会完整重算，
        //     不会拿着一份不可靠的缓存记录继续用。
        auto previous = std::move(previous_prompt_);
        previous_prompt_.clear();
        llama_chat_message msg = {"user", r.prompt.c_str()};
        const char* tmpl = llama_model_chat_template(model_.get(), nullptr);
        // 先按“原文长度的 2 倍再加 1024”准备格式化缓冲区，通常够用
        std::vector<char> formatted(r.prompt.size() * 2 + 1024);
        int count =
            llama_chat_apply_template(tmpl, &msg, 1, true, formatted.data(), int(formatted.size()));
        if (count < 0)
            throw std::runtime_error("unsupported chat template");
        // 返回值大于缓冲区，说明缓冲区不够：按返回的实际长度扩容后重新套一次模板
        if (count > int(formatted.size())) {
            formatted.resize(count + 1);
            count = llama_chat_apply_template(
                tmpl, &msg, 1, true, formatted.data(), int(formatted.size()));
            if (count < 0 || count > int(formatted.size()))
                throw std::runtime_error("chat template formatting failed");
        }


        // 模块：分词与容量检查
        // 功能：把套好模板的文字切成 token，并按真实 token 数检查上下文是否放得下。
        auto vocab = llama_model_get_vocab(model_.get());
        // llama.cpp 的约定：传入空缓冲区时，返回值是“所需 token 数的负数”。
        // 所以先问一次长度（得到 -n），分配好空间后再真正分词。
        int n = llama_tokenize(vocab, formatted.data(), count, nullptr, 0, true, true);
        if (n >= 0)
            throw std::runtime_error("unexpected tokenizer size result");
        std::vector<llama_token> tokens(-n);
        n = llama_tokenize(
            vocab, formatted.data(), count, tokens.data(), int(tokens.size()), true, true);
        if (n <= 0)
            throw std::runtime_error("tokenization failed");
        tokens.resize(n);
        // 容量校验：模板化后的输入 token 数 + 最大输出预留 <= 实际 context 容量。
        // 例：实际分词后输入为 480 个 token，又要最多输出 64 个，
        //     512 的上下文容纳不下这次请求，必须拒绝。
        if (n + r.max_tokens > int(llama_n_ctx(ctx_.get())))
            throw std::invalid_argument("templated input plus output reservation exceeds context");


        // 模块：前缀匹配与缓存裁剪
        // 功能：找出这次输入和上次输入“相同的开头”有多少个 token，这部分的 KV 缓存可以直接用。
        // 要点：
        //   - 逐个 token 比较，不是字符串包含关系；
        //   - 复用数量有三个上限：上次输入的长度、这次输入长度减 1、配置的 prefix_cache_tokens；
        //   - 减 1 是故意的：至少要重新算最后一个输入 token，才能拿到这次预测所需的分数（logits）；
        //     所以即使两次提示词完全一样，也会重算最后一个。
        // 例：上次 token 开头为 [11,22,33]，这次为 [11,22,44]。
        //     从头逐个比，第三项就不同了，所以最多复用 2 个。
        size_t reused = 0;
        const size_t limit =
            std::min({previous.size(), tokens.size() - 1, size_t(c_.prefix_cache_tokens)});
        while (reused < limit && previous[reused] == tokens[reused])
            ++reused;
        // llama_memory_seq_rm(…, 0, reused, -1)：删除序列 0 中第 reused 位及之后的缓存
        // （-1 表示一直删到末尾），也就是把上次生成的内容和不相同的部分都清掉。
        // 没有可复用的前缀，或者删除失败，就退回安全做法：清空全部缓存，从头完整计算。
        if (!reused || !llama_memory_seq_rm(llama_get_memory(ctx_.get()), 0, reused, -1)) {
            reused = 0;
            llama_memory_clear(llama_get_memory(ctx_.get()), true);
        }
        const auto prepare_end = Clock::now();


        // 模块：取消绑定
        // 功能：把 stop 回调交给 llama.cpp，让底层长时间计算时也能检查是否该停，
        //       并保证函数返回前一定解除绑定。
        auto abort = [](void* data) -> bool {
            return (*static_cast<const std::function<bool()>*>(data))();
        };
        llama_set_abort_callback(ctx_.get(), abort, const_cast<std::function<bool()>*>(&stop));

        // stop 回调的地址只在本次 generate 调用中有效，函数返回后就失效了。
        // 所以用 RAII：Reset 对象离开作用域时（无论正常返回还是抛异常）都会解除绑定，
        // 避免 llama.cpp 之后访问一个已经失效的地址。
        struct Reset {
            llama_context* ctx;

            ~Reset() {
                llama_set_abort_callback(ctx, nullptr, nullptr);
            }
        } reset{ctx_.get()};


        // 模块：Prefill（预填充）
        // 功能：分块处理还没复用的提示词 token，建立好 KV 缓存。
        // 目的：让模型先“读完”整段提示词，之后才能逐个预测新 token。
        // 一次最多提交 128 个 token，最后一批按真实剩余量截短；
        //     例如还剩 130 个，就分成 128 和 2 两批。
        // 这里拆的是“一个请求的提示词”，和等待队列里有几个请求无关。
        // 每批之间都检查 stop，所以长提示词也能被及时取消。此阶段还没有输出任何回答文字。
        const auto prefill_begin = Clock::now();
        for (int at = int(reused); at < n; at += 128) {
            if (stop())
                return {n, 0, "cancelled"};
            auto batch = llama_batch_get_one(tokens.data() + at, std::min(128, n - at));
            if (llama_decode(ctx_.get(), batch) != 0)
                throw std::runtime_error("prefill decode failed or aborted");
        }
        const auto prefill_end = Clock::now();


        // 模块：Decode（逐 token 解码）
        // 功能：一个一个采样新 token，转成文字并投递给客户端。
        // 采样方式：贪心采样，每次选当前分数最高的 token，所以同样的输入总是得到同样的输出。
        // 结束方式：遇到 EOG（end of generation，模型主动表示“说完了”）就停，
        //           EOG 本身不算输出 token；否则最多生成 max_tokens 个。
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(), llama_sampler_free);
        if (!sampler)
            throw std::runtime_error("sampler allocation failed");
        Generation g;
        g.prompt_tokens = n;
        g.cached_prompt_tokens = int(reused);
        g.prompt_prepare_ms = millis(begin, prepare_end);
        g.prefill_compute_ms = millis(prefill_begin, prefill_end);
        // 注意：prefill_ms 从函数开头算起，还包含清缓存、套模板、分词和创建 sampler 的时间，
        // 不是单独的矩阵计算时间；纯计算时间看 prefill_compute_ms。
        const auto decode_begin = Clock::now();
        g.prefill_ms = millis(begin, decode_begin);
        std::string pending;  // 还没凑成完整字符的字节
        for (int i = 0; i < r.max_tokens; ++i) {
            if (stop())
                return g;
            // 采样拿到的是 token 编号（下标 -1 表示取最近一次计算出的分数）；
            // 要转成字节、拼成完整字符后才投递，所以一次客户端文本事件不必对应一个 token。
            auto token = llama_sampler_sample(sampler.get(), ctx_.get(), -1);
            if (llama_vocab_is_eog(vocab, token)) {
                g.finish_reason = "stop";
                break;
            }
            std::vector<char> piece(64);
            int bytes =
                llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, false);
            // 这次转换所给空间不足时，返回负数，其绝对值就是需要的容量；
            // 扩容后用同一个 token 重试一次。
            if (bytes < 0) {
                piece.resize(-bytes);
                bytes =
                    llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, false);
            }
            if (bytes < 0)
                throw std::runtime_error("token to piece failed");
            pending.append(piece.data(), bytes);
            ++g.output_tokens;
            // 只取出已经凑齐的完整字符；emit 返回 false 表示接收方不要了，直接结束
            auto text = complete_utf8(pending);
            if (!text.empty() && !emit(text))
                return g;
            // 把刚生成的 token 送回模型，计算下一步的分数；
            // 已达输出上限时不再做这次没有用的计算。
            if (i + 1 < r.max_tokens &&
                llama_decode(ctx_.get(), llama_batch_get_one(&token, 1)) != 0)
                throw std::runtime_error("decode failed or aborted");
        }
        // 生成结束：把 pending 里残余的字节处理掉（残缺的会变成替换字符）
        auto tail = complete_utf8(pending, true);
        if (!tail.empty())
            emit(tail);
        // decode_ms 含采样、文本转换和 emit 背压等待（接收方来不及收而被迫等待），
        // 不能解释为纯 CPU 算子执行时间。
        g.decode_ms = millis(decode_begin, Clock::now());


        // 模块：提交缓存
        // 功能：本次请求正常结束后，把这次的输入 token 记下来，供下一次请求做前缀匹配。
        // 只有走到这里并且没有被停止才保存：
        //     被中断的请求留下的 KV 状态不完整，下次不能从它继续。
        // 保存的数量不超过 prefix_cache_tokens，使缓存记录和 KV 占用都受 context 容量限制。
        // 本次生成出来的 token 虽然也在 KV 里，但不记入前缀，下次会被 seq_rm 删掉。
        if (c_.prefix_cache_tokens && !stop()) {
            tokens.resize(std::min(tokens.size(), size_t(c_.prefix_cache_tokens)));
            previous_prompt_ = std::move(tokens);
        }
        return g;
    }
};
#endif
}


// 模块：引擎工厂（make_engine）
// 功能：根据配置创建 Mock 或 CPU 真实模型引擎。
// 目的：调用方只拿到 Engine 接口，不用关心背后是哪一种实现。
// 提示：如果编译时没有开启 LLMSERVE_WITH_LLAMA，选择真实引擎会直接报错，提示用 cpu 预设重新配置。
std::unique_ptr<Engine> make_engine(const Config& c) {
    if (c.engine == "mock")
        return std::make_unique<MockEngine>(c);
#ifdef LLMSERVE_WITH_LLAMA
    return std::make_unique<LlamaEngine>(c);
#else
    throw std::runtime_error("llama backend not compiled; configure with --preset cpu");
#endif
}
}
