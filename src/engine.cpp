// LLMServe-Lite 本地推理服务 | src/engine.cpp
// Scheduler 通过 Engine 接口调用这里的生成逻辑。MockEngine 按规则吐出固定片段；
// LlamaEngine 加载 GGUF，先处理提示词，再逐个预测新 token，通过 emit 把文字交回 Job。
// 一个 token 可能只包含一个汉字的部分字节，因此输出前要先拼出完整的 UTF-8 字符。


#include "serve.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#ifdef LLMSERVE_WITH_LLAMA
#include "llama.h"
#endif

namespace serve {


// 模块：UTF-8 文本边界处理（complete_utf8）
// 作用：缓存跨 token 的 UTF-8 字节，只吐出完整字符，并把非法序列替换为 U+FFFD（�）。
// 通俗讲：一个汉字占 3 个字节，而模型的一个 token 可能只输出了其中 1~2 个字节。
//         如果直接发给客户端，会看到乱码或半个字。所以没凑齐的字节先留在 buffer 里，下次再拼。
// 规则：首字节决定该字符占 1~4 字节；后续字节必须以二进制 10 开头；
//       非法序列，或结束时仍残缺的序列，用 U+FFFD 替换。
// 参数 flush：为 true 表示生成已结束，不会再有后续字节，残缺部分也要处理掉。


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
        // 字节还没收齐：生成未结束就先留着等下次；已结束则只能当作残缺，替换后丢弃
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
        // 额外排除"过长编码""代理区""超出 Unicode 上限"等非法情形
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
// 作用：仅供 Mock 引擎估算输入有多少个字符。
// 原理：UTF-8 的后续字节都以 10 开头，不以 10 开头的字节才是一个字符的开头，数它们就是字符数。


size_t codepoints(const std::string& s) {
    return std::count_if(s.begin(), s.end(), [](unsigned char c) {
        return (c & 0xc0) != 0x80;
    });
}


// 模块：模拟引擎（MockEngine）
// 固定片段让测试可以控制延迟、队列和错误发生的时机。


class MockEngine final : public Engine {
    Config c_;

public:
    explicit MockEngine(Config c) : c_(std::move(c)) {
    }

    std::string name() const override {
        return "mock";
    }

    Generation generate(const Request& r,
                        const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
        // 字符数+8只是Mock的上下文估算，真实引擎使用聊天模板后的实际分词结果。
        const int n = int(codepoints(r.prompt)) + 8;
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
            if (!emit(pieces[size_t(i) % pieces.size()]))
                return g;
            ++g.output_tokens;
        }
        g.decode_ms = millis(begin, Clock::now());
        return g;
    }
};
#ifdef LLMSERVE_WITH_LLAMA


// 模块：后端生命周期：初始化和释放llama.cpp全局后端


struct Backend {
    Backend() {
        llama_backend_init();
    }

    ~Backend() {
        llama_backend_free();
    }
};


// 模块：CPU模型引擎：持有唯一模型和可复用context
// RAII管理资源：成员按声明的反序销毁，先context、再model、最后backend。
// 模型/context启动时创建一次；后续请求复用，但不会共享对话历史。


class LlamaEngine final : public Engine {
    Backend backend_;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_{nullptr, llama_free};
    Config c_;


    // 模块：前缀缓存身份：只保存上一成功请求的输入token，权重和模板固定在同一引擎


    std::vector<llama_token> previous_prompt_;


public:
    explicit LlamaEngine(Config c) : c_(std::move(c)) {
        // CPU首版不把任何模型层放到GPU计算；GGUF中的量化权重由llama.cpp加载执行。
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model_.reset(llama_model_load_from_file(c_.model_path.c_str(), mp));
        if (!model_)
            throw std::runtime_error("cannot load GGUF model: " + c_.model_path);
        auto cp = llama_context_default_params();
        // n_ctx是上下文容量；128/64限制逻辑批/内部微批的token量，不代表128个并发请求。
        cp.n_ctx = c_.context_tokens;
        cp.n_batch = 128;
        cp.n_ubatch = 64;
        cp.n_threads = c_.threads;
        cp.n_threads_batch = c_.threads;
        ctx_.reset(llama_init_from_model(model_.get(), cp));
        if (!ctx_)
            throw std::runtime_error("cannot allocate llama context");
        if (!llama_model_chat_template(model_.get(), nullptr))
            throw std::runtime_error("model has no chat template");
    }

    std::string name() const override {
        return "llama";
    }

    Generation generate(const Request& r,
                        const std::function<bool(const std::string&)>& emit,
                        const std::function<bool()>& stop) override {
        const auto begin = Clock::now();


        // 模块：1. 请求准备：清理上次状态并应用聊天模板
        // 清空上一请求的KV等上下文状态，保持请求隔离；这不等于重新加载模型权重。
        // 缓存开关默认为关闭。旧请求的KV在分词后按共同前缀裁剪，绝不直接沿用全部历史。


        auto previous = std::move(previous_prompt_);
        previous_prompt_.clear(); // 任意中途失败都会使下一次请求走完整prefill
        llama_chat_message msg = {"user", r.prompt.c_str()};
        const char* tmpl = llama_model_chat_template(model_.get(), nullptr);
        std::vector<char> formatted(r.prompt.size() * 2 + 1024);
        int count =
            llama_chat_apply_template(tmpl, &msg, 1, true, formatted.data(), int(formatted.size()));
        if (count < 0)
            throw std::runtime_error("unsupported chat template");
        if (count > int(formatted.size())) {
            formatted.resize(count + 1);
            count = llama_chat_apply_template(
                tmpl, &msg, 1, true, formatted.data(), int(formatted.size()));
            if (count < 0 || count > int(formatted.size()))
                throw std::runtime_error("chat template formatting failed");
        }


        // 模块：2. 分词与容量：按真实token数检查上下文预留


        auto vocab = llama_model_get_vocab(model_.get());
        // 先传空缓冲查询所需token数（不足时返回负数），按-n分配后再真正分词。
        int n = llama_tokenize(vocab, formatted.data(), count, nullptr, 0, true, true);
        if (n >= 0)
            throw std::runtime_error("unexpected tokenizer size result");
        std::vector<llama_token> tokens(-n);
        n = llama_tokenize(
            vocab, formatted.data(), count, tokens.data(), int(tokens.size()), true, true);
        if (n <= 0)
            throw std::runtime_error("tokenization failed");
        tokens.resize(n);
        // 容量校验：模板化输入token数+最大输出预留 <= 实际context容量。
        if (n + r.max_tokens > int(llama_n_ctx(ctx_.get())))
            throw std::invalid_argument("templated input plus output reservation exceeds context");


        // 模块：前缀匹配：逐token比较，至少重算输入最后一个token以取得本次logits
        // 不是字符串包含关系；共同前缀之外的生成历史必须删除。删除失败则安全退回全量计算。


        size_t reused = 0;
        const size_t limit =
            std::min({previous.size(), tokens.size() - 1, size_t(c_.prefix_cache_tokens)});
        while (reused < limit && previous[reused] == tokens[reused])
            ++reused;
        if (!reused || !llama_memory_seq_rm(llama_get_memory(ctx_.get()), 0, reused, -1)) {
            reused = 0;
            llama_memory_clear(llama_get_memory(ctx_.get()), true);
        }
        const auto prepare_end = Clock::now();


        // 模块：3. 取消绑定：让底层计算检查停止并在返回前解除回调


        auto abort = [](void* data) -> bool {
            return (*static_cast<const std::function<bool()>*>(data))();
        };
        llama_set_abort_callback(ctx_.get(), abort, const_cast<std::function<bool()>*>(&stop));

        // stop回调的地址只在本次调用中有效；RAII在正常返回和抛异常时都解除绑定。
        struct Reset {
            llama_context* ctx;

            ~Reset() {
                llama_set_abort_callback(ctx, nullptr, nullptr);
            }
        } reset{ctx_.get()};


        // 模块：4. Prefill：分块处理完整提示词以建立上下文
        // Prefill：分块处理整段输入并建立KV缓存；每块之间可响应取消。


        const auto prefill_begin = Clock::now();
        for (int at = int(reused); at < n; at += 128) {
            if (stop())
                return {n, 0, "cancelled"};
            auto batch = llama_batch_get_one(tokens.data() + at, std::min(128, n - at));
            if (llama_decode(ctx_.get(), batch) != 0)
                throw std::runtime_error("prefill decode failed or aborted");
        }
        const auto prefill_end = Clock::now();


        // 模块：5. Decode：逐token采样并投递完整UTF-8文本
        // 贪心采样每次选当前最高分token；EOG表示模型主动结束，不计入输出token数。


        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(), llama_sampler_free);
        if (!sampler)
            throw std::runtime_error("sampler allocation failed");
        Generation g;
        g.prompt_tokens = n;
        g.cached_prompt_tokens = int(reused);
        g.prompt_prepare_ms = millis(begin, prepare_end);
        g.prefill_compute_ms = millis(prefill_begin, prefill_end);
        // 此处prefill_ms还含清缓存、模板、分词与sampler准备，不是单独矩阵计算时间。
        const auto decode_begin = Clock::now();
        g.prefill_ms = millis(begin, decode_begin);
        std::string pending;
        for (int i = 0; i < r.max_tokens; ++i) {
            if (stop())
                return g;
            auto token = llama_sampler_sample(sampler.get(), ctx_.get(), -1);
            if (llama_vocab_is_eog(vocab, token)) {
                g.finish_reason = "stop";
                break;
            }
            std::vector<char> piece(64);
            int bytes =
                llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, false);
            if (bytes < 0) {
                piece.resize(-bytes);
                bytes =
                    llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, false);
            }
            if (bytes < 0)
                throw std::runtime_error("token to piece failed");
            pending.append(piece.data(), bytes);
            ++g.output_tokens;
            auto text = complete_utf8(pending);
            if (!text.empty() && !emit(text))
                return g;
            // 将本次生成的token送回模型计算下一步；已达输出上限时不再做无用的下一次计算。
            if (i + 1 < r.max_tokens &&
                llama_decode(ctx_.get(), llama_batch_get_one(&token, 1)) != 0)
                throw std::runtime_error("decode failed or aborted");
        }
        auto tail = complete_utf8(pending, true);
        if (!tail.empty())
            emit(tail);
        // decode_ms含采样、文本转换和emit背压等待，不能解释为纯CPU算子执行时间。
        g.decode_ms = millis(decode_begin, Clock::now());
        if (c_.prefix_cache_tokens && !stop()) {
            // 只在本次请求正常结束后允许复用；缓存元数据和KV容量均受context限制。
            tokens.resize(std::min(tokens.size(), size_t(c_.prefix_cache_tokens)));
            previous_prompt_ = std::move(tokens);
        }
        return g;
    }
};
#endif
}


// 模块：引擎工厂：依据配置创建Mock或CPU后端


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
