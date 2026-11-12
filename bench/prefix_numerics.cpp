#include "serve.hpp"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {


// 模块：提示词准备：使用与服务一致的聊天模板和真实tokenizer，不用字符数冒充分词


std::vector<llama_token> tokenize(llama_model* model, const std::string& text) {
    llama_chat_message message{"user", text.c_str()};
    std::vector<char> buffer(4096);
    int length = llama_chat_apply_template(
        llama_model_chat_template(model, nullptr), &message, 1, true, buffer.data(), buffer.size());
    if (length <= 0 || length > int(buffer.size()))
        throw std::runtime_error("unexpected template length");
    auto vocabulary = llama_model_get_vocab(model);
    int count = llama_tokenize(vocabulary, buffer.data(), length, nullptr, 0, true, true);
    if (count >= 0)
        throw std::runtime_error("unexpected tokenizer size");
    std::vector<llama_token> tokens(-count);
    if (llama_tokenize(
            vocabulary, buffer.data(), length, tokens.data(), tokens.size(), true, true) != -count)
        throw std::runtime_error("tokenizer mismatch");
    return tokens;
}


// 模块：受控Prefill：显式改变分块方式，其他上下文参数保持一致


void feed(llama_context* context, std::vector<llama_token>& tokens, int start, int chunk = 128) {
    for (int index = start; index < int(tokens.size()); index += chunk)
        if (llama_decode(context,
                         llama_batch_get_one(tokens.data() + index,
                                             std::min(chunk, int(tokens.size()) - index))) != 0)
            throw std::runtime_error("numeric probe decode failed");
}


// 模块：数值比较：比较完整logits向量和归一化KL，不只检查生成文本是否非空


serve::Json compare(const float* reference, const float* candidate, int count) {
    double maximum = 0, square_sum = 0, reference_sum = 0, candidate_sum = 0;
    const auto ri = int(std::max_element(reference, reference + count) - reference);
    const auto ci = int(std::max_element(candidate, candidate + count) - candidate);
    for (int index = 0; index < count; index++) {
        if (!std::isfinite(reference[index]) || !std::isfinite(candidate[index]))
            throw std::runtime_error("non-finite logits");
        const double difference = double(reference[index]) - candidate[index];
        maximum = std::max(maximum, std::abs(difference));
        square_sum += difference * difference;
        reference_sum += std::exp(double(reference[index]) - reference[ri]);
        candidate_sum += std::exp(double(candidate[index]) - candidate[ci]);
    }
    double kl = 0;
    for (int index = 0; index < count; index++) {
        const double log_p = double(reference[index]) - reference[ri] - std::log(reference_sum);
        const double log_q = double(candidate[index]) - candidate[ci] - std::log(candidate_sum);
        kl += std::exp(log_p) * (log_p - log_q);
    }
    return {{"max_abs", maximum},
            {"rms", std::sqrt(square_sum / count)},
            {"kl", kl},
            {"same_argmax", ri == ci},
            {"reference_argmax", ri},
            {"candidate_argmax", ci}};
}


}


// 模块：因果对照：无跨请求缓存的分块、原缓存路径与相同形状的因果前缀参考分别对比


int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("model path required");

        struct Backend {
            Backend() {
                llama_backend_init();
            }

            ~Backend() {
                llama_backend_free();
            }
        } backend;

        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model)
            throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 128;
        cp.n_ubatch = 64;
        cp.n_threads = cp.n_threads_batch = 2;
        using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
        std::vector<Context> contexts;
        for (int index = 0; index < 4; index++) {
            contexts.emplace_back(llama_init_from_model(model.get(), cp), llama_free);
            if (!contexts.back())
                throw std::runtime_error("context allocation failed");
        }
        auto a = tokenize(model.get(), "请用一句中文解释为什么服务需要有界队列。");
        auto b = tokenize(model.get(), "请用一句中文解释为什么推理模型只加载一次。");
        int shared = 0;
        while (shared < int(std::min(a.size(), b.size() - 1)) && a[shared] == b[shared])
            shared++;
        if (!shared)
            throw std::runtime_error("probe needs a shared prefix");
        auto prefix = std::vector<llama_token>(b.begin(), b.begin() + shared);
        auto padded_b = b;
        padded_b.resize(a.size(), 0);
        feed(contexts[0].get(), b, 0); // 全量B
        feed(contexts[1].get(), prefix, 0);
        feed(contexts[1].get(), b, shared);   // 无跨请求缓存，仅把B拆成两块
        feed(contexts[2].get(), a, 0);        // A的旧前缀
        feed(contexts[3].get(), padded_b, 0); // 相同批大小、相同因果前缀的参考
        for (int index : {2, 3}) {
            if (!llama_memory_seq_rm(llama_get_memory(contexts[index].get()), 0, shared, -1))
                throw std::runtime_error("prefix trim failed");
            feed(contexts[index].get(), b, shared);
        }
        const int vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        double maximum_shape_error = 0;
        bool split_changed_logits = false;
        int split_argmax_differences = 0, cache_argmax_differences = 0;
        for (int step = 0; step < 16; step++) {
            auto full = llama_get_logits_ith(contexts[0].get(), -1);
            auto split = llama_get_logits_ith(contexts[1].get(), -1);
            auto cached = llama_get_logits_ith(contexts[2].get(), -1);
            auto shaped = llama_get_logits_ith(contexts[3].get(), -1);
            auto split_difference = compare(full, split, vocabulary_size);
            auto cache_difference = compare(full, cached, vocabulary_size);
            auto shape_difference = compare(shaped, cached, vocabulary_size);
            maximum_shape_error =
                std::max(maximum_shape_error, shape_difference["max_abs"].get<double>());
            split_changed_logits |= split_difference["max_abs"].get<double>() > 1e-5;
            split_argmax_differences += !split_difference["same_argmax"].get<bool>();
            cache_argmax_differences += !cache_difference["same_argmax"].get<bool>();
            std::cout << serve::Json{{"step", step},
                                     {"shared_tokens", shared},
                                     {"full_vs_uncached_split", split_difference},
                                     {"full_vs_cached", cache_difference},
                                     {"same_shape_causal_reference_vs_cached", shape_difference}}
                             .dump()
                      << '\n';
            llama_token forced = cache_difference["reference_argmax"].get<int>();
            for (auto& context : contexts)
                if (llama_decode(context.get(), llama_batch_get_one(&forced, 1)) != 0)
                    throw std::runtime_error("teacher-forced decode failed");
        }
        std::cout << serve::Json{{"summary", true},
                                 {"max_same_shape_error", maximum_shape_error},
                                 {"uncached_partition_changes_logits", split_changed_logits},
                                 {"split_argmax_differences", split_argmax_differences},
                                 {"cached_argmax_differences", cache_argmax_differences}}
                         .dump()
                  << '\n';
        if (maximum_shape_error > 1e-4)
            throw std::runtime_error("cached prefix differs from same-shape causal reference");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
