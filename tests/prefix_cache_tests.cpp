// LLMServe-Lite 本地推理服务 | tests/prefix_cache_tests.cpp
// 在真实模型上比较开启和关闭前缀缓存的请求，记录复用数量和文本差异。
// 还主动中止请求、发送超长输入，再检查下一次生成是否带入前一次留下的状态。


#include "serve.hpp"
#include <iostream>
#include <stdexcept>


// 模块：真实缓存回归：记录与全量prefill的文本差异，严格检查失效后的隔离恢复


int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("model path required");
        serve::Config config;
        config.engine = "llama";
        config.model_path = argv[1];
        config.context_tokens = 512;
        config.max_output_tokens = 32;
        config.threads = 2;
        auto baseline = serve::make_engine(config);
        config.prefix_cache_tokens = 256;
        auto cached = serve::make_engine(config);
        auto run = [](serve::Engine& engine, const std::string& prompt) {
            std::string text;
            auto stats = engine.generate(
                {prompt, 16, false},
                [&](const std::string& s) {
                    text += s;
                    return true;
                },
                [] {
                    return false;
                });
            return std::make_pair(text, stats);
        };
        const std::string a = "请用一句中文解释为什么服务需要有界队列。";
        const std::string b = "请用一句中文解释为什么推理模型只加载一次。";
        for (const auto& prompt : {a, a, b, a}) {
            const auto expected = run(*baseline, prompt);
            const auto actual = run(*cached, prompt);
            if (actual.first.empty() || actual.second.output_tokens > 16 ||
                actual.second.prompt_tokens != expected.second.prompt_tokens)
                throw std::runtime_error("invalid cached generation metadata");
            if (actual.second.cached_prompt_tokens == 0 && actual.first != expected.first)
                throw std::runtime_error("uncached generation differs from isolated baseline");
            std::cout << serve::Json{{"case", "cache_regression"},
                                     {"same_text_as_full_prefill", actual.first == expected.first},
                                     {"prompt_tokens", actual.second.prompt_tokens},
                                     {"cached_prompt_tokens", actual.second.cached_prompt_tokens},
                                     {"prefill_compute_ms", actual.second.prefill_compute_ms},
                                     {"output_tokens", actual.second.output_tokens}}
                             .dump()
                      << '\n';
        }
        const auto hit = run(*cached, a);
        const auto full_a = run(*baseline, a);
        if (hit.second.cached_prompt_tokens <= 0 ||
            hit.second.cached_prompt_tokens >= hit.second.prompt_tokens)
            throw std::runtime_error(
                "expected bounded prefix reuse with final prompt token recomputed");
        int stop_checks = 0;
        try {
            cached->generate(
                {a, 16, false},
                [](const std::string&) {
                    return true;
                },
                [&] {
                    return ++stop_checks > 1;
                });
        } catch (const std::runtime_error&) {
            if (stop_checks <= 1)
                throw; // 只接受已触发stop检查的底层取消错误
        }


        // 模块：取消后的第一条请求
        // 复用量应回到零，再将输出与干净上下文的基线比较。


        const auto after_cancel = run(*cached, a);
        if (after_cancel.second.cached_prompt_tokens != 0 || after_cancel.first != full_a.first)
            throw std::runtime_error("cancelled request leaked cached state");
        try {
            run(*cached, std::string(6000, 'x'));
            throw std::runtime_error("oversized context was accepted");
        } catch (const std::invalid_argument&) {
        }
        const auto after_error = run(*cached, a);
        if (after_error.second.cached_prompt_tokens != 0 || after_error.first != full_a.first)
            throw std::runtime_error("invalid request leaked cached state");
        std::cout
            << "PASS real prefix cache metadata, reuse, cancellation and failure recovery; text differences explicitly recorded\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
