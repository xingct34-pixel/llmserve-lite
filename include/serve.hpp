#pragma once
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include "observability.hpp"

namespace serve {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;


// 模块：字段校验：检查JSON整数能否安全放入int32


inline bool is_int(const Json& j) {
    if (!j.is_number_integer())
        return false;
    if (j.is_number_unsigned())
        return j.get<uint64_t>() <= uint64_t(std::numeric_limits<int>::max());
    const auto n = j.get<int64_t>();
    return n >= std::numeric_limits<int>::min() && n <= std::numeric_limits<int>::max();
}


// 模块：服务配置：集中声明计算、队列、HTTP和超时参数
// 资源参数分三层：threads是模型CPU计算线程，http_threads是接待连接的线程，推理worker固定为1。
// queue_capacity限制等待推理的Job数量，event_capacity限制每个Job尚未被读取的文本事件数量。
// context_tokens容纳模板化提示词和生成预留量；max_output_tokens只限制新生成部分。


struct Config {
    std::string engine = "mock", host = "127.0.0.1", model_path;
    int port = 8080, threads = 4, context_tokens = 1024, max_output_tokens = 128;
    int queue_capacity = 4, http_threads = 8, event_capacity = 16;
    int queue_timeout_ms = 30000, generation_timeout_ms = 120000;
    int backpressure_timeout_ms = 2000, mock_delay_ms = 20;


    // 模块：可选前缀缓存：0为关闭；只复用本进程上一条成功请求的相同token前缀


    int prefix_cache_tokens = 0;


    // 模块：排队预算：等待请求max_tokens之和的上限，0为关闭；不是KV内存大小


    int queued_output_budget = 0;


    static Config load(const std::string&);
};


// 模块：请求数据：保存提示词、输出上限和流式选项


struct Request {
    std::string prompt;
    int max_tokens = 128;
    bool stream = false;
};


// 模块：生成统计：保存token数量、结束原因和引擎计时


struct Generation {
    int prompt_tokens = 0, output_tokens = 0;
    std::string finish_reason = "length";
    double prefill_ms = 0, decode_ms = 0;


    // 模块：真实计算口径：准备、实际prefill计算、已复用输入token分别记录


    double prompt_prepare_ms = 0, prefill_compute_ms = 0;
    int cached_prompt_tokens = 0;
};


// 模块：引擎接口：由Mock和llama.cpp实现统一生成回调


class Engine {
public:
    virtual ~Engine() = default;
    // generate仅由单个推理worker调用；emit交出完整文本片段，stop检查取消/执行超时。
    // name是只读名称，也会被HTTP指标接口读取；输出token数不等于文本事件数。
    virtual Generation generate(const Request&,
                                const std::function<bool(const std::string&)>&,
                                const std::function<bool()>&) = 0;
    virtual std::string name() const = 0;
};


// 模块：辅助接口：声明引擎工厂、UTF-8处理、日志及计时函数


std::unique_ptr<Engine> make_engine(const Config&);
std::string complete_utf8(std::string&, bool flush = false);
void log(const Json&);
double millis(Clock::time_point a, Clock::time_point b);


// 模块：任务状态：管理单个请求的生命周期、输出缓冲和终态
// 一个请求对应一个Job；shared_ptr使它跨HTTP处理器、worker、SSE回调保持存活。
// cancelled提供跨线程停止信号；原因、运行状态和事件容器仍由mutex保护。


struct Job {
    const std::string id;
    const Request request;
    const Config config;
    const Clock::time_point submitted = Clock::now();
    Clock::time_point started{}, finished{}, first{};
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::condition_variable cv;
    bool running = false, done = false, terminal_read = false;
    std::deque<Json> events;
    // 终态单独存放，不挤占有限的delta缓冲，慢消费者也不会堵住finish。
    Json terminal;
    std::string cancel_reason;
    explicit Job(std::string, Request, const Config&);
    void cancel(const std::string& reason);
    bool stopped();
    bool emit(const std::string& text);
    void finish(const Json& event);
    std::optional<Json> next();
    bool await_start();
};


// 模块：调度接口：管理有界等待队列、唯一worker和运行指标
// 有界FIFO调度器：等待任务从队头取出；HTTP并发接入不会增加模型解码worker数量。
// 涉及两把锁时顺序为Scheduler锁 -> Job锁；耗时推理不持有Scheduler锁。


class Scheduler {
    Config config_;
    std::unique_ptr<Engine> engine_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::shared_ptr<Job> active_;
    bool stopping_ = false;
    std::thread worker_;
    uint64_t sequence_ = 0, accepted_ = 0, rejected_ = 0, completed_ = 0, failed_ = 0,
             cancelled_ = 0, timed_out_ = 0;
    size_t queue_peak_ = 0;


    // 模块：有界观测：固定桶累计成功请求耗时，不保存无限增长的请求明细


    Histogram queue_seconds_, execution_seconds_, first_output_seconds_;
    Histogram prefill_seconds_, decode_seconds_;
    uint64_t prompt_tokens_ = 0, output_tokens_ = 0, cached_prompt_tokens_ = 0;


    void run();

public:
    Scheduler(Config, std::unique_ptr<Engine>);
    ~Scheduler();
    std::shared_ptr<Job> submit(Request);
    void stop();
    Json metrics() const;
};
}
