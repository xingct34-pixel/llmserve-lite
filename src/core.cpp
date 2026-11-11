#include "serve.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace serve {

// 模块：计时辅助：使用单调时钟计算毫秒差
double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// 模块：日志输出：为事件加时间戳并串行写入JSON日志
// system_clock记录可读墙钟时间；耗时用steady_clock，避免系统时间校准影响差值。
// 日志锁保证多个线程的JSON记录不互相穿插，调用方不把提示词正文写入日志。
void log(const Json& event) {
    static std::mutex mutex;
    auto record = event;
    record["timestamp_unix_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
    std::lock_guard<std::mutex> lock(mutex);
    std::cerr << record.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
}

// 模块：配置加载：检查字段、类型及本项目的资源边界
Config Config::load(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open config: " + path);
    Json j;
    file >> j;
    if (!j.is_object())
        throw std::runtime_error("config must be an object");
    Config c;
    const std::vector<std::string> keys = {"engine",
                                           "host",
                                           "model_path",
                                           "port",
                                           "threads",
                                           "context_tokens",
                                           "max_output_tokens",
                                           "queue_capacity",
                                           "http_threads",
                                           "event_capacity",
                                           "queue_timeout_ms",
                                           "generation_timeout_ms",
                                           "backpressure_timeout_ms",
                                           "mock_delay_ms"};
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (std::find(keys.begin(), keys.end(), it.key()) == keys.end())
            throw std::runtime_error("unknown config key: " + it.key());
        if (it.key() == "engine" || it.key() == "host" || it.key() == "model_path") {
            if (!it.value().is_string())
                throw std::runtime_error("config string required: " + it.key());
        } else if (!is_int(it.value()))
            throw std::runtime_error("config int32 required: " + it.key());
    }
#define READ(key) c.key = j.value(#key, c.key)
    READ(engine);
    READ(host);
    READ(model_path);
    READ(port);
    READ(threads);
    READ(context_tokens);
    READ(max_output_tokens);
    READ(queue_capacity);
    READ(http_threads);
    READ(event_capacity);
    READ(queue_timeout_ms);
    READ(generation_timeout_ms);
    READ(backpressure_timeout_ms);
    READ(mock_delay_ms);
#undef READ
    if (c.engine != "mock" && c.engine != "llama")
        throw std::runtime_error("engine must be mock or llama");
    if (c.host != "127.0.0.1")
        throw std::runtime_error("this build only binds 127.0.0.1");
    // 这些是本项目面向低内存机器的保守边界，不是llama.cpp或模型的通用极限。
    // HTTP线程至少比等待容量多3，配合接入上限给健康检查及拒绝响应留出处理余量。
    if (c.port < 1024 || c.port > 65535 || c.threads < 1 || c.threads > 6 ||
        c.context_tokens < 128 || c.context_tokens > 2048 || c.max_output_tokens < 1 ||
        c.max_output_tokens > 512 || c.max_output_tokens >= c.context_tokens ||
        c.queue_capacity < 1 || c.queue_capacity > 32 || c.http_threads < c.queue_capacity + 3 ||
        c.http_threads > 48 || c.event_capacity < 1 || c.event_capacity > 128 ||
        c.queue_timeout_ms < 10 || c.queue_timeout_ms > 300000 || c.generation_timeout_ms < 10 ||
        c.generation_timeout_ms > 600000 || c.backpressure_timeout_ms < 10 ||
        c.backpressure_timeout_ms > 30000 || c.mock_delay_ms < 0 || c.mock_delay_ms > 1000)
        throw std::runtime_error("config value outside supported resource bounds");
    if (c.engine == "llama" && c.model_path.empty())
        throw std::runtime_error("model_path required");
    return c;
}

// 模块：任务初始化：保存不可变请求及配置快照
Job::Job(std::string value, Request req, const Config& cfg)
    : id(std::move(value)), request(std::move(req)), config(cfg) {
}

// 模块：任务取消：设置原因并通知等待者
void Job::cancel(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!done && !cancelled.load()) {
        cancel_reason = reason;
        cancelled = true;
    }
    cv.notify_all();
}

// 模块：执行期限：协作式检查取消或生成超时
// 生成期限从started算，排队期限从submitted算，二者不会混成一个超时。
// 协作式取消：引擎在检查点调用此函数，不是强杀正在运行的线程。
bool Job::stopped() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!done && running &&
        Clock::now() - started >= std::chrono::milliseconds(config.generation_timeout_ms) &&
        !cancelled.load()) {
        cancel_reason = "generation_timeout";
        cancelled = true;
        cv.notify_all();
    }
    return cancelled.load();
}

// 模块：文本投递：写入有界缓冲并限制慢消费者等待
bool Job::emit(const std::string& text) {
    if (stopped())
        return false;
    std::unique_lock<std::mutex> lock(mutex);
    // 客户端读得慢时，有限事件缓冲会填满；最多等背压期限与执行期限中较早的一个。
    // wait_until睡眠期间释放Job锁，消费者next()取走事件后通知生产者继续。
    const auto until =
        std::min(Clock::now() + std::chrono::milliseconds(config.backpressure_timeout_ms),
                 started + std::chrono::milliseconds(config.generation_timeout_ms));
    if (!cv.wait_until(lock, until, [&] {
            return events.size() < size_t(config.event_capacity) || cancelled.load();
        })) {
        cancel_reason =
            Clock::now() >= started + std::chrono::milliseconds(config.generation_timeout_ms)
                ? "generation_timeout"
                : "slow_consumer";
        cancelled = true;
    }
    if (cancelled.load())
        return false;
    // first是首个完整文本片段入队的时刻，不是客户端真正收到数据的时刻。
    if (first == Clock::time_point{})
        first = Clock::now();
    events.push_back({{"type", "delta"}, {"request_id", id}, {"text", text}});
    cv.notify_all();
    return true;
}

// 模块：任务终结：仅写入一次独立终态并唤醒消费者
void Job::finish(const Json& event) {
    std::lock_guard<std::mutex> lock(mutex);
    // 一个Job只写一次终态，避免异常、退出等路径重复结束同一个请求。
    if (done)
        return;
    terminal = event;
    terminal["request_id"] = id;
    done = true;
    finished = Clock::now();
    cv.notify_all();
}

// 模块：事件消费：先取文本，最后取终态，结束后返回空值
std::optional<Json> Job::next() {
    std::unique_lock<std::mutex> lock(mutex);
    while (events.empty() && !done)
        cv.wait_for(lock, std::chrono::milliseconds(50));
    // 先读完已产生的delta，再读一次终态；随后nullopt表示事件流结束。
    if (!events.empty()) {
        auto e = std::move(events.front());
        events.pop_front();
        cv.notify_all();
        return e;
    }
    if (!terminal_read) {
        terminal_read = true;
        return terminal;
    }
    return std::nullopt;
}

// 模块：排队等待：在等待期限内等到worker开始或任务结束
bool Job::await_start() {
    std::unique_lock<std::mutex> lock(mutex);
    const auto deadline = submitted + std::chrono::milliseconds(config.queue_timeout_ms);
    if (!cv.wait_until(lock, deadline, [&] {
            return running || done;
        })) {
        cancel_reason = "queue_timeout";
        cancelled = true;
        cv.notify_all();
        return false;
    }
    return running;
}

// 模块：调度初始化：接管引擎并启动唯一worker
Scheduler::Scheduler(Config c, std::unique_ptr<Engine> engine)
    : config_(std::move(c)), engine_(std::move(engine)) {
    worker_ = std::thread(&Scheduler::run, this);
}

// 模块：调度清理：析构前停止任务并回收worker
Scheduler::~Scheduler() {
    stop();
}

// 模块：任务接入：清理过期等待者并执行容量检查
std::shared_ptr<Job> Scheduler::submit(Request r) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 接收新任务前清理已取消/过期的等待者，防止无效Job一直占据容量。
    for (auto it = queue_.begin(); it != queue_.end();) {
        auto& j = *it;
        if (j->cancelled.load() ||
            Clock::now() - j->submitted >= std::chrono::milliseconds(config_.queue_timeout_ms)) {
            if (!j->cancelled.load())
                j->cancel("queue_timeout");
            std::string reason;
            {
                std::lock_guard<std::mutex> job_lock(j->mutex);
                reason = j->cancel_reason;
            }
            j->finish({{"type", "error"}, {"code", reason}});
            if (reason == "queue_timeout" || reason == "generation_timeout")
                ++timed_out_;
            else
                ++cancelled_;
            it = queue_.erase(it);
        } else
            ++it;
    }
    // 此容量只计等待队列，不包含已经取出的active任务；接入层另有限制。
    if (stopping_ || queue_.size() >= size_t(config_.queue_capacity)) {
        ++rejected_;
        return {};
    }
    auto job = std::make_shared<Job>(std::to_string(++sequence_), std::move(r), config_);
    queue_.push_back(job);
    ++accepted_;
    queue_peak_ = std::max(queue_peak_, queue_.size());
    cv_.notify_one();
    return job;
}

// 模块：推理循环：FIFO出队，在队列锁外生成并记录终态
void Scheduler::run() {
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] {
                return stopping_ || !queue_.empty();
            });
            if (stopping_ && queue_.empty())
                break;
            // 这里只取任务并更新active，离开作用域释放队列锁，再进行模型计算。
            job = queue_.front();
            queue_.pop_front();
            active_ = job;
        }
        {
            std::lock_guard<std::mutex> lock(job->mutex);
            if (Clock::now() - job->submitted >=
                    std::chrono::milliseconds(config_.queue_timeout_ms) &&
                !job->cancelled.load()) {
                job->cancel_reason = "queue_timeout";
                job->cancelled = true;
            }
            if (!job->cancelled.load()) {
                job->started = Clock::now();
                job->running = true;
            }
            job->cv.notify_all();
        }
        Json terminal;
        try {
            if (job->cancelled.load())
                throw std::runtime_error("cancelled before start");
            // 引擎通过回调向Job投递文本、检查停止；HTTP线程通过next()独立消费这些事件。
            auto g = engine_->generate(
                job->request,
                [&](const std::string& s) {
                    return job->emit(s);
                },
                [&] {
                    return job->stopped();
                });
            if (job->stopped())
                throw std::runtime_error("generation cancelled");
            terminal = {
                {"type", "done"},
                {"finish_reason", g.finish_reason},
                {"usage",
                 {{"prompt_tokens", g.prompt_tokens}, {"completion_tokens", g.output_tokens}}},
                {"timings",
                 {{"queue_ms", millis(job->submitted, job->started)},
                  {"execution_ms", millis(job->started, Clock::now())},
                  {"prefill_ms", g.prefill_ms},
                  {"decode_ms", g.decode_ms}}}};
            // queue_ms=提交到开始；first_token_ms=开始到首片段入队，不含排队和网络读取。
            if (job->first != Clock::time_point{})
                terminal["timings"]["first_token_ms"] = millis(job->started, job->first);
        } catch (const std::invalid_argument& e) {
            terminal = {{"type", "error"}, {"code", "invalid_request"}, {"message", e.what()}};
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(job->mutex);
            terminal = {{"type", "error"},
                        {"code", job->cancelled.load() ? job->cancel_reason : "engine_error"},
                        {"message", job->cancelled.load() ? job->cancel_reason : e.what()}};
        } catch (...) {
            terminal = {
                {"type", "error"}, {"code", "engine_error"}, {"message", "unknown engine failure"}};
        }
        job->finish(terminal);
        Json record;
        {
            std::lock_guard<std::mutex> lock(job->mutex);
            record = job->terminal;
            record["event"] = "request_finished";
            if (job->running) {
                record["queue_ms"] = millis(job->submitted, job->started);
                record["execution_ms"] = millis(job->started, job->finished);
                if (job->first != Clock::time_point{})
                    record["first_token_ms"] = millis(job->started, job->first);
            }
        }
        log(record);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (terminal["type"] == "done")
                ++completed_;
            else {
                auto code = terminal.value("code", "");
                if (code == "queue_timeout" || code == "generation_timeout")
                    ++timed_out_;
                else if (job->cancelled.load())
                    ++cancelled_;
                else
                    ++failed_;
            }
            active_.reset();
        }
    }
}

// 模块：退出处理：取消运行及等待任务，再等待worker结束
// 停止时取消active并给排队任务写终态、通知等待者，再join；这里不是排空所有任务。
// engine_在worker结束后才销毁，避免推理线程访问已释放的模型/context。
void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        if (active_)
            active_->cancel("shutdown");
        for (auto& j : queue_) {
            j->cancel("shutdown");
            j->finish({{"type", "error"}, {"code", "shutdown"}});
            ++cancelled_;
        }
        queue_.clear();
        cv_.notify_all();
    }
    if (worker_.joinable())
        worker_.join();
}

// 模块：指标快照：读取接入、完成、失败和当前容量状态
Json Scheduler::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"engine", engine_->name()},
            {"accepted", accepted_},
            {"rejected", rejected_},
            {"completed", completed_},
            {"failed", failed_},
            {"cancelled", cancelled_},
            {"timed_out", timed_out_},
            {"queue_depth", queue_.size()},
            {"queue_peak", queue_peak_},
            {"active", active_ ? 1 : 0},
            {"inference_workers", 1}};
}
}
