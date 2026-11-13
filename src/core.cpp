// LLMServe-Lite 本地推理服务 | src/core.cpp
// 实现 serve.hpp 里的 Config、Job 和 Scheduler。HTTP 线程提交 Job 后，可以继续接待别人；
// 一个推理 worker 从等待队列取出它，调用 Engine::generate，把文字片段写进该 Job 的事件队列。
// 一条队列存“还没开始的请求”，另一条存“这个请求已经生成、尚未被读走的文字”。


#include "serve.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace serve {


// 模块：计时辅助（millis）
// 作用：计算两个时间点之间相差多少毫秒，返回 double 保留小数。
// 通俗讲：Clock 是单调时钟，只增不减，不受系统改时间影响，量耗时最可靠。


double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}


// 模块：日志输出（log）
// 作用：给每条事件加上时间戳，并加锁后以 JSON 形式写到标准错误，保证多线程日志不会互相穿插。
// system_clock 记录人能看懂的墙钟时间；耗时统计则用 steady_clock，避免系统校时影响差值。
// 调用方约定不把提示词正文写入日志，避免泄露用户内容。


void log(const Json& event) {
    // 函数内静态锁：所有线程共用同一把
    static std::mutex mutex;
    auto record = event;
    // 毫秒级 Unix 时间戳
    record["timestamp_unix_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
    std::lock_guard<std::mutex> lock(mutex);
    // dump 参数：-1 不缩进、不转义非 ASCII、遇到非法 UTF-8 用替换字符而不是抛异常，防止日志本身崩溃
    std::cerr << record.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
}


// 模块：配置加载（Config::load）
// 作用：读取配置文件，检查字段是否合法、类型是否正确，并确保各项数值落在本项目的资源边界内。
// 通俗讲：像入职体检，任何一项不合格就拒绝启动，而不是带病运行。


Config Config::load(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open config: " + path);
    Json j;
    file >> j;
    if (!j.is_object())
        throw std::runtime_error("config must be an object");
    Config c;
    // 允许出现的配置项白名单
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
                                           "mock_delay_ms",
                                           "prefix_cache_tokens",
                                           "queued_output_budget"};
    for (auto it = j.begin(); it != j.end(); ++it) {
        // 拼错的或多余的配置项直接报错，避免"以为生效了其实被忽略"
        if (std::find(keys.begin(), keys.end(), it.key()) == keys.end())
            throw std::runtime_error("unknown config key: " + it.key());
        // 三项是字符串，其余必须是能放进 int32 的整数
        if (it.key() == "engine" || it.key() == "host" || it.key() == "model_path") {
            if (!it.value().is_string())
                throw std::runtime_error("config string required: " + it.key());
        } else if (!is_int(it.value()))
            throw std::runtime_error("config int32 required: " + it.key());
    }
    // 宏：配置文件里有该项就覆盖，没有就保留默认值。#key 把参数转成字符串
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
    READ(prefix_cache_tokens);
    READ(queued_output_budget);
#undef READ
    if (c.engine != "mock" && c.engine != "llama")
        throw std::runtime_error("engine must be mock or llama");
    // 只允许绑定本机回环地址，不对外网暴露
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
    if (c.queued_output_budget < 0 || c.queued_output_budget > 16384)
        throw std::runtime_error("queued_output_budget must be in 0..16384");
    // 前缀缓存的 token 数必须小于上下文，否则没有空间放新内容
    if (c.prefix_cache_tokens < 0 || c.prefix_cache_tokens >= c.context_tokens)
        throw std::runtime_error(
            "prefix_cache_tokens must be >= 0 and smaller than context_tokens");
    // 模拟引擎没有真实 KV 缓存，开前缀缓存没意义
    if (c.engine == "mock" && c.prefix_cache_tokens != 0)
        throw std::runtime_error("prefix cache requires the real llama engine");
    if (c.engine == "llama" && c.model_path.empty())
        throw std::runtime_error("model_path required");
    return c;
}


// 模块：任务初始化（Job 构造函数）
// 作用：保存不可变的请求内容和配置快照。用 move 避免拷贝大字符串。


Job::Job(std::string value, Request req, const Config& cfg)
    : id(std::move(value)), request(std::move(req)), config(cfg) {
}


// 模块：任务取消（Job::cancel）
// 作用：记录取消原因、置取消标志，并唤醒所有在等待的线程。
// 只有"尚未结束且尚未被取消"时才写入原因，所以第一个取消原因会被保留，不会被后来者覆盖。


void Job::cancel(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!done && !cancelled.load()) {
        cancel_reason = reason;
        cancelled = true;
    }
    cv.notify_all();
}


// 模块：执行期限检查（Job::stopped）
// 作用：协作式检查"是否被取消"或"生成是否超时"。
// 生成期限从 started 算，排队期限从 submitted 算，二者不会混成一个超时。
// 协作式取消：引擎在检查点调用此函数，不是强杀正在运行的线程。
// 通俗讲：不是硬把人拖走，而是每隔一会儿问一句"该停了吗"，引擎自己体面地收工。


bool Job::stopped() {
    std::lock_guard<std::mutex> lock(mutex);
    // 正在运行且超过生成时限，并且还没被取消，就标记为超时取消
    if (!done && running &&
        Clock::now() - started >= std::chrono::milliseconds(config.generation_timeout_ms) &&
        !cancelled.load()) {
        cancel_reason = "generation_timeout";
        cancelled = true;
        cv.notify_all();
    }
    return cancelled.load();
}


// 模块：文本投递（Job::emit）
// 作用：引擎产出文本后放入有界缓冲，缓冲满时限时等待，防止慢客户端拖垮服务。
// 返回 true 表示投递成功，false 表示应当停止生成。


bool Job::emit(const std::string& text) {
    if (stopped())
        return false;
    std::unique_lock<std::mutex> lock(mutex);
    // 客户端读得慢时，有限事件缓冲会填满；最多等背压期限与执行期限中较早的一个。
    // wait_until睡眠期间释放Job锁，消费者next()取走事件后通知生产者继续。
    // 通俗讲：传送带满了就等一会儿，等太久就判定"消费者太慢"，不让它一直占着唯一的推理资源。
    const auto until =
        std::min(Clock::now() + std::chrono::milliseconds(config.backpressure_timeout_ms),
                 started + std::chrono::milliseconds(config.generation_timeout_ms));
    // wait_until 返回 false 表示到期时条件仍不满足（缓冲还是满的，也没被取消）
    if (!cv.wait_until(lock, until, [&] {
            return events.size() < size_t(config.event_capacity) || cancelled.load();
        })) {
        // 区分原因：到的是总执行期限就记 generation_timeout，否则是 slow_consumer
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


// 模块：任务终结（Job::finish）
// 作用：写入唯一的终态事件（成功或错误），标记完成并唤醒消费者。


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


// 模块：事件消费（Job::next）
// 作用：HTTP 线程调用它取下一个事件：先取完所有文本片段，最后取一次终态，之后返回空表示流结束。


std::optional<Json> Job::next() {
    std::unique_lock<std::mutex> lock(mutex);
    // 没事件且没结束就等；每 50 毫秒醒一次，作为保险，防止错过通知而永久卡死
    while (events.empty() && !done)
        cv.wait_for(lock, std::chrono::milliseconds(50));
    // 先读完已产生的delta，再读一次终态；随后nullopt表示事件流结束。
    if (!events.empty()) {
        auto e = std::move(events.front());
        events.pop_front();
        // 腾出了空位，通知可能正阻塞在 emit 里的生产者
        cv.notify_all();
        return e;
    }
    if (!terminal_read) {
        terminal_read = true;
        return terminal;
    }
    return std::nullopt;
}


// 模块：排队等待（Job::await_start）
// 作用：在排队期限内等待 worker 开始处理或任务提前结束。
// 返回 true 表示已开始执行；false 表示排队超时或被取消。


bool Job::await_start() {
    std::unique_lock<std::mutex> lock(mutex);
    // 排队期限从提交时刻算起
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


// 模块：调度初始化（Scheduler 构造函数）
// 作用：接管引擎，并启动唯一的推理 worker 线程。
// 注意：成员按声明顺序初始化，worker_ 在函数体里才启动，此时其他成员都已就绪。


Scheduler::Scheduler(Config c, std::unique_ptr<Engine> engine)
    : config_(std::move(c)), engine_(std::move(engine)) {
    worker_ = std::thread(&Scheduler::run, this);
}


// 模块：调度清理（Scheduler 析构函数）
// 作用：析构前先停止任务并回收 worker 线程，避免线程还在跑时对象已被销毁。


Scheduler::~Scheduler() {
    stop();
}


// 模块：任务接入（Scheduler::submit）
// 作用：先清理已取消或已超时的等待者，再做预算和容量检查，通过后创建任务入队。
// 返回空指针表示拒绝（队列满或预算不足）。


std::shared_ptr<Job> Scheduler::submit(Request r) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 接收新任务前清理已取消/过期的等待者，防止无效Job一直占据容量。
    for (auto it = queue_.begin(); it != queue_.end();) {
        auto& j = *it;
        if (j->cancelled.load() ||
            Clock::now() - j->submitted >= std::chrono::milliseconds(config_.queue_timeout_ms)) {
            // 过期但还没被标记取消的，补上 queue_timeout 原因
            if (!j->cancelled.load())
                j->cancel("queue_timeout");
            std::string reason;
            {
                // 加锁顺序：Scheduler 锁 → Job 锁，与全局约定一致
                std::lock_guard<std::mutex> job_lock(j->mutex);
                reason = j->cancel_reason;
            }
            // 给消费者写终态，让 HTTP 线程能得到明确的错误而不是一直等
            j->finish({{"type", "error"}, {"code", reason}});
            if (reason == "queue_timeout" || reason == "generation_timeout")
                ++timed_out_;
            else
                ++cancelled_;
            it = queue_.erase(it);  // erase 返回下一个有效迭代器
        } else
            ++it;
    }


    // 模块：等待计算预算
    // 作用：把所有排队请求的 max_tokens 加起来作为"排队债务"，超过预算就拒绝新请求。
    // 仍保持 FIFO，不推测 GPU 显存占用。


    int reserved_output = 0;
    for (const auto& pending : queue_)
        reserved_output += pending->request.max_tokens;
    // 预算为 0 表示关闭此功能
    if (config_.queued_output_budget &&
        reserved_output + r.max_tokens > config_.queued_output_budget) {
        ++rejected_;
        return {};
    }


    // 此容量只计等待队列，不包含已经取出的active任务；接入层另有限制。
    if (stopping_ || queue_.size() >= size_t(config_.queue_capacity)) {
        ++rejected_;
        return {};
    }
    // 请求编号用自增序列号
    auto job = std::make_shared<Job>(std::to_string(++sequence_), std::move(r), config_);
    queue_.push_back(job);
    ++accepted_;
    queue_peak_ = std::max(queue_peak_, queue_.size());
    // 叫醒 worker 来取任务
    cv_.notify_one();
    return job;
}


// 模块：推理循环（Scheduler::run）
// 作用：worker 线程的主循环：从队头取任务 → 在队列锁外执行生成 → 记录终态与统计。
// 通俗讲：唯一的厨师不停地从订单队列拿单子做菜，做菜时不占着订单台，别人仍可下单。


void Scheduler::run() {
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // 等到有任务或要停止
            cv_.wait(lock, [&] {
                return stopping_ || !queue_.empty();
            });
            // 停止且队列已空才退出循环
            if (stopping_ && queue_.empty())
                break;
            // 这里只取任务并更新active，离开作用域释放队列锁，再进行模型计算。
            job = queue_.front();
            queue_.pop_front();
            active_ = job;
        }
        {
            std::lock_guard<std::mutex> lock(job->mutex);
            // 取出时发现已等太久，直接标记超时，不再开始计算
            if (Clock::now() - job->submitted >=
                    std::chrono::milliseconds(config_.queue_timeout_ms) &&
                !job->cancelled.load()) {
                job->cancel_reason = "queue_timeout";
                job->cancelled = true;
            }
            // 没被取消才算正式开始
            if (!job->cancelled.load()) {
                job->started = Clock::now();
                job->running = true;
            }
            // 唤醒在 await_start 里等待的 HTTP 线程
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
            // 生成虽返回，但若中途被取消或超时，不能当作成功
            if (job->stopped())
                throw std::runtime_error("generation cancelled");
            // 构造成功终态：结束原因、token 用量、各阶段耗时
            terminal = {{"type", "done"},
                        {"finish_reason", g.finish_reason},
                        {"usage",
                         {{"prompt_tokens", g.prompt_tokens},
                          {"completion_tokens", g.output_tokens},
                          {"cached_prompt_tokens", g.cached_prompt_tokens}}},
                        {"timings",
                         {{"queue_ms", millis(job->submitted, job->started)},
                          {"execution_ms", millis(job->started, Clock::now())},
                          {"prefill_ms", g.prefill_ms},
                          {"decode_ms", g.decode_ms},
                          {"prompt_prepare_ms", g.prompt_prepare_ms},
                          {"prefill_compute_ms", g.prefill_compute_ms}}}};
            // queue_ms=提交到开始；first_token_ms=开始到首片段入队，不含排队和网络读取。
            if (job->first != Clock::time_point{})
                terminal["timings"]["first_token_ms"] = millis(job->started, job->first);
        } catch (const std::invalid_argument& e) {
            // 引擎认为请求非法
            terminal = {{"type", "error"}, {"code", "invalid_request"}, {"message", e.what()}};
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(job->mutex);
            // 如果是被取消的，错误码用取消原因（如超时、断连）；否则是引擎自身错误
            terminal = {{"type", "error"},
                        {"code", job->cancelled.load() ? job->cancel_reason : "engine_error"},
                        {"message", job->cancelled.load() ? job->cancel_reason : e.what()}};
        } catch (...) {
            // 兜底：捕获非标准异常，保证 worker 线程不会因此崩溃
            terminal = {
                {"type", "error"}, {"code", "engine_error"}, {"message", "unknown engine failure"}};
        }
        // 写入终态，唤醒等待的 HTTP 线程
        job->finish(terminal);
        // 组装一条完成日志，包含终态和各项耗时
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
            if (terminal["type"] == "done") {
                ++completed_;


                // 模块：成功耗时归档
                // 作用：把成功请求的各阶段耗时放进直方图，并累计 token 数。
                // 每个请求只统计一次；失败、取消和超时由独立计数表示，不混进耗时分布。


                const auto& t = terminal.at("timings");
                // 毫秒转秒，符合 Prometheus 习惯
                queue_seconds_.observe(t.at("queue_ms").get<double>() / 1000);
                execution_seconds_.observe(t.at("execution_ms").get<double>() / 1000);
                prefill_seconds_.observe(t.at("prefill_compute_ms").get<double>() / 1000);
                decode_seconds_.observe(t.at("decode_ms").get<double>() / 1000);
                if (t.contains("first_token_ms"))
                    first_output_seconds_.observe(t.at("first_token_ms").get<double>() / 1000);
                const auto& usage = terminal.at("usage");
                prompt_tokens_ += usage.at("prompt_tokens").get<uint64_t>();
                output_tokens_ += usage.at("completion_tokens").get<uint64_t>();
                cached_prompt_tokens_ += usage.at("cached_prompt_tokens").get<uint64_t>();


            } else {
                // 失败分类计数：超时、被取消、引擎故障互不重叠
                auto code = terminal.value("code", "");
                if (code == "queue_timeout" || code == "generation_timeout")
                    ++timed_out_;
                else if (job->cancelled.load())
                    ++cancelled_;
                else
                    ++failed_;
            }
            // 任务处理完毕，清除当前活动任务
            active_.reset();
        }
    }
}


// 模块：退出处理（Scheduler::stop）
// 作用：取消正在运行的任务和所有排队任务，然后等待 worker 线程结束。
// 停止时取消active并给排队任务写终态、通知等待者，再join；这里不是排空所有任务。
// engine_在worker结束后才销毁，避免推理线程访问已释放的模型/context。


void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        // 正在跑的任务：发取消信号，引擎下次检查时会自行停下
        if (active_)
            active_->cancel("shutdown");
        // 排队中的任务：直接取消并写终态，让等待的 HTTP 线程立刻得到 shutdown 响应
        for (auto& j : queue_) {
            j->cancel("shutdown");
            j->finish({{"type", "error"}, {"code", "shutdown"}});
            ++cancelled_;
        }
        queue_.clear();
        cv_.notify_all();
    }
    // 在锁外 join，否则 worker 想拿锁就会死锁；joinable 判断保证重复调用 stop 也安全
    if (worker_.joinable())
        worker_.join();
}


// 模块：指标快照（Scheduler::metrics）
// 作用：在加锁状态下读取接入、完成、失败、队列深度等计数，打包成 JSON 返回给 /metrics。


Json Scheduler::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // 当前排队请求预留的输出 token 总量
    int reserved_output = 0;
    for (const auto& job : queue_)
        reserved_output += job->request.max_tokens;
    return {{"engine", engine_->name()},
            {"accepted", accepted_},
            {"rejected", rejected_},
            {"completed", completed_},
            {"failed", failed_},
            {"cancelled", cancelled_},
            {"timed_out", timed_out_},
            {"queue_depth", queue_.size()},
            {"queued_output_tokens", reserved_output},
            {"queued_output_budget", config_.queued_output_budget},
            {"queue_peak", queue_peak_},
            {"active", active_ ? 1 : 0},
            {"inference_workers", 1},
            {"prompt_tokens", prompt_tokens_},
            {"output_tokens", output_tokens_},
            {"cached_prompt_tokens", cached_prompt_tokens_},
            {"histograms",
             {{"queue", queue_seconds_.json()},
              {"execution", execution_seconds_.json()},
              {"first_output", first_output_seconds_.json()},
              {"prefill_compute", prefill_seconds_.json()},
              {"decode", decode_seconds_.json()}}}};
}
}
