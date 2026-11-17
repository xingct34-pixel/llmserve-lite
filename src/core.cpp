// LLMServe-Lite 本地推理服务 | src/core.cpp
// 功能：实现 serve.hpp 里声明的三样东西：Config 的加载与检查、Job（单个请求的任务状态）、Scheduler（排队与执行）。
//       Scheduler 安排哪个请求先算，Job 把该请求的生成结果交给 HTTP 层。
// 目的：把"等模型空闲"和"等客户端接收"分开：
//       HTTP 处理线程提交任务后，仍要等任务启动和首个事件，但模型计算只由唯一的 worker 执行。
//
// 两条队列、两种等待：
//   等待队列（Scheduler::queue_）：由 HTTP 处理线程提交，唯一的推理 worker 取任务。
//       空时 worker 睡眠；满时提交直接失败，提交者不会留在这里等名额。
//   事件队列（Job::events）：由 worker 写、HTTP 层读。
//       空且任务未结束时，读者等待；满时，写者限时等空位。
// 停机：清掉排队任务并写入错误终态，给运行中的任务发取消信号。
// 读取顺序：Job::next 先交付已有的文本片段，再交付一次终态，之后返回空。
//
// 阅读顺序：配置和日志 → Job 交接文本和终态 → Scheduler 接收与执行请求 → 停机与指标。
//
// 待核对（依赖 engine.cpp 和 observability.hpp，本文件里看不出来）：
//   1. 引擎是否按约定工作：定期调用 stop 回调；请求非法时抛出 std::invalid_argument
//   2. Histogram::observe 的桶划分和线程安全：run 里是在 Scheduler 锁内调用它的


#include "serve.hpp"  // HTTP、调度器与生成引擎共用的配置、请求和任务类型
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace serve {


// 模块：计时辅助（millis）
// 功能：计算两个时间点之间相差多少毫秒，返回 double 以保留小数。
// 目的：统一耗时的计算方式。
// 通俗讲：Clock 是单调时钟，只增不减，不受系统改时间影响，量耗时最可靠。
double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}


// 模块：日志输出（log）
// 功能：给每条事件加上时间戳，并加锁后以 JSON 形式写到标准错误。
// 目的：保证多线程日志不会互相穿插，每条日志占一行，方便按行解析。
// 说明：时间戳用 system_clock，记录人能看懂的墙钟时间；耗时统计则用 steady_clock，避免系统校时影响差值。
//       调用方约定不把提示词正文写入日志，避免泄露用户内容。
void log(const Json& event) {
    // 函数内静态锁：所有线程共用同一把
    static std::mutex mutex;
    auto record = event;
    // 毫秒级 Unix 时间戳
    record["timestamp_unix_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
    std::lock_guard<std::mutex> lock(mutex);
    // dump 参数依次是：-1 不缩进、缩进字符、false 不转义非 ASCII（中文原样输出）、
    // 遇到非法 UTF-8 用替换字符而不是抛异常，防止日志本身崩溃
    std::cerr << record.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
}


// 模块：配置加载（Config::load）
// 功能：读取配置文件，检查字段是否合法、类型是否正确，并确保各项数值落在本项目的资源边界内。
// 目的：任何一项不合格就拒绝启动，而不是带着错误配置运行。
// 通俗讲：像入职体检，一项不合格就不让上岗。
// 说明：这里抛出的异常（包括 JSON 本身写错时 nlohmann 抛出的解析异常）都会在 main 里被接住，
//       记成 startup_error 并以退出码 1 结束，tests/integration.py 里的配置测试检查的就是这个行为。
Config Config::load(const std::string& path) {
    std::ifstream file(path);


    // 读取配置
    // 先读 JSON，再按字段检查类型和范围。拼错字段名要在启动时暴露，否则服务可能带着默认值运行。
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
        // 三项是字符串，其余必须是能放进 int32 的整数（小数、过大的数都会在这里被拒绝）
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
    // 这些是本项目面向低内存机器的保守边界，不是 llama.cpp 或模型的通用极限：
    //   端口 1024 到 65535（避开需要特权的端口）；计算线程 1 到 6；上下文 128 到 2048；
    //   单次输出 1 到 512，且必须小于上下文长度；等待队列容量 1 到 32；
    //   HTTP 线程至少比等待容量多 3、最多 48，配合接入上限给健康检查和拒绝响应留出处理余量；
    //   事件缓冲 1 到 128；各类超时有各自的上下限；模拟延迟 0 到 1000 毫秒。
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
// 功能：保存不可变的请求内容和配置快照。
// 说明：参数用 move 转移进来，避免拷贝大字符串（比如很长的提示词）。
Job::Job(std::string value, Request req, const Config& cfg)
    : id(std::move(value)), request(std::move(req)), config(cfg) {
}


// 模块：任务取消（Job::cancel）
// 功能：记录取消原因、置取消标志，并通知等待这个 Job 的线程。
// 目的：多个线程都可能要求停止同一个请求，所以只有"尚未结束且尚未被取消"时才写入原因，
//       先发生的取消原因会被保留，后续的错误才有可解释的来源。
// 说明：被唤醒的线程仍要各自检查自己的等待条件。
void Job::cancel(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!done && !cancelled.load()) {
        cancel_reason = reason;
        cancelled = true;
    }
    cv.notify_all();
}


// 模块：执行期限检查（Job::stopped）
// 功能：协作式检查"是否被取消"或"生成是否超时"，返回 true 表示应当停止。
// 目的：生成期限从 started 算，排队期限从 submitted 算，二者不会混成一个超时。
// 说明：这是协作式取消：引擎在检查点调用此函数，不是强杀正在运行的线程。
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
// 功能：引擎产出文本后放入有界缓冲，缓冲满时限时等待，防止慢客户端拖垮服务。
// 目的：返回 true 表示投递成功，false 表示应当停止生成。
// 通俗讲：传送带满了就等一会儿，等太久就判定"消费者太慢"，不让它一直占着唯一的推理资源。
// 例：event_capacity=2，先写 A、B 就占满了。worker 要写 C 时，要等读者取走一段再继续；
//     若一直没有空位，等待最多到背压期限或生成期限中较早的一个。
bool Job::emit(const std::string& text) {
    if (stopped())
        return false;
    std::unique_lock<std::mutex> lock(mutex);
    // 最多等背压期限与生成期限中较早的一个
    const auto until =
        std::min(Clock::now() + std::chrono::milliseconds(config.backpressure_timeout_ms),
                 started + std::chrono::milliseconds(config.generation_timeout_ms));


    // 慢客户端
    // worker 产出片段比客户端读得快时，events 会填满。
    // wait_until 睡眠期间会释放 Job 的锁，读线程才能进来取走片段、腾出空位；
    // 消费者在 next() 里取走事件后会通知这里继续。
    // wait_until 返回 false 表示到期时条件仍不满足（缓冲还是满的，也没被取消）。
    if (!cv.wait_until(lock, until, [&] {
            return events.size() < size_t(config.event_capacity) || cancelled.load();
        })) {
        // 区分原因：到的是总生成期限就记 generation_timeout，否则是客户端读得太慢 slow_consumer
        cancel_reason =
            Clock::now() >= started + std::chrono::milliseconds(config.generation_timeout_ms)
                ? "generation_timeout"
                : "slow_consumer";
        cancelled = true;
    }
    if (cancelled.load())
        return false;


    // 首段时间
    // first 是第一个完整文本片段"入队"的时刻，不是客户端真正收到数据的时刻。
    // 它只在第一次投递时记录；后面再来十个片段也不能覆盖它，否则首段延迟会变成最后一段延迟。
    if (first == Clock::time_point{})
        first = Clock::now();
    events.push_back({{"type", "delta"}, {"request_id", id}, {"text", text}});
    cv.notify_all();
    return true;
}


// 模块：任务终结（Job::finish）
// 功能：写入唯一的终态事件（成功或错误），标记完成并唤醒消费者。
// 目的：一个 Job 只写一次终态，避免异常、退出等多条路径重复结束同一个请求。
void Job::finish(const Json& event) {
    std::lock_guard<std::mutex> lock(mutex);
    if (done)
        return;
    terminal = event;
    terminal["request_id"] = id;
    done = true;
    finished = Clock::now();
    cv.notify_all();
}


// 模块：事件消费（Job::next）
// 功能：HTTP 线程调用它取下一个事件：先取完所有文本片段，最后取一次终态，之后返回空表示流结束。
// 目的：读取顺序可以是 A → B → done → 空。
//       finish 把终态放在独立字段 terminal 里，所以写入终态不需要先挤掉 A 或 B；
//       最后一个 delta 已生成，并不等于客户端已经消费了它，所以必须先读完文本再读终态。
std::optional<Json> Job::next() {
    std::unique_lock<std::mutex> lock(mutex);
    // 有通知，或等满 50 毫秒，就重新检查 events 和 done。
    // 单独调用 cancel() 不会让这里返回终态，要等 finish() 设置 done
    while (events.empty() && !done)
        cv.wait_for(lock, std::chrono::milliseconds(50));
    // 先读剩余文本
    if (!events.empty()) {
        auto e = std::move(events.front());
        events.pop_front();
        // 腾出了空位，通知可能正阻塞在 emit 里的生产者
        cv.notify_all();
        return e;
    }
    // 终态只读取一次。已经取过终态后，next 返回空，告知调用者结束循环
    if (!terminal_read) {
        terminal_read = true;
        return terminal;
    }
    return std::nullopt;
}


// 模块：排队等待（Job::await_start）
// 功能：在排队期限内等待 worker 开始处理，或任务提前结束。
// 目的：返回 true 表示 worker 已开始执行；返回 false 表示没等到，原因记在 cancel_reason 里
//       （排队超时，或在开始前就已结束，如服务关闭）。
// 说明：等待条件是 running 或 done。到期仍未满足就标记排队超时；若先收到终态，则用 running 判断任务是否真的开始过。
//       排队超时时，这个 Job 仍留在 queue_ 里，由之后的 submit 清理，或 worker 取出时发现后收尾。
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
// 功能：接管配置和引擎，并启动唯一的推理 worker 线程。
// 说明：成员按声明顺序初始化，worker_ 在函数体里才启动，此时其他成员都已就绪，线程不会读到半成品。
Scheduler::Scheduler(Config c, std::unique_ptr<Engine> engine)
    : config_(std::move(c)), engine_(std::move(engine)) {
    worker_ = std::thread(&Scheduler::run, this);
}


// 模块：调度清理（Scheduler 析构函数）
// 功能：析构前先停止任务并回收 worker 线程。
// 目的：避免线程还在跑时对象已被销毁。stop() 可以重复调用，main 里已经调用过的话，这里不会出问题。
Scheduler::~Scheduler() {
    stop();
}


// 模块：任务接入（Scheduler::submit）
// 功能：先清理已取消或已超时的等待者，再做预算和容量检查，通过后创建任务入队。
// 目的：容量已满、排队输出预算不足、或正在停机，都会返回空指针；提交者不会留在这里等名额。
// 说明：看容量时只数仍在 queue_ 中的任务。例：容量 2，一个正在算、两个在等，此时队列深度是 2；
//       正在算的那个由 active_ 单独记录。
std::shared_ptr<Job> Scheduler::submit(Request r) {
    std::lock_guard<std::mutex> lock(mutex_);


    // 清理排队任务
    // 接收新任务前，先清理已取消或过期的等待者，防止无效的 Job 一直占着容量。
    // erase 之后原来的迭代器失效，要用它的返回值继续前进，才能检查紧邻的下一个过期任务。
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


    // 等待计算预算
    // 功能：把所有排队请求的 max_tokens 加起来作为"排队债务"，超过预算就拒绝新请求。
    // 目的：仍保持先进先出，也不去推测显存占用，只限制排队负载。
    // 例：预算 100，队里已预留 80，新请求要 32。80+32 超出预算，即使队列还有空位也拒绝；
    //     预算配置为 0 则跳过这项检查。
    int reserved_output = 0;
    for (const auto& pending : queue_)
        reserved_output += pending->request.max_tokens;
    if (config_.queued_output_budget &&
        reserved_output + r.max_tokens > config_.queued_output_budget) {
        ++rejected_;
        return {};
    }


    // 容量检查
    // 这里的容量只计等待队列，不包含已经取出的 active 任务；接入层（main.cpp 里的 handling）另有限制。
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
// 功能：worker 线程的主循环：从队头取任务 → 在队列锁外执行生成 → 记录终态与统计。
// 目的：做菜时不占着订单台，别的请求仍能提交。
// 通俗讲：唯一的厨师不停地从订单队列拿单子做菜，做菜时不占着订单台，别人仍可下单。
void Scheduler::run() {


    // 单 worker 循环
    // 先在队列锁内取一个任务，再离开队列临界区去跑推理；其他 HTTP 线程这时仍可提交任务。
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
            // 这里只取任务并更新 active_，离开作用域就释放队列锁，之后才进行模型计算
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
            // 两个回调把引擎和任务接起来：
            //   引擎每产出一段文本，就通过第一个回调投递给 job；
            //   生成过程中，引擎通过第二个回调查询"是否该停"。
            // HTTP 线程则通过 next() 独立地消费这些事件。
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
            // queue_ms 是提交到开始的时间；first_token_ms 是开始到首片段入队的时间，不含排队和网络读取
            if (job->first != Clock::time_point{})
                terminal["timings"]["first_token_ms"] = millis(job->started, job->first);
        } catch (const std::invalid_argument& e) {
            // 引擎认为请求非法（比如提示词不符合要求）：对外会变成 400
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
        // 组装一条完成日志，包含终态和各项耗时（不含提示词）
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


            // 成功与失败分开统计
            // 出错的请求可能没有首段文本和完整的 decode 数据，不能用 0 填进成功延迟分布。
            if (terminal["type"] == "done") {
                ++completed_;


                // 成功耗时归档
                // 功能：把成功请求的各阶段耗时放进直方图，并累计 token 数。
                // 目的：每个请求只统计一次；失败、取消和超时由独立计数表示，不混进耗时分布。
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
// 功能：取消正在运行的任务和所有排队任务，然后等待 worker 线程结束。
// 目的：有序关闭：先取消、再 join，不是强杀线程，也不是等所有任务排空。
// 说明：engine_ 要等 worker 结束后才会销毁，避免推理线程访问已释放的模型或上下文。
//       stop 可以重复调用：第二次调用时队列已空，worker_ 也已回收，不会重复计数或重复 join。
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
    // 要先唤醒等待的线程、发出停止，离开队列锁之后再 join：
    // worker 退出前还要再次取队列锁，如果这里持锁 join，就会互相等待。
    // join 只是等线程返回，本身不会把睡眠中的线程叫醒。
    if (worker_.joinable())
        worker_.join();
}


// 模块：指标快照（Scheduler::metrics）
// 功能：在加锁状态下读取接入、完成、失败、队列深度等计数，打包成 JSON 返回给 /metrics。
// 说明：queue_depth 直接数 queue_ 里的任务，已经超时但还没被清理的 Job 也会算进去，
//       它们会在下一次 submit 的清理或 worker 取出时被移除。
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
}  // namespace serve
