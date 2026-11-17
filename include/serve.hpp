// LLMServe-Lite 本地推理服务 | include/serve.hpp
// 功能：LLMServe-Lite 把本地语言模型包装成 HTTP 服务。这个头文件主要声明接口，也含少量内联实现，
//       定义接入、调度、生成之间约定的数据结构，相当于整套系统的"设计图"。
// 目的：约定一次请求从 HTTP、等待队列、引擎到响应端传递哪些数据，以及谁持有任务状态。
//
// 阅读顺序：配置与请求格式 → 引擎接口 → 单请求的事件与终态（Job） → 多请求的排队和统计（Scheduler）。
//
// 文件分工：
//   serve.hpp   本文件，声明所有核心类型
//   core.cpp    Config 加载，Job 和 Scheduler 的实现
//   engine.cpp  MockEngine（模拟引擎）和 LlamaEngine（llama.cpp 真实引擎）
//   main.cpp    HTTP 路由，启动和退出
//
// 四类核心对象：
//   Config     集中管理线程数、队列容量、超时和上下文长度等资源参数
//   Engine     统一的引擎抽象接口，模拟引擎和真实引擎都按它实现
//   Job        一个请求的生命周期，内部有容量固定的事件缓冲区，用 mutex 和 condition_variable 做线程同步，
//              支持协作式取消（引擎自己定期检查是否该停）和背压（客户端读得慢时，生成端要等一等）
//   Scheduler  有界 FIFO 调度器：固定只有 1 个推理 worker，其余请求排队，队列满了直接拒绝，
//              并用 Histogram 的固定桶统计排队、首字、prefill、decode 等耗时
//
// 一个请求的流转路径：
//   HTTP 线程收到请求并校验 → 交给 Scheduler 排队 → 唯一的 worker 线程取出，调用 Engine 生成文字
//   → 文字放进 Job 的缓冲区 → HTTP 线程从同一个 Job 里取出 → 返回给客户端。
//
// 通俗地说，这是一家只有一个厨师的餐厅：
//   HTTP 线程是服务员，数量多，负责接待客人和端菜；
//   Scheduler 是座位有限的订单队列，满了就拒绝新客人；
//   Engine 是厨师，同一时刻只做一道菜；
//   Job 是一张订单，同时也是出菜口：厨师做一点放一点，服务员取一点端一点，这就是 SSE 流式输出
//   （SSE：服务器一段一段主动推送文字，所以客户端能边生成边显示）。
//
// 设计取舍：选择单 worker，是为了复用一个模型上下文并限制内存占用；多 worker 是否更快，要在目标硬件上实测。
// 整体思路是"处处有界，过载时拒绝而不是崩溃"。
//
// 待核对（头文件里看不出来，需要对照实现确认）：
//   1. "限定在低内存机器可承受的范围内"：范围限制是否真的写在 Config::load（core.cpp）里
//   2. 两把锁的顺序、推理时不持有 Scheduler 锁：是否与 core.cpp 的实际写法一致
//   3. 直方图是否只统计成功的请求：看 Scheduler::run
//   4. prompt_prepare_ms 具体指分词还是套模板：看 engine.cpp


#pragma once
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include "observability.hpp"  // 直方图与 Prometheus 指标文本的公共接口

namespace serve {
// Json 是 nlohmann 库的类型别名；Clock 选用 steady_clock，它只会单调递增，
// 不受系统改时间影响，适合测耗时和做超时判断
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;


// 模块：字段校验（is_int）
// 功能：判断一个 JSON 值是不是整数，并且能否安全放进 32 位 int。
// 目的：防止过大的数字溢出后变成错误的值。
// 通俗讲：用户传来的数字可能大得离谱（比如 max_tokens 填一个 100 位的数），先检查装不装得下。
inline bool is_int(const Json& j) {
    // 不是整数（比如小数、字符串）直接不合格
    if (!j.is_number_integer())
        return false;
    // 无符号整数不可能为负，只需检查上限；
    // 并且要先和 int 的上界比较再转换，大数若先转成 int，可能已经失去原值，范围检查就没意义了。
    if (j.is_number_unsigned())
        return j.get<uint64_t>() <= uint64_t(std::numeric_limits<int>::max());
    // 有符号整数要同时检查上下限
    const auto n = j.get<int64_t>();
    return n >= std::numeric_limits<int>::min() && n <= std::numeric_limits<int>::max();
}


// 模块：服务配置（Config）
// 功能：集中声明计算、队列、HTTP 和超时相关的所有参数，并给出默认值。
// 目的：让所有资源上限都有明确的数字，过载时按这些数字拒绝，而不是无限占用内存。
// 资源参数分三层：
//   threads       模型做计算用的 CPU 线程数
//   http_threads  负责接待网络连接的线程数
//   推理 worker   固定只有 1 个，同一时刻只跑一个请求（不是配置项）
// queue_capacity 限制"排队等待推理"的请求数；event_capacity 限制每个请求"还没被客户端读走"的文本事件数。
// context_tokens 要装下"套完模板的提示词 + 预留的生成长度"；max_output_tokens 只限制新生成的部分。
// 通俗讲：threads 是厨师的手数，http_threads 是服务员数量，queue_capacity 是等位座位数。
struct Config {
    // engine：引擎类型（mock 为模拟引擎，方便测试）；host：监听地址，默认只允许本机访问；model_path：模型文件路径
    std::string engine = "mock", host = "127.0.0.1", model_path;
    // port：端口；threads：模型计算用的 CPU 线程数；
    // context_tokens：上下文窗口大小；max_output_tokens：单次最多生成多少 token
    int port = 8080, threads = 4, context_tokens = 1024, max_output_tokens = 128;
    // queue_capacity：等待队列容量，满了就拒绝新请求；http_threads：接待网络连接的线程数；
    // event_capacity：单请求的输出缓冲上限
    int queue_capacity = 4, http_threads = 8, event_capacity = 16;
    // queue_timeout_ms：排队最长等待时间；generation_timeout_ms：生成阶段最长执行时间
    int queue_timeout_ms = 30000, generation_timeout_ms = 120000;
    // backpressure_timeout_ms：客户端读得太慢导致缓冲写满时，最多等多久；mock_delay_ms：模拟引擎每步的人为延迟
    int backpressure_timeout_ms = 2000, mock_delay_ms = 20;


    // 可选前缀缓存
    // 功能：0 表示关闭；开启后，只复用本进程上一条成功请求中相同的 token 前缀。
    // 目的：省掉重复的 prefill 计算。
    // 通俗讲：如果两个请求开头一模一样（比如相同的系统提示词），开头那段就不用重新算了。
    int prefix_cache_tokens = 0;


    // 排队预算
    // 功能：限制所有"等待中"请求的 max_tokens 之和，0 表示关闭。
    // 说明：它只是排队负载的上限，不代表 KV 缓存占多少内存。
    int queued_output_budget = 0;


    // 从指定路径加载配置（实现在 core.cpp）
    static Config load(const std::string&);
};


// 模块：请求数据（Request）
// 功能：保存一次请求的提示词、输出长度上限，以及是否流式返回。
// 目的：Request 只存"这次要生成什么"。Job 在它外面再加提交时间、取消状态和输出队列，用来跨线程跟踪这次生成。
// 说明：HTTP 解析完字段就能构造 Request，填充代码在 main.cpp。
//       每个 HTTP 请求各有自己的 prompt 和 max_tokens，进程启动时的 Config 则被所有请求共享。
struct Request {
    std::string prompt;        // 用户提示词
    int max_tokens = 128;      // 本次最多生成多少个 token
    bool stream = false;       // true 则边生成边返回（SSE），false 则等全部生成完一次返回
};


// 模块：生成统计（Generation）
// 功能：保存 token 数量、结束原因和引擎各阶段耗时。
// 目的：引擎生成完后把这些数据交回去，Scheduler 据此更新累计指标。
struct Generation {
    int prompt_tokens = 0, output_tokens = 0;
    // 结束原因：length 表示达到长度上限，另外也可能是自然结束(stop)等
    std::string finish_reason = "length";
    // prefill 是"读完整个提示词"的阶段，decode 是"逐个 token 生成"的阶段
    double prefill_ms = 0, decode_ms = 0;


    // 真实计算口径
    // 功能：把"准备耗时""真正的 prefill 计算耗时""被缓存复用而没重算的输入 token 数"分开记录。
    // 目的：开了前缀缓存后，统计不会虚高或虚低。
    double prompt_prepare_ms = 0, prefill_compute_ms = 0;
    int cached_prompt_tokens = 0;
};


// 模块：引擎接口（Engine）
// 功能：定义统一的生成接口，模拟引擎和 llama.cpp 引擎都按此实现。
// 目的：Scheduler 通过这个接口调用生成，不用关心具体是哪种引擎。
// 通俗讲：像插座标准，不管插的是哪种电器，墙上的插座都一样。
// 说明：生成过程中，emit 用来交出一段完整文字，stop 用来查询是否该结束。
//       这两个动作以回调的形式传进来，引擎就能配合取消和慢客户端，而不必知道 HTTP 连接怎么写。
class Engine {
public:
    // 虚析构：保证通过基类指针删除子类对象时，子类的资源也能正确释放
    virtual ~Engine() = default;
    // generate 只会被唯一的推理 worker 线程调用。
    // 第二个参数 emit：每产出一段完整文本就调用它交出去，返回 false 表示下游不要了；
    // 第三个参数 stop：引擎要定期调用它，检查是否被取消或执行超时。
    // 注意：输出 token 数不等于文本事件数，因为一个字可能由多个 token 拼成。
    virtual Generation generate(const Request&,
                                const std::function<bool(const std::string&)>&,
                                const std::function<bool()>&) = 0;
    // 返回引擎名称，只读（待核对：HTTP 指标接口是否会读取它）
    virtual std::string name() const = 0;
};


// 模块：辅助接口（make_engine 等）
// 功能：声明引擎工厂、UTF-8 处理、日志和计时函数，具体实现在其他文件。
// 目的：把这些零散的工具函数集中放在一处，调用方只需包含本头文件。
// 根据配置创建对应的引擎（mock 或 llama.cpp）
std::unique_ptr<Engine> make_engine(const Config&);
// 取出缓冲区里"完整的 UTF-8 字符"。
// 通俗讲：例如"中"的 UTF-8 编码占 3 个字节，token 可能只输出了半个字，直接发出去会乱码，所以先攒够再发；
// flush 为 true 时表示结尾了，剩余的全部取出
std::string complete_utf8(std::string&, bool flush = false);
// 输出一条 JSON 格式的日志
void log(const Json&);
// 计算两个时间点之间相差多少毫秒
double millis(Clock::time_point a, Clock::time_point b);


// 模块：任务状态（Job）
// 功能：管理单个请求的整个生命周期：输出缓冲、取消、结束状态。
// 目的：HTTP 线程取事件的同时，worker 能在另一端继续放入事件，两边共享同一份状态。
// 说明：一个请求对应一个 Job，用 shared_ptr 管理，这样 HTTP 处理线程、推理 worker、SSE 回调
//       谁最后用完谁负责释放，不会出现某个线程还在用、对象却被销毁的情况。
//       cancelled 是原子变量，用来跨线程发"停止"信号；取消原因、运行状态和事件队列仍由 mutex 保护。
struct Job {
    const std::string id;                              // 请求唯一编号
    const Request request;                             // 请求内容，创建后不可修改
    const Config config;                               // 创建时拷贝一份配置，避免之后被改动
    const Clock::time_point submitted = Clock::now();  // 提交时刻，用来算排队时长
    // started：开始执行；finished：结束；first：产出第一段输出（用来算首字延迟）
    Clock::time_point started{}, finished{}, first{};
    std::atomic<bool> cancelled{false};                // 取消标志，无需加锁即可读写
    std::mutex mutex;                                  // 保护下面的状态和事件队列
    std::condition_variable cv;                        // 用来唤醒在等待事件或状态变化的线程
    // running：worker 已开始过这个任务，结束时不会清零；done：已结束；terminal_read：终态事件是否已被消费者读走
    bool running = false, done = false, terminal_read = false;
    std::deque<Json> events;                           // 待读取的文本增量事件
    // 终态单独存放，不挤占有限的 delta 缓冲，慢消费者也不会堵住 finish。
    // 通俗讲：就算缓冲区满了，"结束"这个消息也一定有地方放，不会丢
    Json terminal;
    std::string cancel_reason;                         // 取消原因，如超时、客户端断开
    explicit Job(std::string, Request, const Config&);
    void cancel(const std::string& reason);            // 请求取消
    bool stopped();                                    // 是否已被取消或超时
    // 引擎往缓冲区推送一段文本；缓冲满时最多等 backpressure_timeout_ms，超时就按取消处理
    bool emit(const std::string& text);
    void finish(const Json& event);                    // 标记结束并写入终态事件
    std::optional<Json> next();                        // 消费者取下一个事件，全部取完返回空
    bool await_start();                                // 等待任务开始执行
};


// 模块：调度器（Scheduler）
// 功能：管理有界等待队列、唯一的推理 worker 和运行指标。
// 目的：先到先服务，从队头取任务；HTTP 并发再高，也不会增加模型解码的 worker 数量。
// 加锁顺序约定：涉及两把锁时，先拿 Scheduler 锁，再拿 Job 锁，所有代码都遵守，才能避免死锁。
// 耗时的推理过程不持有 Scheduler 锁，否则别的请求连提交都会被卡住。（以上两条待核对 core.cpp）
class Scheduler {
    Config config_;
    std::unique_ptr<Engine> engine_;
    mutable std::mutex mutex_;                         // 保护队列和计数器；mutable 让 const 的 metrics() 也能加锁
    std::condition_variable cv_;                       // 有新任务或要停止时唤醒 worker
    std::deque<std::shared_ptr<Job>> queue_;           // 等待队列
    std::shared_ptr<Job> active_;                      // 当前正在执行的任务
    bool stopping_ = false;                            // 是否正在关闭
    // worker_ 的生命周期属于调度器；不是每到一条 HTTP 请求就在这里新建一个线程。
    std::thread worker_;                               // 唯一的推理 worker 线程
    // 各类累计计数：编号、接受、拒绝、完成、失败、取消、超时
    uint64_t sequence_ = 0, accepted_ = 0, rejected_ = 0, completed_ = 0, failed_ = 0,
             cancelled_ = 0, timed_out_ = 0;
    size_t queue_peak_ = 0;                            // 队列历史最高长度，用来判断容量够不够


    // 有界观测
    // 功能：用固定的"桶"累计请求的耗时分布，不保存每个请求的明细。
    // 目的：内存不会随运行时间无限增长。（是否只统计成功请求，待核对）
    // 通俗讲：不记每个人的成绩，只记"0~1 秒有几人、1~2 秒有几人"，占用空间永远固定。
    Histogram queue_seconds_, execution_seconds_, first_output_seconds_;
    Histogram prefill_seconds_, decode_seconds_;
    // 累计的提示词 token 数、输出 token 数、被缓存复用的 token 数
    uint64_t prompt_tokens_ = 0, output_tokens_ = 0, cached_prompt_tokens_ = 0;


    void run();                                        // worker 线程的主循环：取任务、执行、记录统计

public:
    Scheduler(Config, std::unique_ptr<Engine>);
    ~Scheduler();                                      // 析构时需停止并回收 worker 线程
    std::shared_ptr<Job> submit(Request);              // 提交请求；队列满时拒绝
    void stop();                                       // 停止调度器
    Json metrics() const;                              // 导出当前运行指标
};
}  // namespace serve
