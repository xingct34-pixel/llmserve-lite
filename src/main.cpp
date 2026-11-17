// LLMServe-Lite 本地推理服务 | src/main.cpp
// 功能：本地推理服务的 HTTP 入口。启动时读配置、创建一个引擎和调度器，之后才接收请求。
//       同一句提示词有两种回程：普通请求收齐文字后返回 JSON；流式请求通过 SSE 分段发送。
// 目的：把客户端请求转换成推理任务，并根据错误发生在发送响应头之前还是之后，选用合适的反馈方式：
//       之前能返回真实的 HTTP 状态码，之后只能在数据流里用事件告知。
//
// 阅读顺序：启动服务 → 注册健康和指标接口 → 接收请求并等首个事件 → 普通 JSON 或 SSE → 退出。
// 沿着一个 POST 请求往下读，会依次遇到三个交接点：提交 Job、等第一个事件、写给客户端。
//
// 待核对（本文件里看不出来，需要对照别的文件确认）：
//   1. httplib::ThreadPool 第二个参数的含义：新版库里是"最大线程数"，旧版库里是"最大排队请求数"，要看所用版本
//   2. http_threads 是否有下限保证（接入上限 queue_capacity+1 之外还要给健康、指标、拒绝响应留线程）：看 Config::load
//   3. Scheduler::stop 是否可以重复调用（监视线程和 main 各调用了一次），以及它是否取消 Job 并 join worker
//   4. submit 返回空指针的全部原因（队列满、预算不足、停机）和 await_start 返回 false 时的取消原因：看 core.cpp


#include "serve.hpp"  // HTTP、调度器与生成引擎共用的配置、请求和任务类型
#include <httplib/httplib.h>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>


// 模块：HTTP 辅助（信号标志、JSON 响应、错误码映射）
// 功能：提供处理 Ctrl+C 的停止标志、统一的 JSON 回复函数，以及把内部错误码翻译成 HTTP 状态码。
// 目的：后面的各个路由共用这几个小工具，回复格式和状态码保持一致。
namespace {
// 信号处理函数里只能安全地做很少的事，所以标志必须是"无锁"的原子变量，编译期检查这一点
static_assert(std::atomic<bool>::is_always_lock_free, "signal flag must be lock-free");
// 收到 SIGINT/SIGTERM 后置为 true，表示"请准备退出"
std::atomic<bool> interrupted{false};

// 信号回调只设置这个无锁原子标志；关闭 socket、取消任务和 join 线程都交给正常线程处理。
// 通俗讲：信号来得很突然，在里面做复杂操作容易出错，所以只"举个旗子"，真正的收尾交给普通线程。
void on_signal(int) {
    interrupted.store(true, std::memory_order_relaxed);
}

// 统一回复：设置状态码，并以 UTF-8 JSON 格式返回内容
void reply(httplib::Response& res, int status, const serve::Json& data) {
    res.status = status;
    res.set_content(data.dump(), "application/json; charset=utf-8");
}

// 把引擎/调度器的错误码映射成 HTTP 状态码
int error_status(const serve::Json& event) {
    auto code = event.value("code", "");
    if (code == "invalid_request")
        return 400;  // 请求本身有问题
    if (code == "queue_timeout" || code == "generation_timeout")
        return 504;  // 排队或生成超时
    if (code == "shutdown")
        return 503;  // 服务正在关闭
    return 500;      // 其余都当作内部错误
}
}  // namespace


// 模块：服务入口（main）
// 功能：加载配置和引擎，注册所有路由，监听本地端口，并在收到停止信号时有序退出。
// 目的：启动阶段任何一步失败都直接报错退出，不会出现"端口开了但模型没加载好"的半成品状态。
int main(int argc, char** argv) {
    try {
        // 必须恰好传入一个配置文件路径，否则打印用法并返回 2
        if (argc != 2) {
            std::cerr << "usage: llmserve configs/server.json\n";
            return 2;
        }


        // 服务初始化
        // 功能：加载引擎，设置 HTTP 线程池和资源限制。
        // 目的：先加载唯一的引擎，再启动单 worker；加载失败时不会监听端口，也不会宣称服务就绪。
        // 通俗讲：模型没加载好之前，服务不会对外说"我准备好了"。
        const auto config = serve::Config::load(argv[1]);
        serve::Scheduler scheduler(config, serve::make_engine(config));
        httplib::Server server;
        // 自定义 HTTP 线程池，线程数由配置决定
        server.new_task_queue = [&] {


            // HTTP 接入层
            // 线程池只负责处理连接和写响应，真正跑模型的是 Scheduler 的 worker。
            // 所以增大 HTTP 线程数，不会让模型多加载几份。
            // 两个参数这里都填 http_threads，含义取决于 httplib 版本（见文件头"待核对"第 1 条）
            return new httplib::ThreadPool(config.http_threads, config.http_threads);
        };
        // 请求体上限 16384 字节，下面对 prompt 另限 12000 字节；字节数不等于模型的 token 数
        server.set_payload_max_length(16384);
        server.set_read_timeout(3, 0);       // 读请求最多 3 秒，防止慢速连接占着不放
        server.set_write_timeout(2, 0);      // 写响应最多 2 秒
        server.set_keep_alive_max_count(1);  // 每个连接只服务 1 次请求，不复用
        // 所有响应默认带上的头：禁止缓存；禁止浏览器猜测内容类型（安全加固）
        server.set_default_headers(
            {{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
        // handling 统计"当前占用推理名额的请求数"：1 个在执行 + queue_capacity 个在排队。
        // 接入上限定为 queue_capacity+1，再配合足够的 HTTP 线程，给健康检查、指标和拒绝响应留出余量；
        // 这个计数不会增加推理 worker 的数量。
        std::atomic<int> handling{0};


        // 健康检查
        // 功能：提供存活探针（/healthz）和就绪探针（/readyz），供负载均衡或容器编排系统探测。
        // 目的：上游据此决定"进程还活着吗"和"还能不能把新请求发给我"。
        // 存活：只要进程能响应就返回 ok
        server.Get("/healthz", [&](const auto&, auto& res) {
            reply(res, 200, {{"status", "ok"}});
        });
        // 就绪：收到停止信号后返回 503，通知上游"别再给我发新请求了"
        server.Get("/readyz", [&](const auto&, auto& res) {
            reply(res, interrupted ? 503 : 200, {{"ready", !interrupted}});
        });


        // 指标接口
        // 功能：统计 HTTP 层的状态码，并与调度器的统计合并后通过 /metrics 以 JSON 暴露。
        // 目的：HTTP 层和推理层各记各的，对照着看才能分清"请求被拒绝"和"生成失败"。
        // started：开始处理的请求数；logged：完成并记录的响应数；后面是各类状态码计数（429 同时也计入 4xx）
        std::atomic<uint64_t> http_started{0}, http_logged{0}, http_429{0}, http_4xx{0},
            http_5xx{0};
        // HTTP 状态和推理终态是分开计数的：流式响应一旦发出 200 头，之后生成失败，HTTP 层仍然记成 200。
        // 通俗讲：真实的推理结果要看调度器的指标，不能只看 HTTP 状态码。
        server.set_logger([&](const httplib::Request& req, const httplib::Response& res) {
            // 只统计 POST /v1/completions 这一条生成路由，健康检查和指标接口被频繁轮询也不会影响这里的计数
            if (req.method != "POST" || req.path != "/v1/completions")
                return;
            ++http_logged;
            if (res.status == 429)
                ++http_429;
            if (res.status >= 400 && res.status < 500)
                ++http_4xx;
            if (res.status >= 500)
                ++http_5xx;
        });
        // JSON 格式指标：调度器指标 + HTTP 层计数
        server.Get("/metrics", [&](const auto&, auto& res) {
            auto metrics = scheduler.metrics();
            metrics["http_requests_started"] = http_started.load();
            metrics["http_responses_logged"] = http_logged.load();
            metrics["http_status_429"] = http_429.load();
            metrics["http_status_4xx"] = http_4xx.load();
            metrics["http_status_5xx"] = http_5xx.load();
            reply(res, 200, metrics);
        });


        // Prometheus 接口
        // 功能：保留原有的 JSON 指标，另在独立地址提供 Prometheus 标准文本格式，方便监控系统抓取。
        // 目的：不使用 request_id 或 prompt 作为标签，避免每个请求都产生一条新的时间序列，把监控系统撑爆。
        server.Get("/metrics/prometheus", [&](const auto&, auto& res) {
            auto snapshot = scheduler.metrics();
            snapshot["http_requests_started"] = http_started.load();
            snapshot["http_responses_logged"] = http_logged.load();
            snapshot["http_status_429"] = http_429.load();
            snapshot["http_status_4xx"] = http_4xx.load();
            snapshot["http_status_5xx"] = http_5xx.load();
            res.set_content(serve::prometheus(snapshot),
                            "text/plain; version=0.0.4; charset=utf-8");
        });


        // 推理接口（POST /v1/completions）
        // 功能：校验请求、占用接入名额、提交排队，并根据 stream 选择一次性 JSON 或 SSE 流式输出。
        // 目的：保证接入名额"占一次、还一次"，不论中途走哪条分支（成功、拒绝、异常、客户端断开）都不漏还、不重复还。
        server.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
            ++http_started;
            // owns_slot 记录"当前这个处理函数是否占着接入名额"，出异常时据此决定要不要归还，避免漏还或重复还
            bool owns_slot = false;
            std::shared_ptr<serve::Job> job;
            try {


                // 请求校验
                // 功能：检查 JSON 字段类型、未知字段、提示词长度和输出上限。
                // 目的：JSON 里数字、字符串、布尔值是不同类型，比如把 max_tokens 写成字符串，
                //       服务应在接入层就给出参数错误，而不是等到引擎里才出问题。
                auto j = serve::Json::parse(req.body);
                // 必须是对象，prompt 必须是字符串，stream 必须是布尔，max_tokens 必须是能放进 int 的整数
                if (!j.is_object() || !j.contains("prompt") || !j["prompt"].is_string() ||
                    (j.contains("stream") && !j["stream"].is_boolean()) ||
                    (j.contains("max_tokens") && !serve::is_int(j["max_tokens"])))
                    throw std::invalid_argument("invalid JSON field types");
                // 白名单：只允许 prompt、max_tokens、stream 三个字段，多出来的直接拒绝
                for (auto it = j.begin(); it != j.end(); ++it)
                    if (it.key() != "prompt" && it.key() != "max_tokens" && it.key() != "stream")
                        throw std::invalid_argument("unknown request field: " + it.key());
                // 未提供的字段用默认值：max_tokens 取配置上限，stream 取 false
                serve::Request request{j.at("prompt").get<std::string>(),
                                       j.value("max_tokens", config.max_output_tokens),
                                       j.value("stream", false)};
                // 提示词不能为空、不超过 12000 字节、不含空字符；max_tokens 在 1 到配置上限之间
                if (request.prompt.empty() || request.prompt.size() > 12000 ||
                    request.prompt.find('\0') != std::string::npos || request.max_tokens < 1 ||
                    request.max_tokens > config.max_output_tokens)
                    throw std::invalid_argument("prompt or max_tokens outside allowed bounds");
                // 正在关闭时不再接新请求
                if (interrupted) {
                    reply(res, 503, {{"error", "shutting_down"}});
                    return;
                }


                // 有界接入
                // 功能：先占 HTTP 接入名额，再尝试提交到等待队列。
                // 目的：两层拒绝，客户端看到的原因不同：
                //   第一层：接入名额已满，请求根本没进 Scheduler，返回 429 admission_full；
                //   第二层：占到名额后，Scheduler 因队列满、预算不足等原因拒绝提交，返回 429 queue_full。
                //   因此 Scheduler 统计的 rejected 不包含所有的 HTTP 429。
                // 通俗讲：餐厅门口先限制总人数（第一层），进门后座位也满了再拒绝（第二层）。
                // fetch_add 在同一次原子操作里"取旧值并加 1"，并发到达的线程不会拿到同一个旧计数。
                // 旧值已达上限说明满了：立刻撤销这次加 1，返回 429，并让客户端 1 秒后重试
                if (handling.fetch_add(1) >= config.queue_capacity + 1) {
                    handling.fetch_sub(1);
                    res.set_header("Retry-After", "1");
                    reply(res, 429, {{"error", "admission_full"}});
                    return;
                }
                owns_slot = true;
                job = scheduler.submit(request);
                // 队列满、预算不足或调度器正在停机，submit 都可能返回空指针。
                // 这条分支统一归还接入名额，并回 429 queue_full
                if (!job) {
                    handling.fetch_sub(1);
                    owns_slot = false;
                    res.set_header("Retry-After", "1");
                    reply(res, 429, {{"error", "queue_full"}});
                    return;
                }
                // 把请求编号放进响应头，便于客户端和日志对账
                res.set_header("X-Request-ID", job->id);


                // 等待执行
                // 功能：先等任务开始，再读首个事件。
                // 目的：此时响应头还没发出，早期的失败仍能返回真实的 400/504 等状态码。
                // await_start 返回 false 表示在开始执行前就被取消了（比如排队超时或服务关闭）
                if (!job->await_start()) {
                    std::string reason;
                    {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        reason = job->cancel_reason;
                    }
                    handling.fetch_sub(1);
                    owns_slot = false;
                    reply(res,
                          reason == "shutdown" ? 503 : 504,
                          {{"request_id", job->id}, {"error", reason}});
                    return;
                }


                // 等第一条事件
                // 功能：取出第一个事件。如果没有，或者是 error，说明引擎一开始就失败了，直接用对应的状态码返回。
                // 目的：一旦把 SSE 响应头发出去，就很难再换 HTTP 状态码，之后的同类错误只能放在数据流里作为事件告知。
                //       先看首个事件，就能把"一开始就失败"的情况当作普通的请求错误返回。
                auto initial = job->next();
                if (!initial || (*initial)["type"] == "error") {
                    handling.fetch_sub(1);
                    owns_slot = false;
                    auto err =
                        initial.value_or(serve::Json{{"type", "error"}, {"code", "engine_error"}});
                    reply(res, error_status(err), err);
                    return;
                }


                // JSON 响应（非流式）
                // 功能：消费事件缓冲，把所有文本片段拼起来一次性返回。
                // 说明：非流式请求读的也是同一个有界事件缓冲，只是攒齐文本之后才返回 JSON。
                if (!request.stream) {
                    std::string output;
                    auto event = initial;
                    // 不断读取 delta 事件并拼接，直到遇到终态事件（done 或 error）
                    while (event && (*event)["type"] == "delta") {
                        output += event->at("text").get<std::string>();
                        event = job->next();
                    }
                    // 生成已结束，归还接入名额
                    handling.fetch_sub(1);
                    owns_slot = false;
                    if (!event || (*event)["type"] == "error") {
                        auto err = event.value_or(serve::Json{{"code", "engine_error"}});
                        reply(res, error_status(err), err);
                        return;
                    }
                    // 把拼好的完整文本塞进终态事件里返回
                    (*event)["text"] = output;
                    reply(res, 200, *event);
                    return;
                }


                // SSE 响应（流式）
                // 功能：先发出前面已经取出的首个事件，之后逐段写给客户端，并处理中途断连。
                // 说明：SSE 是 HTTP 响应里的事件流，不是 WebSocket；每一帧写成 "data: JSON" 加一个空行。
                // 通俗讲：像水管一样，生成一点就推送一点，客户端能立刻看到文字一个个冒出来。
                // 告诉 Nginx 之类的反向代理不要缓冲，否则流式效果会被卡住
                res.set_header("X-Accel-Buffering", "no");


                // 保留首个事件
                // 前面为了检查错误已经取出过一次事件，流回调要先把它发出去，
                // 否则第一个片段就丢了。回调按值捕获 job 和 pending 两个 shared_ptr，
                // 保证后续每次发送时 Job 都还没被销毁。
                auto pending = std::make_shared<std::optional<serve::Json>>(std::move(initial));
                res.set_chunked_content_provider(
                    "text/event-stream; charset=utf-8",
                    // 内容提供回调：库每次需要写下一块数据时调用它
                    [job, pending](size_t, httplib::DataSink& sink) {
                        // 连接已断开：取消任务，释放推理资源，返回 false 终止发送
                        if (!sink.is_writable()) {
                            job->cancel("client_disconnect");
                            return false;
                        }
                        // 优先发送之前已取出的首事件，用完清空；之后才从队列取新事件
                        auto event = *pending;
                        if (event)
                            pending->reset();
                        else
                            event = job->next();
                        // 没有更多事件，正常结束流
                        if (!event) {
                            sink.done();
                            return true;
                        }
                        const auto type = event->value("type", "");
                        // SSE 帧格式：错误事件多一行 "event: error"，然后是 "data: JSON" 和空行
                        const std::string frame = (type == "error" ? "event: error\n" : "") +
                                                  std::string("data: ") + event->dump() + "\n\n";
                        // 写给已断开的客户端可能失败：取消对应任务，让 worker 有机会停下来，
                        // 不再为已经离开的人生成后续文字
                        if (!sink.write(frame.data(), frame.size())) {
                            job->cancel("client_disconnect");
                            return false;
                        }
                        // 只有成功结束才追加流协议的 [DONE]。
                        // 生成失败要让客户端看到错误事件，不能把失败当成正常结束
                        if (type == "done") {
                            const std::string end = "data: [DONE]\n\n";
                            if (!sink.write(end.data(), end.size())) {
                                job->cancel("client_disconnect");
                                return false;
                            }
                            sink.done();
                        } else if (type == "error")
                            sink.done();
                        return true;
                    },
                    // 资源释放回调：发送结束或中断时调用，success 为 false 表示异常结束
                    [job, &handling](bool success) {
                        if (!success)
                            job->cancel("client_disconnect");
                        // 在这里归还接入名额，保证流式期间名额一直被占用
                        handling.fetch_sub(1);
                    });
                // 从现在起，接入名额由上面的资源释放回调负责归还（发送完成或断连时），
                // 所以这里把 owns_slot 清零：既不能在处理函数返回时提前归还，也不能重复归还
                owns_slot = false;
            } catch (const serve::Json::exception&) {
                // JSON 解析失败或类型不对：取消已提交的任务、归还名额，返回 400
                if (job)
                    job->cancel("invalid_request");
                if (owns_slot)
                    handling.fetch_sub(1);
                reply(res, 400, {{"error", "invalid_json_or_type"}});
            } catch (const std::invalid_argument& e) {
                // 字段校验失败：同上，并把具体原因告诉客户端
                if (job)
                    job->cancel("invalid_request");
                if (owns_slot)
                    handling.fetch_sub(1);
                reply(res, 400, {{"error", e.what()}});
            } catch (const std::exception& e) {
                // 其他未预料的异常：记日志，对外只说 internal_error，不泄露内部细节
                if (job)
                    job->cancel("handler_error");
                if (owns_slot)
                    handling.fetch_sub(1);
                serve::log({{"event", "handler_error"}, {"message", e.what()}});
                reply(res, 500, {{"error", "internal_error"}});
            }
        });


        // 监听与退出
        // 功能：统一处理错误响应，绑定端口，并监视停止信号，实现有序关闭。
        // 目的：退出时先取消任务并回收推理 worker，再停止 HTTP 服务，不是强杀线程或进程。
        // 兜底：路由处理中漏掉的异常，统一返回 500
        server.set_exception_handler([](const auto&, auto& res, std::exception_ptr) {
            reply(res, 500, {{"error", "internal_error"}});
        });
        // 库自己产生的错误（如 404、413）若没有响应体，补上统一的 JSON 格式
        server.set_error_handler([](const auto&, auto& res) {
            if (res.body.empty())
                reply(res, res.status, {{"error", "http_error"}, {"status", res.status}});
        });
        // 先绑定端口（默认只绑本机回环地址），失败就直接报错退出
        if (!server.bind_to_port(config.host, config.port))
            throw std::runtime_error("cannot bind loopback port");
        // 注册信号：Ctrl+C 和 kill 触发优雅退出；忽略 SIGPIPE，避免客户端断开时写数据导致整个进程被杀
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::signal(SIGPIPE, SIG_IGN);
        std::atomic<bool> finished{false};
        // 监视线程每 20 毫秒检查一次停止标志，发现后依次停调度器、停 HTTP 服务。
        // 信号回调只记录退出意图，由监视线程看到标志后才组织关闭，最后由 main 回收线程
        std::thread monitor([&] {
            while (!finished.load()) {
                if (interrupted) {
                    scheduler.stop();
                    server.stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        serve::log({{"event", "listening"},
                    {"host", config.host},
                    {"port", config.port},
                    {"engine", config.engine}});
        // 阻塞在这里处理请求，直到 server.stop() 被调用或出错
        bool ok = server.listen_after_bind();
        // 通知监视线程结束并等它退出，再确保调度器已停止（这里会再调用一次 stop），最后记录最终指标
        finished = true;
        monitor.join();
        scheduler.stop();
        serve::log({{"event", "stopped"}, {"metrics", scheduler.metrics()}});
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        // 启动阶段（加载配置、模型、绑定端口）失败：记录日志并返回 1
        serve::log({{"event", "startup_error"}, {"message", e.what()}});
        return 1;
    }
}
