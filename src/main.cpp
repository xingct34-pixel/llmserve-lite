#include "serve.hpp"
#include <httplib/httplib.h>
#include <csignal>
#include <iostream>


// 模块：HTTP 辅助（信号标志、JSON 响应、错误码映射）
// 作用：提供处理 Ctrl+C 的停止标志、统一的 JSON 回复函数，以及把内部错误码翻译成 HTTP 状态码。


namespace {
// 信号处理函数里只能安全地做很少的事，所以标志必须是"无锁"的原子变量，编译期检查这一点
static_assert(std::atomic<bool>::is_always_lock_free, "signal flag must be lock-free");
// 收到 SIGINT/SIGTERM 后置为 true，表示"请准备退出"
std::atomic<bool> interrupted{false};

// 信号回调仅置无锁原子标志；关闭socket、取消任务和join由正常线程处理。
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
}


// 模块：服务入口（main）
// 作用：加载配置和引擎，注册所有路由，监听本地端口，并在收到停止信号时有序退出。


int main(int argc, char** argv) {
    try {
        // 必须恰好传入一个配置文件路径，否则打印用法并返回 2
        if (argc != 2) {
            std::cerr << "usage: llmserve configs/server.json\n";
            return 2;
        }


        // 模块：1. 服务初始化
        // 作用：加载引擎，设置 HTTP 线程池和资源限制。


        const auto config = serve::Config::load(argv[1]);
        // 先加载唯一引擎，再启动单worker；加载失败时不会监听端口或宣称服务就绪。
        // 通俗讲：模型没加载好之前，服务不会对外说"我准备好了"。
        serve::Scheduler scheduler(config, serve::make_engine(config));
        httplib::Server server;
        // 自定义 HTTP 线程池，线程数由配置决定（两个参数分别是基础线程数和最大线程数，这里相同即固定大小）
        server.new_task_queue = [&] {
            return new httplib::ThreadPool(config.http_threads, config.http_threads);
        };
        // HTTP请求体上限16384字节，下面prompt另限12000字节；字节数不等于模型token数。
        server.set_payload_max_length(16384);
        server.set_read_timeout(3, 0);    // 读请求最多 3 秒，防止慢速连接占着不放
        server.set_write_timeout(2, 0);   // 写响应最多 2 秒
        server.set_keep_alive_max_count(1);  // 每个连接只服务 1 次请求，不复用
        // 所有响应默认带上的头：禁止缓存；禁止浏览器猜测内容类型（安全加固）
        server.set_default_headers(
            {{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
        // 接入上限=等待容量+1，配合线程数下限给健康/指标/拒绝响应留余量；不增加推理worker。
        // handling 统计"当前占用推理名额的请求数"：1 个在执行 + queue_capacity 个在排队
        std::atomic<int> handling{0};


        // 模块：2. 健康检查
        // 作用：提供存活探针（/healthz）和就绪探针（/readyz），供负载均衡或容器编排系统探测。


        // 存活：只要进程能响应就返回 ok
        server.Get("/healthz", [&](const auto&, auto& res) {
            reply(res, 200, {{"status", "ok"}});
        });
        // 就绪：收到停止信号后返回 503，通知上游"别再给我发新请求了"
        server.Get("/readyz", [&](const auto&, auto& res) {
            reply(res, interrupted ? 503 : 200, {{"ready", !interrupted}});
        });


        // 模块：3. 指标接口
        // 作用：统计 HTTP 层的状态码，并与调度器的统计合并后通过 /metrics 暴露。


        // started：开始处理的请求数；logged：完成并记录的响应数；后面是各类状态码计数
        std::atomic<uint64_t> http_started{0}, http_logged{0}, http_429{0}, http_4xx{0},
            http_5xx{0};
        // HTTP状态和推理终态分别计数：SSE头已发出后生成失败，HTTP仍可能记录200。
        // 通俗讲：流式响应一旦发出 200 头，之后即使生成出错，HTTP 层也只会记成 200，
        // 所以真实的推理结果要看调度器的指标。
        server.set_logger([&](const httplib::Request& req, const httplib::Response& res) {
            // 只统计推理接口，健康检查和指标接口不计入
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


        // 模块：Prometheus 接口
        // 作用：保留原 JSON 指标，另在独立地址提供 Prometheus 标准文本格式，方便监控系统抓取。
        // 不使用 request_id 或 prompt 作为标签，避免每个请求都产生一条新的时间序列，把监控系统撑爆。


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


        // 模块：4. 推理接口（POST /v1/completions）
        // 作用：校验请求、占用接入名额、提交排队，并根据 stream 选择一次性 JSON 或 SSE 流式输出。


        server.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
            ++http_started;
            // owns_slot 记录"当前这个处理函数是否占着接入名额"，出异常时据此决定要不要归还，避免漏还或重复还
            bool owns_slot = false;
            std::shared_ptr<serve::Job> job;
            try {


                // 模块：4.1 请求校验
                // 作用：检查 JSON 字段类型、未知字段、提示词长度和输出上限。


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


                // 模块：4.2 有界接入
                // 作用：先占 HTTP 名额，再尝试提交等待队列。
                // 两层拒绝：此处接入已满返回 admission_full；Scheduler 等待队列满返回 queue_full。
                // 接入层拒绝没有进入调度器，因此 Scheduler.rejected 不包含所有 HTTP 429。
                // 通俗讲：餐厅门口先限制总人数（第一层），进门后座位也满了再拒绝（第二层）。


                // fetch_add 先加 1 并返回旧值；旧值已达上限说明满了，立刻撤销并返回 429，让客户端 1 秒后重试
                if (handling.fetch_add(1) >= config.queue_capacity + 1) {
                    handling.fetch_sub(1);
                    res.set_header("Retry-After", "1");
                    reply(res, 429, {{"error", "admission_full"}});
                    return;
                }
                owns_slot = true;
                job = scheduler.submit(request);
                // submit 返回空指针表示等待队列已满，归还名额并返回 429
                if (!job) {
                    handling.fetch_sub(1);
                    owns_slot = false;
                    res.set_header("Retry-After", "1");
                    reply(res, 429, {{"error", "queue_full"}});
                    return;
                }
                // 把请求编号放进响应头，便于客户端和日志对账
                res.set_header("X-Request-ID", job->id);


                // 模块：4.3 等待执行
                // 作用：先等任务开始、再读首个事件；此时响应头还没发出，早期失败仍能返回真实的 400/504 等状态码。


                // await_start 返回 false 表示在开始执行前就被取消了（排队超时或服务关闭）
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
                // 取第一个事件：若没有或是 error，说明引擎一开始就失败了，直接用对应状态码返回
                auto initial = job->next();
                if (!initial || (*initial)["type"] == "error") {
                    handling.fetch_sub(1);
                    owns_slot = false;
                    auto err =
                        initial.value_or(serve::Json{{"type", "error"}, {"code", "engine_error"}});
                    reply(res, error_status(err), err);
                    return;
                }


                // 模块：4.4 JSON 响应（非流式）
                // 作用：消费事件缓冲，把所有文本片段拼起来一次性返回。
                // 非流式也消费同一个有界事件缓冲，只是攒齐文本后一次返回 JSON。


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


                // 模块：4.5 SSE 响应（流式）
                // 作用：保留已取出的首事件，之后逐段写入客户端，并处理中途断连。
                // SSE 是 HTTP 响应中的事件流，不是 WebSocket。每帧用 data: JSON 加空行分隔。
                // pending 保留已取出的首事件；shared_ptr 按值捕获，保证后续发送回调中 Job 仍有效。
                // 通俗讲：像水管一样，生成一点就推送一点，客户端能立刻看到文字一个个冒出来。


                // 告诉 Nginx 之类的反向代理不要缓冲，否则流式效果会被卡住
                res.set_header("X-Accel-Buffering", "no");
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
                        // 写失败说明客户端断了，取消任务
                        if (!sink.write(frame.data(), frame.size())) {
                            job->cancel("client_disconnect");
                            return false;
                        }
                        // 成功终态后只发一次[DONE]；中途错误发event:error后结束，不冒充成功。
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
                // 发送完成/断连时由releaser释放接入名额，不能在handler返回时提前归还或重复归还。
                owns_slot = false; // resource releaser now owns the admission slot
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


        // 模块：5. 监听与退出
        // 作用：统一处理错误响应，绑定端口，并监视停止信号，实现有序关闭。


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
        // 退出顺序：取消Job并join推理worker，再停止HTTP服务；不是强杀其他线程或进程。
        // 监视线程每 20 毫秒检查一次停止标志，发现后依次停调度器、停 HTTP 服务
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
        // 通知监视线程结束并等它退出，再确保调度器已停止，最后记录最终指标
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
