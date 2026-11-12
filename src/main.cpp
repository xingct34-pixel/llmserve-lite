#include "serve.hpp"
#include <httplib/httplib.h>
#include <csignal>
#include <iostream>


// 模块：HTTP辅助：信号标志、JSON响应和错误码映射


namespace {
static_assert(std::atomic<bool>::is_always_lock_free, "signal flag must be lock-free");
std::atomic<bool> interrupted{false};

// 信号回调仅置无锁原子标志；关闭socket、取消任务和join由正常线程处理。
void on_signal(int) {
    interrupted.store(true, std::memory_order_relaxed);
}

void reply(httplib::Response& res, int status, const serve::Json& data) {
    res.status = status;
    res.set_content(data.dump(), "application/json; charset=utf-8");
}

int error_status(const serve::Json& event) {
    auto code = event.value("code", "");
    if (code == "invalid_request")
        return 400;
    if (code == "queue_timeout" || code == "generation_timeout")
        return 504;
    if (code == "shutdown")
        return 503;
    return 500;
}
}


// 模块：服务入口：加载配置/引擎，注册路由并监听本地端口


int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: llmserve configs/server.json\n";
            return 2;
        }


        // 模块：1. 服务初始化：加载引擎，设置线程池和HTTP资源限制


        const auto config = serve::Config::load(argv[1]);
        // 先加载唯一引擎，再启动单worker；加载失败时不会监听端口或宣称服务就绪。
        serve::Scheduler scheduler(config, serve::make_engine(config));
        httplib::Server server;
        server.new_task_queue = [&] {
            return new httplib::ThreadPool(config.http_threads, config.http_threads);
        };
        // HTTP请求体上限16384字节，下面prompt另限12000字节；字节数不等于模型token数。
        server.set_payload_max_length(16384);
        server.set_read_timeout(3, 0);
        server.set_write_timeout(2, 0);
        server.set_keep_alive_max_count(1);
        server.set_default_headers(
            {{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
        // 接入上限=等待容量+1，配合线程数下限给健康/指标/拒绝响应留余量；不增加推理worker。
        std::atomic<int> handling{0};


        // 模块：2. 健康检查：提供存活与就绪探针


        server.Get("/healthz", [&](const auto&, auto& res) {
            reply(res, 200, {{"status", "ok"}});
        });
        server.Get("/readyz", [&](const auto&, auto& res) {
            reply(res, interrupted ? 503 : 200, {{"ready", !interrupted}});
        });


        // 模块：3. 指标接口：分别记录HTTP状态与调度器结果


        std::atomic<uint64_t> http_started{0}, http_logged{0}, http_429{0}, http_4xx{0},
            http_5xx{0};
        // HTTP状态和推理终态分别计数：SSE头已发出后生成失败，HTTP仍可能记录200。
        server.set_logger([&](const httplib::Request& req, const httplib::Response& res) {
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
        server.Get("/metrics", [&](const auto&, auto& res) {
            auto metrics = scheduler.metrics();
            metrics["http_requests_started"] = http_started.load();
            metrics["http_responses_logged"] = http_logged.load();
            metrics["http_status_429"] = http_429.load();
            metrics["http_status_4xx"] = http_4xx.load();
            metrics["http_status_5xx"] = http_5xx.load();
            reply(res, 200, metrics);
        });


        // 模块：Prometheus接口：保留原JSON指标，在独立地址提供标准文本格式
        // 不使用request_id或prompt作为标签，避免为每条请求创建一批新时间序列。


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


        // 模块：4. 推理接口：校验、接入、排队并选择JSON或SSE输出


        server.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
            ++http_started;
            bool owns_slot = false;
            std::shared_ptr<serve::Job> job;
            try {


                // 模块：4.1 请求校验：检查JSON字段和提示词/输出范围


                auto j = serve::Json::parse(req.body);
                if (!j.is_object() || !j.contains("prompt") || !j["prompt"].is_string() ||
                    (j.contains("stream") && !j["stream"].is_boolean()) ||
                    (j.contains("max_tokens") && !serve::is_int(j["max_tokens"])))
                    throw std::invalid_argument("invalid JSON field types");
                for (auto it = j.begin(); it != j.end(); ++it)
                    if (it.key() != "prompt" && it.key() != "max_tokens" && it.key() != "stream")
                        throw std::invalid_argument("unknown request field: " + it.key());
                serve::Request request{j.at("prompt").get<std::string>(),
                                       j.value("max_tokens", config.max_output_tokens),
                                       j.value("stream", false)};
                if (request.prompt.empty() || request.prompt.size() > 12000 ||
                    request.prompt.find('\0') != std::string::npos || request.max_tokens < 1 ||
                    request.max_tokens > config.max_output_tokens)
                    throw std::invalid_argument("prompt or max_tokens outside allowed bounds");
                if (interrupted) {
                    reply(res, 503, {{"error", "shutting_down"}});
                    return;
                }


                // 模块：4.2 有界接入：先占HTTP名额，再尝试提交等待队列
                // 两层拒绝：此处接入已满返回admission_full；Scheduler等待队列满返回queue_full。
                // 接入层拒绝没有进入调度器，因此Scheduler.rejected不包含所有HTTP 429。


                if (handling.fetch_add(1) >= config.queue_capacity + 1) {
                    handling.fetch_sub(1);
                    res.set_header("Retry-After", "1");
                    reply(res, 429, {{"error", "admission_full"}});
                    return;
                }
                owns_slot = true;
                job = scheduler.submit(request);
                if (!job) {
                    handling.fetch_sub(1);
                    owns_slot = false;
                    res.set_header("Retry-After", "1");
                    reply(res, 429, {{"error", "queue_full"}});
                    return;
                }
                res.set_header("X-Request-ID", job->id);


                // 模块：4.3 等待执行：早期失败仍返回真实HTTP错误码
                // 先等开始、再读首事件，响应头尚未发出，早期错误仍能用400/504等真实HTTP状态。


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
                auto initial = job->next();
                if (!initial || (*initial)["type"] == "error") {
                    handling.fetch_sub(1);
                    owns_slot = false;
                    auto err =
                        initial.value_or(serve::Json{{"type", "error"}, {"code", "engine_error"}});
                    reply(res, error_status(err), err);
                    return;
                }


                // 模块：4.4 JSON响应：消费事件缓冲并汇总为一次返回
                // 非流式也消费同一个有界事件缓冲，只是攒齐文本后一次返回JSON。


                if (!request.stream) {
                    std::string output;
                    auto event = initial;
                    while (event && (*event)["type"] == "delta") {
                        output += event->at("text").get<std::string>();
                        event = job->next();
                    }
                    handling.fetch_sub(1);
                    owns_slot = false;
                    if (!event || (*event)["type"] == "error") {
                        auto err = event.value_or(serve::Json{{"code", "engine_error"}});
                        reply(res, error_status(err), err);
                        return;
                    }
                    (*event)["text"] = output;
                    reply(res, 200, *event);
                    return;
                }


                // 模块：4.5 SSE响应：保留首事件，逐段写入并处理断连
                // SSE是HTTP响应中的事件流，不是WebSocket。每帧用data: JSON加空行分隔。
                // pending保留已取出的首事件；shared_ptr按值捕获，保证后续发送回调中Job仍有效。


                res.set_header("X-Accel-Buffering", "no");
                auto pending = std::make_shared<std::optional<serve::Json>>(std::move(initial));
                res.set_chunked_content_provider(
                    "text/event-stream; charset=utf-8",
                    [job, pending](size_t, httplib::DataSink& sink) {
                        if (!sink.is_writable()) {
                            job->cancel("client_disconnect");
                            return false;
                        }
                        auto event = *pending;
                        if (event)
                            pending->reset();
                        else
                            event = job->next();
                        if (!event) {
                            sink.done();
                            return true;
                        }
                        const auto type = event->value("type", "");
                        const std::string frame = (type == "error" ? "event: error\n" : "") +
                                                  std::string("data: ") + event->dump() + "\n\n";
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
                    [job, &handling](bool success) {
                        if (!success)
                            job->cancel("client_disconnect");
                        handling.fetch_sub(1);
                    });
                // 发送完成/断连时由releaser释放接入名额，不能在handler返回时提前归还或重复归还。
                owns_slot = false; // resource releaser now owns the admission slot
            } catch (const serve::Json::exception&) {
                if (job)
                    job->cancel("invalid_request");
                if (owns_slot)
                    handling.fetch_sub(1);
                reply(res, 400, {{"error", "invalid_json_or_type"}});
            } catch (const std::invalid_argument& e) {
                if (job)
                    job->cancel("invalid_request");
                if (owns_slot)
                    handling.fetch_sub(1);
                reply(res, 400, {{"error", e.what()}});
            } catch (const std::exception& e) {
                if (job)
                    job->cancel("handler_error");
                if (owns_slot)
                    handling.fetch_sub(1);
                serve::log({{"event", "handler_error"}, {"message", e.what()}});
                reply(res, 500, {{"error", "internal_error"}});
            }
        });


        // 模块：5. 监听与退出：统一处理错误、绑定端口和监视停止信号


        server.set_exception_handler([](const auto&, auto& res, std::exception_ptr) {
            reply(res, 500, {{"error", "internal_error"}});
        });
        server.set_error_handler([](const auto&, auto& res) {
            if (res.body.empty())
                reply(res, res.status, {{"error", "http_error"}, {"status", res.status}});
        });
        if (!server.bind_to_port(config.host, config.port))
            throw std::runtime_error("cannot bind loopback port");
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::signal(SIGPIPE, SIG_IGN);
        std::atomic<bool> finished{false};
        // 退出顺序：取消Job并join推理worker，再停止HTTP服务；不是强杀其他线程或进程。
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
        bool ok = server.listen_after_bind();
        finished = true;
        monitor.join();
        scheduler.stop();
        serve::log({{"event", "stopped"}, {"metrics", scheduler.metrics()}});
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        serve::log({{"event", "startup_error"}, {"message", e.what()}});
        return 1;
    }
}
