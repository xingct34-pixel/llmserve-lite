#include "serve.hpp"
#include <httplib/httplib.h>
#include <csignal>
#include <iostream>

namespace {
static_assert(std::atomic<bool>::is_always_lock_free,"signal flag must be lock-free");
std::atomic<bool> interrupted{false};
void on_signal(int){interrupted.store(true,std::memory_order_relaxed);}
void reply(httplib::Response& res,int status,const serve::Json& data) {
    res.status=status;res.set_content(data.dump(),"application/json; charset=utf-8");
}
int error_status(const serve::Json& event) {
    auto code=event.value("code","");
    if(code=="invalid_request") return 400;
    if(code=="queue_timeout"||code=="generation_timeout") return 504;
    if(code=="shutdown") return 503;
    return 500;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2){std::cerr<<"usage: llmserve configs/server.json\n";return 2;}
        const auto config=serve::Config::load(argv[1]);
        serve::Scheduler scheduler(config,serve::make_engine(config));
        httplib::Server server;
        server.new_task_queue=[&]{return new httplib::ThreadPool(config.http_threads,config.http_threads);};
        server.set_payload_max_length(16384);
        server.set_read_timeout(3,0);server.set_write_timeout(2,0);
        server.set_keep_alive_max_count(1);
        server.set_default_headers({{"Cache-Control","no-store"},{"X-Content-Type-Options","nosniff"}});
        // Admission cap reserves at least two HTTP workers for health/metrics and rejections.
        std::atomic<int> handling{0};
        server.Get("/healthz",[&](const auto&,auto& res){reply(res,200,{{"status","ok"}});});
        server.Get("/readyz",[&](const auto&,auto& res){reply(res,interrupted?503:200,{{"ready",!interrupted}});});

        std::atomic<uint64_t> http_started{0},http_logged{0},http_429{0},http_4xx{0},http_5xx{0};
        // HTTP status accounting is separate from inference completion accounting.
        server.set_logger([&](const httplib::Request& req,const httplib::Response& res){
            if(req.method!="POST"||req.path!="/v1/completions")return;
            ++http_logged;if(res.status==429)++http_429;
            if(res.status>=400&&res.status<500)++http_4xx;
            if(res.status>=500)++http_5xx;
        });
        server.Get("/metrics",[&](const auto&,auto& res){
            auto metrics=scheduler.metrics();
            metrics["http_requests_started"]=http_started.load();
            metrics["http_responses_logged"]=http_logged.load();
            metrics["http_status_429"]=http_429.load();
            metrics["http_status_4xx"]=http_4xx.load();
            metrics["http_status_5xx"]=http_5xx.load();
            reply(res,200,metrics);
        });
        server.Post("/v1/completions",[&](const httplib::Request& req,httplib::Response& res){
            ++http_started;
            bool owns_slot=false;
            std::shared_ptr<serve::Job> job;
            try {
                auto j=serve::Json::parse(req.body);
                if(!j.is_object()||!j.contains("prompt")||!j["prompt"].is_string()||
                   (j.contains("stream")&&!j["stream"].is_boolean())||
                   (j.contains("max_tokens")&&!serve::is_int(j["max_tokens"]))) throw std::invalid_argument("invalid JSON field types");
                for(auto it=j.begin();it!=j.end();++it) if(it.key()!="prompt"&&it.key()!="max_tokens"&&it.key()!="stream") throw std::invalid_argument("unknown request field: "+it.key());
                serve::Request request{j.at("prompt").get<std::string>(),j.value("max_tokens",config.max_output_tokens),j.value("stream",false)};
                if(request.prompt.empty()||request.prompt.size()>12000||request.prompt.find('\0')!=std::string::npos||request.max_tokens<1||request.max_tokens>config.max_output_tokens) throw std::invalid_argument("prompt or max_tokens outside allowed bounds");
                if(interrupted){reply(res,503,{{"error","shutting_down"}});return;}
                if(handling.fetch_add(1)>=config.queue_capacity+1) {
                    handling.fetch_sub(1);res.set_header("Retry-After","1");reply(res,429,{{"error","admission_full"}});return;
                }
                owns_slot=true;
                job=scheduler.submit(request);
                if(!job) {handling.fetch_sub(1);owns_slot=false;res.set_header("Retry-After","1");reply(res,429,{{"error","queue_full"}});return;}
                res.set_header("X-Request-ID",job->id);
                if(!job->await_start()) {
                    std::string reason;
                    {std::lock_guard<std::mutex> lock(job->mutex);reason=job->cancel_reason;}
                    handling.fetch_sub(1);owns_slot=false;
                    reply(res,reason=="shutdown"?503:504,{{"request_id",job->id},{"error",reason}});return;
                }
                auto initial=job->next();
                if(!initial || (*initial)["type"]=="error") {
                    handling.fetch_sub(1);owns_slot=false;
                    auto err=initial.value_or(serve::Json{{"type","error"},{"code","engine_error"}});
                    reply(res,error_status(err),err);return;
                }
                if(!request.stream) {
                    std::string output;
                    auto event=initial;
                    while(event && (*event)["type"]=="delta") {output+=event->at("text").get<std::string>();event=job->next();}
                    handling.fetch_sub(1);owns_slot=false;
                    if(!event || (*event)["type"]=="error") {
                        auto err=event.value_or(serve::Json{{"code","engine_error"}});reply(res,error_status(err),err);return;
                    }
                    (*event)["text"]=output;reply(res,200,*event);return;
                }
                res.set_header("X-Accel-Buffering","no");
                auto pending=std::make_shared<std::optional<serve::Json>>(std::move(initial));
                res.set_chunked_content_provider("text/event-stream; charset=utf-8",
                    [job,pending](size_t,httplib::DataSink& sink){
                        if(!sink.is_writable()){job->cancel("client_disconnect");return false;}
                        auto event=*pending;
                        if(event) pending->reset();else event=job->next();
                        if(!event){sink.done();return true;}
                        const auto type=event->value("type","");
                        const std::string frame=(type=="error"?"event: error\n":"")+std::string("data: ")+event->dump()+"\n\n";
                        if(!sink.write(frame.data(),frame.size())){job->cancel("client_disconnect");return false;}
                        if(type=="done") {
                            const std::string end="data: [DONE]\n\n";
                            if(!sink.write(end.data(),end.size())){job->cancel("client_disconnect");return false;}
                            sink.done();
                        }else if(type=="error") sink.done();
                        return true;
                    },[job,&handling](bool success){if(!success)job->cancel("client_disconnect");handling.fetch_sub(1);});
                owns_slot=false; // resource releaser now owns the admission slot
            }catch(const serve::Json::exception&){
                if(job)job->cancel("invalid_request");
                if(owns_slot)handling.fetch_sub(1);
                reply(res,400,{{"error","invalid_json_or_type"}});
            }catch(const std::invalid_argument& e){
                if(job)job->cancel("invalid_request");
                if(owns_slot)handling.fetch_sub(1);
                reply(res,400,{{"error",e.what()}});
            }catch(const std::exception& e){
                if(job)job->cancel("handler_error");
                if(owns_slot)handling.fetch_sub(1);
                serve::log({{"event","handler_error"},{"message",e.what()}});
                reply(res,500,{{"error","internal_error"}});
            }
        });
        server.set_exception_handler([](const auto&,auto& res,std::exception_ptr){reply(res,500,{{"error","internal_error"}});});
        server.set_error_handler([](const auto&,auto& res){if(res.body.empty())reply(res,res.status,{{"error","http_error"},{"status",res.status}});});
        if(!server.bind_to_port(config.host,config.port)) throw std::runtime_error("cannot bind loopback port");
        std::signal(SIGINT,on_signal);std::signal(SIGTERM,on_signal);std::signal(SIGPIPE,SIG_IGN);
        std::atomic<bool> finished{false};
        std::thread monitor([&]{
            while(!finished.load()) {
                if(interrupted) {scheduler.stop();server.stop();return;}
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        serve::log({{"event","listening"},{"host",config.host},{"port",config.port},{"engine",config.engine}});
        bool ok=server.listen_after_bind();
        finished=true;
        monitor.join();scheduler.stop();
        serve::log({{"event","stopped"},{"metrics",scheduler.metrics()}});
        return ok?0:1;
    }catch(const std::exception& e){serve::log({{"event","startup_error"},{"message",e.what()}});return 1;}
}
