#include "serve.hpp"
#include <iostream>
#include <stdexcept>
using namespace serve;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
Json drain(const std::shared_ptr<Job>& j){Json last;while(auto e=j->next())last=*e;return last;}
class Gate final:public Engine {
public:
    std::atomic<bool> release{false};
    std::string name() const override{return "gate";}
    Generation generate(const Request&,const std::function<bool(const std::string&)>& emit,const std::function<bool()>& stop) override {
        while(!release && !stop())std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if(!stop())emit("好");return {1,1,"stop"};
    }
};
int main(){
    try {
        Config c;c.mock_delay_ms=0;
        {
            std::string bytes="\xe4";
            check(complete_utf8(bytes).empty(),"partial UTF-8 lead byte");
            bytes+="\xbd";check(complete_utf8(bytes).empty(),"partial UTF-8 continuation");
            bytes+="\xa0好";check(complete_utf8(bytes)=="你好" && bytes.empty(),"UTF-8 across token boundaries");
            bytes="\xf0\x9f\x98";check(complete_utf8(bytes).empty(),"partial four-byte UTF-8");
            bytes+="\x80";check(complete_utf8(bytes)=="😀","four-byte UTF-8 completed");
            bytes="\xff" "A";check(complete_utf8(bytes)=="�A","invalid UTF-8 replaced");
            bytes="\xe4";check(complete_utf8(bytes,true)=="�","unfinished final bytes replaced");
        }
        {
            Scheduler s(c,make_engine(c));
            auto j=s.submit({"hello",8,false});check(j->await_start(),"mock start");
            std::string out;Json terminal;
            while(auto e=j->next()){if((*e)["type"]=="delta")out+=(*e)["text"].get<std::string>();else terminal=*e;}
            check(out=="你好，这是测试。","mock UTF-8");
            check(terminal["usage"]["completion_tokens"]==8,"output accounting");
            auto bad=s.submit({std::string(1020,'x'),8,false});bad->await_start();
            check(drain(bad)["code"]=="invalid_request","context rejection");
            auto failure=s.submit({"__error__",8,false});failure->await_start();
            check(drain(failure)["code"]=="engine_error","exception isolation");
            auto recovery=s.submit({"recovery",1,false});recovery->await_start();check(drain(recovery)["type"]=="done","worker recovers");
        }
        {
            auto engine=std::make_unique<Gate>();auto* gate=engine.get();
            Scheduler s(c,std::move(engine));
            auto active=s.submit({"active",1,false});check(active->await_start(),"gate start");
            std::vector<std::shared_ptr<Job>> waiting;
            for(int i=0;i<4;++i)waiting.push_back(s.submit({"wait",1,false}));
            check(!s.submit({"overflow",1,false}),"queue must reject overflow");
            check(s.metrics()["queue_depth"]==4,"bounded queue");
            gate->release=true;check(drain(active)["type"]=="done","gate finish");
            for(auto& j:waiting)check(drain(j)["type"]=="done","FIFO jobs finish");
        }
        {
            c.queue_timeout_ms=30;
            auto engine=std::make_unique<Gate>();auto* gate=engine.get();
            Scheduler s(c,std::move(engine));
            auto a=s.submit({"active",1,false});a->await_start();auto b=s.submit({"wait",1,false});
            check(!b->await_start(),"queue timeout");gate->release=true;drain(a);
            check(drain(b)["code"]=="queue_timeout","queue timeout code");
        }
        {
            c.generation_timeout_ms=25;
            Scheduler s(c,std::make_unique<Gate>());auto j=s.submit({"timeout",1,false});j->await_start();
            check(drain(j)["code"]=="generation_timeout","generation timeout");
        }
        {
            c=Config{};c.mock_delay_ms=0;c.event_capacity=1;c.backpressure_timeout_ms=20;
            Scheduler s(c,make_engine(c));auto j=s.submit({"slow",20,false});j->await_start();
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            check(drain(j)["code"]=="slow_consumer","bounded output backpressure");
        }
        {
            c=Config{};Scheduler s(c,std::make_unique<Gate>());
            auto a=s.submit({"active",1,false});a->await_start();auto b=s.submit({"wait",1,false});
            s.stop();check(drain(a)["code"]=="shutdown","active shutdown");
            check(drain(b)["code"]=="shutdown","queued shutdown");
            check(!s.submit({"closed",1,false}),"reject after shutdown");
        }
        std::cout<<"PASS: output, UTF-8, context, failure recovery, bounded FIFO, timeouts, backpressure, shutdown\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
