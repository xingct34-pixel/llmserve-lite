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

namespace serve {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
inline bool is_int(const Json& j) {
    if(!j.is_number_integer()) return false;
    if(j.is_number_unsigned()) return j.get<uint64_t>()<=uint64_t(std::numeric_limits<int>::max());
    const auto n=j.get<int64_t>();
    return n>=std::numeric_limits<int>::min() && n<=std::numeric_limits<int>::max();
}
struct Config {
    std::string engine="mock", host="127.0.0.1", model_path;
    int port=8080, threads=4, context_tokens=1024, max_output_tokens=128;
    int queue_capacity=4, http_threads=8, event_capacity=16;
    int queue_timeout_ms=30000, generation_timeout_ms=120000;
    int backpressure_timeout_ms=2000, mock_delay_ms=20;
    static Config load(const std::string&);
};
struct Request { std::string prompt; int max_tokens=128; bool stream=false; };
struct Generation {
    int prompt_tokens=0, output_tokens=0;
    std::string finish_reason="length";
    double prefill_ms=0, decode_ms=0;
};
class Engine {
public:
    virtual ~Engine()=default;
    // All engine methods are owned by the inference worker after startup.
    virtual Generation generate(const Request&, const std::function<bool(const std::string&)>&,
                                const std::function<bool()>&)=0;
    virtual std::string name() const=0;
};
std::unique_ptr<Engine> make_engine(const Config&);
std::string complete_utf8(std::string&,bool flush=false);
void log(const Json&);
double millis(Clock::time_point a, Clock::time_point b);

struct Job {
    const std::string id;
    const Request request;
    const Config config;
    const Clock::time_point submitted=Clock::now();
    Clock::time_point started{}, finished{}, first{};
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::condition_variable cv;
    bool running=false, done=false, terminal_read=false;
    std::deque<Json> events;
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

class Scheduler {
    Config config_;
    std::unique_ptr<Engine> engine_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::shared_ptr<Job> active_;
    bool stopping_=false;
    std::thread worker_;
    uint64_t sequence_=0, accepted_=0, rejected_=0, completed_=0, failed_=0, cancelled_=0, timed_out_=0;
    size_t queue_peak_=0;
    void run();
public:
    Scheduler(Config, std::unique_ptr<Engine>);
    ~Scheduler();
    std::shared_ptr<Job> submit(Request);
    void stop();
    Json metrics() const;
};
}
