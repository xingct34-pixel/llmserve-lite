#include "observability.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

namespace serve {


// 模块：样本入桶：只接受有限非负秒数，内存与请求总数无关


void Histogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0)
        return;
    ++counts[size_t(std::lower_bound(bounds.begin(), bounds.end(), seconds) - bounds.begin())];
    ++count;
    sum += seconds;
}


// 模块：快照生成：所有桶使用累计计数，最后一个表示正无穷


nlohmann::json Histogram::json() const {
    nlohmann::json buckets = nlohmann::json::array();
    uint64_t cumulative = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        cumulative += counts[i];
        buckets.push_back(cumulative);
    }
    return {{"buckets", buckets}, {"count", count}, {"sum_seconds", sum}};
}


// 模块：文本格式：固定指标与单位，不把用户文本或请求ID放进标签


std::string prometheus(const nlohmann::json& snapshot) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const std::array<const char*, 14> counters{"accepted",
                                               "rejected",
                                               "completed",
                                               "failed",
                                               "cancelled",
                                               "timed_out",
                                               "prompt_tokens",
                                               "output_tokens",
                                               "cached_prompt_tokens",
                                               "http_requests_started",
                                               "http_responses_logged",
                                               "http_status_429",
                                               "http_status_4xx",
                                               "http_status_5xx"};
    for (const auto* key : counters) {
        if (!snapshot.contains(key))
            continue;
        out << "# HELP llmserve_" << key << "_total Cumulative " << key << ".\n"
            << "# TYPE llmserve_" << key << "_total counter\n"
            << "llmserve_" << key << "_total " << snapshot.at(key).get<uint64_t>() << '\n';
    }
    for (const char* key : {"queue_depth",
                            "queue_peak",
                            "active",
                            "inference_workers",
                            "queued_output_tokens",
                            "queued_output_budget"}) {
        out << "# HELP llmserve_" << key << " Current " << key << ".\n"
            << "# TYPE llmserve_" << key << " gauge\n"
            << "llmserve_" << key << ' ' << snapshot.at(key).get<uint64_t>() << '\n';
    }
    for (auto it = snapshot.at("histograms").begin(); it != snapshot.at("histograms").end(); ++it) {
        const auto name = "llmserve_success_" + it.key() + "_seconds";
        const auto& h = it.value();
        out << "# HELP " << name << " Successful request stage duration in seconds.\n"
            << "# TYPE " << name << " histogram\n";
        for (size_t i = 0; i <= Histogram::bounds.size(); ++i) {
            out << name << "_bucket{le=\"";
            if (i == Histogram::bounds.size())
                out << "+Inf";
            else
                out << Histogram::bounds[i];
            out << "\"} " << h.at("buckets").at(i).get<uint64_t>() << '\n';
        }
        out << name << "_sum " << h.at("sum_seconds").get<double>() << '\n'
            << name << "_count " << h.at("count").get<uint64_t>() << '\n';
    }
    return out.str();
}


}
