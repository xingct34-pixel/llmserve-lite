// LLMServe-Lite 本地推理服务 | src/observability.cpp
// 功能：把耗时记入固定区间，再导出各时间上限内的累计次数。
// 例：观测 4 ms、12 ms 各一次，导出时 ≤5 ms 的桶为 1，≤20 ms 的桶为 2。
// 目的：通过 /metrics/prometheus 看有多少次耗时落在目标以内，同时免去保存每一次历史耗时的开销。
//
// 阅读顺序：耗时样本入桶 → 生成累计快照 → 导出指标文本。


#include "observability.hpp"  // 直方图与 Prometheus 指标文本的公共接口
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

namespace serve {


// 模块 1：样本入桶
// 功能：只接受有限非负秒数，内存与请求总数无关


void Histogram::observe(double seconds) {


    // 模块 2：合法样本
    // NaN 没有可靠的大小关系，负时长没有业务意义，先过滤才能稳定地定位桶。


    if (!std::isfinite(seconds) || seconds < 0)
        return;
    ++counts[size_t(std::lower_bound(bounds.begin(), bounds.end(), seconds) - bounds.begin())];
    ++count;
    sum += seconds;
}


// 模块 3：快照生成
// 功能：所有桶使用累计计数，最后一个表示正无穷


nlohmann::json Histogram::json() const {
    nlohmann::json buckets = nlohmann::json::array();
    uint64_t cumulative = 0;


    // 模块 4：累计分布
    // 如果前两个桶各有 2 次和 3 次，第二个累计桶就要输出 5，表示不超过第二个上限的一共有 5 次。


    for (size_t i = 0; i < counts.size(); ++i) {
        cumulative += counts[i];
        buckets.push_back(cumulative);
    }
    return {{"buckets", buckets}, {"count", count}, {"sum_seconds", sum}};
}


// 模块 5：文本格式
// 功能：固定指标与单位，不把用户文本或请求ID放进标签


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
        // 指标导出按已知名字读取快照，缺字段就跳过；从这里追到 core.cpp 可找到相应计数何时变化。
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


    // 模块 6：输出耗时分布
    // 累计桶告诉监控系统“小于等于这个上限的有多少次”，最后一个桶覆盖全部样本。


    for (auto it = snapshot.at("histograms").begin(); it != snapshot.at("histograms").end(); ++it) {
        const auto name = "llmserve_success_" + it.key() + "_seconds";
        const auto& h = it.value();
        out << "# HELP " << name << " Successful request stage duration in seconds.\n"
            << "# TYPE " << name << " histogram\n";
        for (size_t i = 0; i <= Histogram::bounds.size(); ++i) {
            out << name << "_bucket{le=\"";
            // 最后一个桶表示正无穷，应覆盖所有被接纳的时长；它也是核对总样本数的办法。
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
