// LLMServe-Lite 本地推理服务 | tests/observability_tests.cpp
// 功能：把恰好等于桶上界、超过上界以及无效的耗时送进 Histogram。
// 对照 json() 导出的累计桶与总次数，能查出一个请求被重复记录或落错桶的问题。
// 目的：用边界值检查监控是否把每个耗时算在正确区间，防止报表可读但结果错误。


#include "observability.hpp"  // 直方图与 Prometheus 指标文本的公共接口
#include <cassert>
#include <cmath>
#include <limits>


// 模块：边界验收（main）
// 功能：检查等于边界、越界、无效样本与累计桶总数
int main() {
    serve::Histogram h;


    // 构造边界值
    // 一次送入刚好等于桶上界的值，再送一个大于最大上界的值。既能查等号放哪边，也能查最后的无穷桶。
    // 0.001 秒恰等于 1 ms 上界，应进第一个桶；0.002 秒才进下一档。一起测试这两个值能抓到把包含边界误写成严格小于的错误。
    for (double value : {0.0, 0.001, 0.002, 0.1, 121.0})
        h.observe(value);
    h.observe(-1);
    h.observe(std::numeric_limits<double>::infinity());
    h.observe(std::numeric_limits<double>::quiet_NaN());
    const auto data = h.json();
    assert(data["count"] == 5);
    assert(data["buckets"][0] == 2);
    assert(data["buckets"][1] == 3);
    assert(data["buckets"][3] == 4);
    assert(data["buckets"].back() == 5);
    assert(std::abs(data["sum_seconds"].get<double>() - 121.103) < 1e-9);
    return 0;
}
