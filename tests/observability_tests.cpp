// LLMServe-Lite 本地推理服务 | tests/observability_tests.cpp
// 把恰好等于桶上界、超过上界以及无效的耗时送进 Histogram。
// 对照 json() 导出的累计桶与总次数，能查出一个请求被重复记录或落错桶的问题。


#include "observability.hpp"
#include <cassert>
#include <cmath>
#include <limits>


// 模块：边界验收：检查等于边界、越界、无效样本与累计桶总数


int main() {
    serve::Histogram h;
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
