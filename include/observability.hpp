#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <string>

namespace serve {


// 模块：有界直方图：固定时间边界和桶数量，调用方负责线程同步
// counts记录每个区间的数量；导出时累加，生成Prometheus需要的累计桶。


struct Histogram {
    static constexpr std::array<double, 9> bounds{0.001, 0.005, 0.02, 0.1, 0.5, 1, 5, 30, 120};
    std::array<uint64_t, bounds.size() + 1> counts{};
    double sum = 0;
    uint64_t count = 0;
    void observe(double seconds);
    nlohmann::json json() const;
};


// 模块：标准指标导出：把调度快照转成固定名称的Prometheus文本


std::string prometheus(const nlohmann::json& snapshot);


}
