#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <string>

namespace serve {


// 模块：有界直方图（Histogram）
// 作用：把耗时按固定的区间统计"落在每个区间的请求有多少个"，而不保存每个请求的明细。
// 固定时间边界和桶数量，所以无论服务运行多久，占用的内存都不变。
// 调用方负责线程同步：这个结构自己不加锁，Scheduler 在持有 mutex_ 时才会调用它。
// counts 记录每个区间的数量；导出时累加，生成 Prometheus 需要的累计桶。
// 通俗讲：像体育课测跑步成绩，不记每个人的具体秒数，只记"10 秒内几人、10~20 秒几人、20 秒以上几人"，
//         记录本永远只有这么几行，但依然能看出整体快慢分布。


struct Histogram {
    // 各桶的上边界，单位为秒：1 毫秒、5 毫秒、20 毫秒、100 毫秒、0.5 秒、1 秒、5 秒、30 秒、120 秒。
    // 前几档精细，后几档粗略，覆盖从"极快"到"很慢"的范围；constexpr 表示编译期就确定，运行时不可改
    static constexpr std::array<double, 9> bounds{0.001, 0.005, 0.02, 0.1, 0.5, 1, 5, 30, 120};
    // 每个区间的计数。长度比边界多 1，多出的最后一格用来装"超过最大边界（120 秒）"的数据；
    // 花括号 {} 让所有计数初始为 0
    std::array<uint64_t, bounds.size() + 1> counts{};
    // 所有观测值的总和，配合 count 可以算平均值
    double sum = 0;
    // 观测的总次数
    uint64_t count = 0;
    // 记录一次观测：找到 seconds 所属的区间，对应计数加 1，同时累加 sum 和 count
    void observe(double seconds);
    // 导出为 JSON，供 /metrics 接口使用
    nlohmann::json json() const;
};


// 模块：标准指标导出（prometheus）
// 作用：把调度器的指标快照（JSON）转成 Prometheus 能识别的文本格式，指标名称固定。
// 通俗讲：Prometheus 是常用的监控系统，只认它规定的文本格式，这个函数负责"翻译"，监控系统就能定时来抓取。


std::string prometheus(const nlohmann::json& snapshot);


}
