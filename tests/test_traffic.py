# LLMServe-Lite 本地推理服务 | tests/test_traffic.py
# 功能：给 traffic.py 的汇总函数传入事先知道结果的请求样本。
# 有人按时完成、有人迟到、有人收到 429；核对分母，才能正确解释压测报告的达标率。
# 目的：固定成功、迟到、拒绝与客户端丢弃的数量，核对报告里的不同分母。


import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from traffic import summarize


# 模块：达标率验收（TrafficTests）
# 功能：成功但超时、429和客户端未发出样本都不能计为达标
class TrafficTests(unittest.TestCase):


    # 模块：构造分母（test_denominators）
    # 把成功、超时和未发出混在同一组，逐一检查哪些算成功、哪些算达标以及分母保留了多少项。
    # 5 行代表 5 次计划到达，其中 429 与客户端自己丢弃的那次都占分母。成功有两次，按时的只有一次，所以即使成功率为 2/5，达标率仍只有 1/5。
    def test_denominators(self):
        rows = [
            {'success': True, 'status': 200, 'offered_ttft_ms': 10, 'offered_latency_ms': 100},
            {'success': True, 'status': 200, 'offered_ttft_ms': 50, 'offered_latency_ms': 100},
            {'success': False, 'status': 429},
            {'success': False, 'error': 'client_capacity', 'status': None},
            {'success': False, 'status': 500},
        ]
        report = summarize(rows, 2, 20, 200)
        self.assertEqual(report['slo_met'], 1)
        self.assertEqual(report['slo_attainment'], 0.2)
        self.assertEqual(report['goodput_requests_per_second'], 0.5)
        self.assertEqual(report['successful_requests_per_second'], 1)
        self.assertEqual(report['rejected_429'], 1)
        self.assertEqual(report['client_capacity_drops'], 1)
        self.assertEqual(report['other_failures'], 1)


if __name__ == '__main__':
    unittest.main()
