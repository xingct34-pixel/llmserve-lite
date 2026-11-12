import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from traffic import summarize


# 模块：达标率验收：成功但超时、429和客户端未发出样本都不能计为达标


class TrafficTests(unittest.TestCase):
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
