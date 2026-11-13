# LLMServe-Lite 本地推理服务 | tests/test_benchmark_helpers.py
# 给 run.py 的文件校验函数提供小文件，再检查它能否分块读出正确 SHA-256。
# 测试也覆盖路径不存在的情况，防止基准报告把没有读到的模型当作已校验。


import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).resolve().parents[1] / 'bench/run.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ArtifactHashTests(unittest.TestCase):


    # 模块：限制校验内存
    # 即使哈希正确，也要检查实现没有改成一次读入整个文件。


    def test_hash_without_whole_file_read(self):
        payload = bytes(range(256)) * 10000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights'
            path.write_bytes(payload)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file allocation')):
                self.assertEqual(benchmark.sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                benchmark.sha256_file(Path(directory) / 'missing')


if __name__ == '__main__':
    unittest.main()
