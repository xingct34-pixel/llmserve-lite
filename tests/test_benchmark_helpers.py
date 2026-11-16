# LLMServe-Lite 本地推理服务 | tests/test_benchmark_helpers.py
# 给 run.py 的文件校验函数提供小文件，再检查它能否分块读出正确 SHA-256。
# 测试也覆盖路径不存在的情况，防止基准报告把没有读到的模型当作已校验。
#
# 阅读地图（按源码顺序）
#   1 限制校验内存


import hashlib  # 按文件或字节内容计算指纹
import importlib.util  # 按文件路径装载被测模块或检查依赖
from pathlib import Path  # 拼接目录、文件名并读取文件
import tempfile  # 为一次测试创建临时文件或目录
import unittest  # 组织测试用例并核对预期结果
from unittest.mock import patch  # 用可控对象替换依赖，构造可重复的测试条件

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).resolve().parents[1] / 'bench/run.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ArtifactHashTests(unittest.TestCase):


    # 模块 1：限制校验内存
    # 即使哈希正确，也要检查实现没有改成一次读入整个文件。


    # 核对哈希一致还不足以验证内存用量；测试还要让一次性读整文件的实现暴露出来。
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
