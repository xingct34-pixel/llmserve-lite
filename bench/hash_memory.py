# LLMServe-Lite 本地推理服务 | bench/hash_memory.py
# 比较校验同一个模型文件的两种读法：整份读入，或复用 run.py 的分块哈希。
# 看相同 SHA-256 对应的进程峰值 RSS，可估计仅仅为了校验文件多占了多少内存。


# 模块：校验内存实验：比较整文件读取与分块哈希


"""Measure benchmark-client hashing memory in a fresh Linux process."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

from run import sha256_file


# 模块：执行入口：读取参数并组织该工具的完整流程


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['whole-file', 'stream'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('ru_maxrss units in this experiment require Linux')


    # 模块：切换读取策略
    # 两条路径校验同一份字节，内存差别来自读取方式。


    if args.mode == 'whole-file':
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    else:
        digest = sha256_file(args.model)
    print(
        json.dumps(
            {
                'mode': args.mode,
                'model_bytes': args.model.stat().st_size,
                'sha256': digest,
                'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'scope': 'benchmark client only, not inference server',
            }
        )
    )


if __name__ == '__main__':
    main()
