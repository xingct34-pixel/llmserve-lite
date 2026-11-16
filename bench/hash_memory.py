# LLMServe-Lite 本地推理服务 | bench/hash_memory.py
# 功能：比较校验同一个模型文件的两种读法：整份读入，或复用 run.py 的分块哈希。
# 看相同 SHA-256 对应的进程峰值 RSS，可估计仅仅为了校验文件多占了多少内存。
# 目的：量出基准客户端校验大文件时的内存成本，避免把这部分算到模型头上。


# 模块：校验内存实验
# 功能：比较整文件读取与分块哈希
"""Measure benchmark-client hashing memory in a fresh Linux process."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

from run import sha256_file


# 模块：执行入口（main）
# 功能：读取参数并组织该工具的完整流程
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['whole-file', 'stream'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('ru_maxrss units in this experiment require Linux')


    # 切换读取策略
    # 两条路径校验同一份字节，内存差别来自读取方式。
    # 两种读法要给相同文件算出相同的 SHA-256。比较的是读文件进程的峰值内存；若在同一进程先整读再分块，先前已经抬高的历史峰值不会降回去。
    # 一次读完整个文件会同时保留整块字节，分块读取只需保存当前块和哈希状态。
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
