# LLMServe-Lite 本地推理服务 | bench/resource_matrix.py
# 让现有服务以不同上下文长度和等待队列容量启动，对每种组合发请求并采集资源。
# 上下文决定一次生成能容纳多少 token，队列容量决定能有多少请求等待，二者分别记录。
#
# 阅读地图（按源码顺序）
#   1 资源矩阵 → 2 执行入口 → 3 模型身份
#   4 逐组改变资源上限 → 5 过载与失败


# 模块 1：资源矩阵
# 功能：比较上下文与等待容量的六种配置


"""Bounded CPU context/queue resource experiments. Existing model only."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import cpu_matrix as matrix


# 模块 2：执行入口
# 功能：读取参数并组织该工具的完整流程


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['mock', 'llama'], default='mock')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 64:
        parser.error('samples must be in [1,64]')
    if args.output.exists():
        parser.error('choose a new output directory to preserve previous evidence')
    args.output.mkdir(parents=True)
    root = matrix.ROOT
    binary = root / ('build/cpu/llmserve' if args.engine == 'llama' else 'build/mock/llmserve')
    base = json.loads((root / 'configs/cpu.json').read_text())
    base.update(
        engine=args.engine,
        threads=2,
        max_output_tokens=64,
        mock_delay_ms=10 if args.engine == 'mock' else 0,
        http_threads=12,
    )
    report = {
        'engine': args.engine,
        'backend': 'CPU',
        'passed': False,
        'binary_sha256': matrix.bench.sha256_file(binary),
        'cases': [],
        'note': 'One worker; changing queue capacity does not add parallel model decoding.',
    }
    if args.engine == 'llama':
        model_hash = matrix.bench.sha256_file(root / base['model_path'])


        # 模块 3：模型身份
        # 相同文件名下面也能换权重，真实分支先验内容摘要，才能拿不同配置的结果互相比较。


        if model_hash != '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db':
            raise ValueError('model integrity check failed')
        report['model_sha256'] = model_hash
    prompt = '请简短解释请求队列。'
    try:
        # 3种上下文容量x2种等待容量=6例；每例独立启动服务，模型worker始终只有1个。
        # 配置改变用于观察资源/排队/拒绝，不表示实现了连续批处理或并行模型解码。


        # 模块 4：逐组改变资源上限
        # 每组都启动对应配置，同模型下才能比较上下文容量和等待队列的影响。


        # 三个上下文长度乘两个队列容量，共六组。前者限制一次请求能用的上下文，后者限制等待的请求数。
        for context in (512, 1024, 2048):
            for capacity in (1, 4):
                label = f'context-{context}-queue-{capacity}'
                with matrix.server(
                    binary,
                    base,
                    args.output,
                    label,
                    context_tokens=context,
                    queue_capacity=capacity,
                ) as (port, resources):
                    if not matrix.bench.request(port, prompt, 8)['success']:
                        raise RuntimeError('warmup failed')
                    started = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(8) as executor:
                        rows = list(
                            executor.map(
                                lambda _: matrix.bench.request(port, prompt, 16),
                                range(args.samples),
                            )
                        )
                    elapsed = time.monotonic() - started
                    # TTFT分位只看成功请求，必须连同rejected_429一起解读；拒绝不算引擎生成失败。
                    good = [row for row in rows if row['success']]


                    # 模块 5：过载与失败
                    # 容量不足的 429 单列；其他错误意味着本组实验出了问题，不能混进正常的吞吐结果。


                    bad = [row for row in rows if not row['success'] and row['status'] != 429]
                    if bad or not good:
                        raise RuntimeError('unexpected failure or no successful sample')
                    report['cases'].append(
                        {
                            'context_tokens': context,
                            'queue_capacity': capacity,
                            'clients': 8,
                            'samples': len(rows),
                            'success': len(good),
                            'rejected_429': sum(row['status'] == 429 for row in rows),
                            'other_failures': len(bad),
                            'elapsed_seconds': elapsed,
                            'successful_requests_per_second': len(good) / elapsed,
                            'successful_ttft_p95_ms': matrix.bench.percentile(
                                [row['ttft_ms'] for row in good], 0.95
                            ),
                            'requests': rows,
                            'resources': resources,
                            'metrics': matrix.bench.get(port, '/metrics'),
                        }
                    )
                print(label, 'passed', flush=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n'
        )


if __name__ == '__main__':
    main()
