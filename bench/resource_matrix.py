"""Bounded CPU context/queue resource experiments. Existing model only."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import cpu_matrix as matrix


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
    base.update(engine=args.engine, threads=2, max_output_tokens=64,
                mock_delay_ms=10 if args.engine == 'mock' else 0, http_threads=12)
    report = {'engine': args.engine, 'backend': 'CPU', 'passed': False,
              'binary_sha256': matrix.bench.sha256_file(binary), 'cases': [],
              'note': 'One worker; changing queue capacity does not add parallel model decoding.'}
    if args.engine == 'llama':
        model_hash = matrix.bench.sha256_file(root / base['model_path'])
        if model_hash != '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db':
            raise ValueError('model integrity check failed')
        report['model_sha256'] = model_hash
    prompt = '请简短解释请求队列。'
    try:
        for context in (512, 1024, 2048):
            for capacity in (1, 4):
                label = f'context-{context}-queue-{capacity}'
                with matrix.server(binary, base, args.output, label,
                                   context_tokens=context, queue_capacity=capacity) as (port, resources):
                    if not matrix.bench.request(port, prompt, 8)['success']:
                        raise RuntimeError('warmup failed')
                    started = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(8) as executor:
                        rows = list(executor.map(lambda _: matrix.bench.request(port, prompt, 16), range(args.samples)))
                    elapsed = time.monotonic() - started
                    good = [row for row in rows if row['success']]
                    bad = [row for row in rows if not row['success'] and row['status'] != 429]
                    if bad or not good:
                        raise RuntimeError('unexpected failure or no successful sample')
                    report['cases'].append({
                        'context_tokens': context, 'queue_capacity': capacity,
                        'clients': 8, 'samples': len(rows), 'success': len(good),
                        'rejected_429': sum(row['status'] == 429 for row in rows),
                        'other_failures': len(bad), 'elapsed_seconds': elapsed,
                        'successful_requests_per_second': len(good) / elapsed,
                        'successful_ttft_p95_ms': matrix.bench.percentile(
                            [row['ttft_ms'] for row in good], .95),
                        'requests': rows, 'resources': resources,
                        'metrics': matrix.bench.get(port, '/metrics')})
                print(label, 'passed', flush=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
