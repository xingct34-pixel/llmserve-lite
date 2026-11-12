#!/usr/bin/env python3
"""Bounded scheduled-arrival CPU workload, including client scheduling delay and SLO goodput."""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import time

from run import ROOT, get, percentile, request, sha256_file, snapshot


# 模块：结果统计：拒绝和失败保留在分母内；达标请求须同时满足成功、首输出和总延迟条件


def summarize(rows, elapsed, ttft_slo_ms, latency_slo_ms):
    success = [r for r in rows if r.get('success')]
    good = [
        r
        for r in success
        if r.get('offered_ttft_ms', float('inf')) <= ttft_slo_ms
        and r['offered_latency_ms'] <= latency_slo_ms
    ]
    return {
        'offered': len(rows),
        'success': len(success),
        'rejected_429': sum(r.get('status') == 429 for r in rows),
        'client_capacity_drops': sum(r.get('error') == 'client_capacity' for r in rows),
        'other_failures': sum(
            not r.get('success') and r.get('status') != 429 and r.get('error') != 'client_capacity'
            for r in rows
        ),
        'slo_met': len(good),
        'slo_attainment': len(good) / len(rows) if rows else 0,
        'successful_requests_per_second': len(success) / elapsed,
        'goodput_requests_per_second': len(good) / elapsed,
        'success_ttft_p95_ms': percentile(
            [r['offered_ttft_ms'] for r in success if 'offered_ttft_ms' in r], 0.95
        ),
        'success_latency_p95_ms': percentile([r['offered_latency_ms'] for r in success], 0.95),
        'max_client_lag_ms': max((r.get('client_lag_ms', 0) for r in rows), default=0),
    }


# 模块：计划负载：固定间隔发送，交替长短输入；忙时丢弃客户端样本并明确计数


def run_workload(port, samples, interval, clients, output_tokens):
    origin = time.monotonic()
    rows, pending = [], []
    prompts = [
        '你好，请简短回答。',
        '请阅读以下说明并简短回答：' + '可靠服务需要明确资源边界。' * 12,
    ]

    def send(index, due):
        lag_ms = max(0, (time.monotonic() - due) * 1000)
        row = request(
            port, prompts[index % 2], output_tokens if index % 2 else min(8, output_tokens)
        )
        row.update(index=index, workload='long' if index % 2 else 'short', client_lag_ms=lag_ms)
        row['offered_latency_ms'] = lag_ms + row['latency_ms']
        if 'ttft_ms' in row:
            row['offered_ttft_ms'] = lag_ms + row['ttft_ms']
        # TPOT是请求级平均；SSE分段不等于token，不能把分段间隔冒充逐token时间。
        if row.get('success') and row['output_tokens'] > 1 and 'ttft_ms' in row:
            row['client_tpot_ms'] = (row['latency_ms'] - row['ttft_ms']) / (
                row['output_tokens'] - 1
            )
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        for index in range(samples):
            due = origin + index * interval
            time.sleep(max(0, due - time.monotonic()))
            # 不在ThreadPool里悄悄积累无限等待任务；记录未发送请求，避免隐藏过载。
            if sum(not f.done() for f in pending) >= clients:
                rows.append(
                    {
                        'index': index,
                        'success': False,
                        'error': 'client_capacity',
                        'offered_latency_ms': 0,
                        'status': None,
                    }
                )
            else:
                pending.append(pool.submit(send, index, due))
        rows.extend(f.result() for f in pending)
    return sorted(rows, key=lambda r: r['index']), time.monotonic() - origin


# 模块：运行入口：仅启动和停止自己创建的子进程，原始请求及资源信息留在本地


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build/mock/llmserve')
    parser.add_argument('--config', default='configs/mock.json')
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--interval', type=float, default=0.03)
    parser.add_argument('--clients', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--ttft-slo-ms', type=float, default=1000)
    parser.add_argument('--latency-slo-ms', type=float, default=5000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.samples <= 1000
        or not 1 <= args.clients <= 16
        or not math.isfinite(args.interval)
        or args.interval < 0
    ):
        parser.error('samples 1..1000, clients 1..16, interval >= 0 required')
    if not 0 < args.ttft_slo_ms < float('inf') or not 0 < args.latency_slo_ms < float('inf'):
        parser.error('SLOs must be positive finite milliseconds')
    cfg = json.loads((ROOT / args.config).read_text())
    if not 1 <= args.max_tokens <= cfg['max_output_tokens']:
        parser.error('max-tokens outside configured output bound')
    if cfg['engine'] == 'llama':
        model = ROOT / cfg['model_path']
        manifest = json.loads((ROOT / 'models/manifest.json').read_text())
        if sha256_file(model) != manifest['expected_sha256']:
            raise RuntimeError('real model hash mismatch')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        cfg['port'] = sock.getsockname()[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix('.config.json')
    config_path.write_text(json.dumps(cfg))
    binary = (ROOT / args.binary).resolve()
    with args.output.with_suffix('.server.log').open('w') as log:
        process = subprocess.Popen(
            [str(binary), str(config_path.resolve())], cwd=ROOT, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    get(cfg['port'], '/readyz')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError('server startup timeout')
                    time.sleep(0.05)
            rows, elapsed = run_workload(
                cfg['port'], args.samples, args.interval, args.clients, args.max_tokens
            )
            report = {
                'engine': cfg['engine'],
                'binary_sha256': sha256_file(binary),
                'slo': {'ttft_ms': args.ttft_slo_ms, 'latency_ms': args.latency_slo_ms},
                'elapsed_seconds': elapsed,
                'resources': snapshot(process.pid),
                'summary': summarize(rows, elapsed, args.ttft_slo_ms, args.latency_slo_ms),
                'scheduler': get(cfg['port'], '/metrics'),
                'rows': rows,
                'scope': 'client scheduled-arrival test; offered latency includes client lag; max16 senders; no external users',
            }
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=15)
        report['server_exit_code'] = process.returncode
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if process.returncode != 0:
            raise RuntimeError('server did not stop cleanly')
        print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
