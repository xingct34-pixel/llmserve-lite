#!/usr/bin/env python3
"""Reproducible low-resource benchmark and soak. No external Python packages."""
import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]

def percentile(values, p):
    if not values:
        return None
    a = sorted(values)
    return a[min(len(a)-1, max(0, int((len(a)-1)*p+.5)))]

def get(port, route):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    try:
        conn.request('GET', route)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'{route}: {response.status}')
        return json.loads(response.read())
    finally:
        conn.close()

def request(port, prompt, output):
    row = {'status': None, 'success': False, 'output_tokens': 0}
    begin = time.monotonic()
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=150)
    try:
        conn.request('POST', '/v1/completions', json.dumps({'prompt': prompt, 'max_tokens': output, 'stream': True}),
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        row['status'] = response.status
        done, marker = None, False
        if response.status == 200:
            for line in response:
                text = line.decode('utf-8').strip()
                if not text.startswith('data: '):
                    continue
                text = text[6:]
                if text == '[DONE]':
                    marker = True
                    continue
                event = json.loads(text)
                if event.get('type') == 'delta' and event.get('text') and 'ttft_ms' not in row:
                    row['ttft_ms'] = (time.monotonic()-begin)*1000
                elif event.get('type') == 'done':
                    done = event
                elif event.get('type') == 'error':
                    row['error'] = event.get('code')
            if done and marker:
                row.update(success=True, output_tokens=done['usage']['completion_tokens'],
                           prompt_tokens=done['usage']['prompt_tokens'], timings=done['timings'])
        else:
            row['error'] = response.read().decode('utf-8')
    except Exception as exc:
        row['error'] = str(exc)
    finally:
        row['latency_ms'] = (time.monotonic()-begin)*1000
        conn.close()
    return row

def snapshot(pid):
    values = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:', 'Threads:')):
            values[line.split(':')[0]] = int(line.split()[1])
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'SwapFree:')):
            values[line.split(':')[0]] = int(line.split()[1])
    values['at_s'] = time.time()
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default='build/mock/llmserve')
    parser.add_argument('--config', default='configs/server.json')
    parser.add_argument('--output', default='results/mock-benchmark.json')
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1,2,4,8])
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--prompt', default='请简短介绍一个可靠的请求队列。')
    parser.add_argument('--soak-seconds', type=float, default=0)
    parser.add_argument('--interval', type=float, default=.2)
    args = parser.parse_args()
    if args.samples < 1 or any(c < 1 or c > 16 for c in args.concurrency) or args.interval < 0:
        parser.error('invalid samples, concurrency or interval')
    binary = (ROOT/args.binary).resolve()
    config = json.loads((ROOT/args.config).read_text())
    if args.max_tokens < 1 or args.max_tokens > config['max_output_tokens']:
        parser.error('max-tokens outside configured limit')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        config['port'] = sock.getsockname()[1]
    if config['engine'] == 'mock':
        config['mock_delay_ms'] = 2
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = output.with_suffix('.config.json')
    cfg_path.write_text(json.dumps(config, indent=2))
    report = {'engine': config['engine'], 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'config': config, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'dependencies': json.loads((ROOT/'dependencies.lock.json').read_text()),
              'uname': list(os.uname()), 'load_average': os.getloadavg(),
              'client_cpu_seconds': 0, 'resources': [], 'groups': [], 'soak': None}
    stop = threading.Event()
    cpu_begin = time.process_time()
    with open(output.with_suffix('.server.log'), 'w') as log:
        process = subprocess.Popen([str(binary), str(cfg_path)], cwd=ROOT, stdout=log, stderr=log)
        def monitor():
            while not stop.wait(1):
                try:
                    report['resources'].append(snapshot(process.pid))
                except FileNotFoundError:
                    break
        watcher = threading.Thread(target=monitor)
        try:
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    get(config['port'], '/readyz')
                    break
                except OSError:
                    time.sleep(.05)
            else:
                raise RuntimeError('startup timeout')
            report['resources'].append(snapshot(process.pid))
            watcher.start()
            for _ in range(3):
                row = request(config['port'], args.prompt, min(8,args.max_tokens))
                if not row['success']:
                    raise RuntimeError(f'warmup failed: {row}')
            if not args.soak_seconds:
                for concurrency in args.concurrency:
                    started = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                        rows = list(pool.map(lambda _: request(config['port'], args.prompt, args.max_tokens), range(args.samples)))
                    elapsed = time.monotonic()-started
                    good = [r for r in rows if r['success']]
                    group = {'concurrency': concurrency, 'wall_seconds': elapsed, 'requests': rows,
                             'successes': len(good), 'rejected': sum(r['status']==429 for r in rows),
                             'failures': sum(not r['success'] and r['status']!=429 for r in rows),
                             'latency_p50_ms': percentile([r['latency_ms'] for r in good],.5),
                             'latency_p95_ms': percentile([r['latency_ms'] for r in good],.95),
                             'ttft_p50_ms': percentile([r['ttft_ms'] for r in good if 'ttft_ms' in r],.5),
                             'ttft_p95_ms': percentile([r['ttft_ms'] for r in good if 'ttft_ms' in r],.95),
                             'output_tokens_per_second': sum(r['output_tokens'] for r in good)/elapsed,
                             'success_requests_per_second': len(good)/elapsed}
                    report['groups'].append(group)
                    print(json.dumps({k:v for k,v in group.items() if k!='requests'}), flush=True)
            else:
                started = time.monotonic()
                end = started + args.soak_seconds
                count = failures = 0
                health, recent = [], []
                while time.monotonic() < end:
                    row = request(config['port'], args.prompt, args.max_tokens)
                    count += 1
                    failures += not row['success']
                    recent.append(row)
                    if len(recent)>20:
                        recent.pop(0)
                    t = time.monotonic()
                    get(config['port'], '/healthz')
                    health.append((time.monotonic()-t)*1000)
                    if count % 200 == 0:
                        print(json.dumps({'soak_elapsed_seconds': time.monotonic()-started, 'requests': count,
                                          'failures': failures, 'rss_kib': snapshot(process.pid)['VmRSS']}), flush=True)
                    if process.poll() is not None:
                        raise RuntimeError('server crashed')
                    time.sleep(args.interval)
                report['soak'] = {'seconds': time.monotonic()-started, 'requests': count, 'failures': failures,
                                  'health_p95_ms': percentile(health,.95), 'last_requests': recent}
            report['metrics'] = get(config['port'], '/metrics')
            report['resources'].append(snapshot(process.pid))
        finally:
            stop.set()
            if watcher.ident:
                watcher.join()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            report['exit_code'] = process.returncode
            report['client_cpu_seconds'] = time.process_time()-cpu_begin
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
            with output.with_suffix('.resources.csv').open('w', newline='') as handle:
                fields = ['at_s','VmRSS','VmHWM','VmSwap','Threads','MemAvailable','SwapFree']
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(report['resources'])
    if report['exit_code'] != 0 or any(g['failures'] for g in report['groups']) or (report['soak'] and report['soak']['failures']):
        raise SystemExit('benchmark failed; inspect report')
    print('Saved', output, flush=True)

if __name__ == '__main__':
    main()
