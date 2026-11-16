# LLMServe-Lite 本地推理服务 | bench/cpu_matrix.py
# 在真实模型上分组改变 CPU 计算线程数和客户端并发度，复用 run.py 的请求与采样函数。
# 同一模型、同一组输入配不同参数，可以观察多等一会儿来自排队还是生成变慢。
#
# 阅读地图（按源码顺序）
#   1 真实CPU实验 → 2 服务生命周期 → 3 监控子进程
#   4 执行入口 → 5 改变引擎计算线程数 → 6 改变客户端并发
#   7 超时后的状态


# 模块 1：真实CPU实验
# 功能：组织线程数与客户端并发测量


"""Controlled real-model validation. Standard library only; no downloads."""
import argparse, concurrent.futures, contextlib, hashlib, http.client, importlib.util, json, signal, socket, subprocess, tempfile, threading, time
from pathlib import Path  # 拼接目录、文件名并读取文件

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bench', ROOT / 'bench/run.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


# 模块 2：服务生命周期
# 功能：启动测试子进程并保证退出清理


@contextlib.contextmanager
def server(binary, base, out, label, **changes):
    cfg = dict(base, **changes)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        cfg['port'] = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='llmserve-matrix-') as tmp:
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(cfg))
        samples = []
        stop = threading.Event()
        with (out / (label + '.log')).open('w') as log:
            p = subprocess.Popen([str(binary), str(path)], cwd=ROOT, stdout=log, stderr=log)


            # 模块 3：监控子进程
            # 上下文管理器负责成对启动和清理服务。统计中的内存要从服务进程采，不能读压力客户端自己。


            def monitor():
                while not stop.wait(0.2):
                    try:
                        sample = bench.snapshot(p.pid)
                        samples.append(sample)
                        if sample['MemAvailable'] < 700 * 1024:
                            p.send_signal(signal.SIGINT)
                            break
                    except (FileNotFoundError, ProcessLookupError):
                        break

            watcher = threading.Thread(target=monitor)
            watcher.start()
            resource = {'samples': samples}
            try:
                for _ in range(600):
                    if p.poll() is not None:
                        raise RuntimeError(label + ' startup exited')
                    try:
                        bench.get(cfg['port'], '/readyz')
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    raise RuntimeError('startup timeout')
                yield cfg['port'], resource
            finally:
                try:
                    samples.append(bench.snapshot(p.pid))
                except FileNotFoundError:
                    pass
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
                    try:
                        p.wait(15)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                stop.set()
                watcher.join()
                resource['exit_code'] = p.returncode
                resource['peak_rss_kib'] = max((s.get('VmHWM', 0) for s in samples), default=0)
                resource['peak_swap_kib'] = max((s.get('VmSwap', 0) for s in samples), default=0)
                resource['min_available_kib'] = min((s['MemAvailable'] for s in samples), default=0)
                assert p.returncode == 0, (label, p.returncode)


# 模块 4：执行入口
# 功能：读取参数并组织该工具的完整流程


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--soak-seconds', type=int, default=600)
    args = parser.parse_args()
    assert 0 <= args.soak_seconds <= 3600
    out = ROOT / 'results/cpu-matrix'
    out.mkdir(parents=True, exist_ok=True)
    binary = ROOT / 'build/cpu/llmserve'
    base = json.loads((ROOT / 'configs/cpu.json').read_text())
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    model = ROOT / base['model_path']
    actual_hash = bench.sha256_file(model)
    assert actual_hash == '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
    report = {
        'engine': 'llama',
        'backend': 'CPU',
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'model_sha256': actual_hash,
        'context_tokens': base['context_tokens'],
        'matrix': [],
        'concurrency': [],
    }
    short = '请按序号列出十个编程学习建议。'
    long = ('背景资料：服务器需要管理请求队列、超时、取消和有限的内存。' * 20) + '请总结这些要求。'
    try:


        # 模块 5：改变引擎计算线程数
        # 这里调的是服务处理一个请求时可用的 CPU 线程；后面的并发实验才改变客户端数量。


        # 同一模型分别用 1、2、4、6 个线程跑。每组里都分开记录长短输入和输出上限。
        for threads in [1, 2, 4, 6]:
            with server(binary, base, out, 'threads-' + str(threads), threads=threads) as (
                port,
                resources,
            ):
                assert bench.request(port, short, 8)['success']
                group = {'threads': threads, 'cases': [], 'resources': resources}
                for label, prompt in [('short', short), ('long', long)]:
                    for limit in [16, 64]:
                        rows = [bench.request(port, prompt, limit) for _ in range(2)]
                        assert all(r['success'] for r in rows), rows
                        group['cases'].append(
                            {'prompt_kind': label, 'output_limit': limit, 'requests': rows}
                        )
                report['matrix'].append(group)
            print('matrix threads', threads, 'passed', flush=True)
        with server(binary, base, out, 'concurrency', threads=4) as (port, resources):
            report['concurrency_resources'] = resources
            assert bench.request(port, short, 8)['success']


            # 模块 6：改变客户端并发
            # 固定服务端计算线程数再改客户端数量，这样能把多请求排队的影响和单次推理多线程区分开。


            for concurrency in [1, 2, 4, 8]:
                begin = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                    rows = list(pool.map(lambda _: bench.request(port, short, 32), range(12)))
                assert all(r['success'] or r['status'] == 429 for r in rows), rows
                elapsed = time.monotonic() - begin
                good = [r for r in rows if r['success']]
                report['concurrency'].append(
                    {
                        'clients': concurrency,
                        'samples': len(rows),
                        'success': len(good),
                        'rejected_429': sum(r['status'] == 429 for r in rows),
                        'elapsed_seconds': elapsed,
                        'successful_requests_per_second': len(good) / elapsed,
                        'successful_ttft_p95_ms': bench.percentile(
                            [r['ttft_ms'] for r in good], 0.95
                        ),
                        'requests': rows,
                    }
                )
                print(
                    'concurrency',
                    concurrency,
                    'success',
                    len(good),
                    'rejected',
                    sum(r['status'] == 429 for r in rows),
                    flush=True,
                )
            # Close an actual SSE response and prove the worker can accept another request.
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
            conn.request(
                'POST',
                '/v1/completions',
                json.dumps({'prompt': short, 'max_tokens': 128, 'stream': True}),
            )
            response = conn.getresponse()
            assert response.status == 200
            response.readline()
            response.close()
            conn.close()
            for _ in range(300):
                metrics = bench.get(port, '/metrics')
                if not metrics['active']:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError('disconnect did not release worker')
            assert metrics['cancelled'] >= 1, metrics
            assert bench.request(port, short, 8)['success']
            report['disconnect_recovery_passed'] = True


        # 模块 7：超时后的状态
        # 记录超时还不够，要观察 active 回落，确认计算任务确实不再占着 worker。


        with server(binary, base, out, 'timeout', threads=4, generation_timeout_ms=10) as (
            port,
            resources,
        ):
            row = bench.request(port, long, 64)
            assert not row['success'] and row['status'] in [200, 504], row
            for _ in range(100):
                metrics = bench.get(port, '/metrics')
                if metrics['timed_out'] >= 1 and not metrics['active']:
                    break
                time.sleep(0.02)
            assert metrics['timed_out'] >= 1 and not metrics['active']
            report['real_generation_timeout_passed'] = True
        if args.soak_seconds:
            with server(binary, base, out, 'soak', threads=4) as (port, resources):
                begin = time.monotonic()
                rows = []
                next_update = 60
                while time.monotonic() - begin < args.soak_seconds:
                    row = bench.request(port, short, 16)
                    rows.append(row)
                    assert row['success'], row
                    assert bench.get(port, '/healthz')['status'] == 'ok'
                    elapsed = time.monotonic() - begin
                    if elapsed >= next_update:
                        print(
                            'real soak seconds', round(elapsed), 'requests', len(rows), flush=True
                        )
                        next_update += 60
                    time.sleep(0.1)
                report['soak'] = {
                    'elapsed_seconds': time.monotonic() - begin,
                    'success': len(rows),
                    'failures': 0,
                    'requests': rows,
                    'resources': resources,
                }
        report['passed'] = True
    except Exception as exc:
        report['passed'] = False
        report['error'] = str(exc)
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('real matrix complete', flush=True)


if __name__ == '__main__':
    main()
