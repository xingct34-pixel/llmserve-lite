# LLMServe-Lite 本地推理服务 | tests/integration.py
# 功能：启动本地 Mock 服务，通过真实 HTTP 连接检查参数、流式响应、拒绝、断连及退出。
# 目的：从客户端可见的行为验证整个服务，包括参数校验、断开连接和过载时的反馈。
# 读法：先读 server 和 call，掌握怎样启动服务、怎样发请求；再选一个 test，对照 main.cpp 的对应路由。
#
# 阅读顺序：准备服务和请求工具 → 检查指标与预算 → 验证 HTTP/SSE → 满载、恢复与退出。
#
# 运行方式：第一个命令行参数是服务可执行文件的路径，其余参数交给 unittest。
#
# 测试依赖的前提（需要和 configs/server.json、engine.cpp 里的 Mock 引擎对照确认）：
#   1. server.json 里 queue_capacity 为 4、max_output_tokens 为 128、context_tokens 为 1024
#   2. Mock 引擎对特殊提示词有固定行为：
#        __error__        一开始就失败
#        __error_after__  先输出一部分再失败
#        其他提示词       输出固定的一句话（如“你好”对应“你好，这是测试。”）
#   3. 每个测试启动自己的服务进程，互不影响，所以可以把队列、缓冲和超时调得很小来稳定触发边界


"""Standard-library HTTP tests. Starts/stops only child servers created by this test."""

import concurrent.futures
import contextlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

# 第一个命令行参数是服务可执行文件路径；用 pop 取走它，避免 unittest 把它当成测试名去解析
BINARY = str(Path(sys.argv.pop(1)).resolve())
# 项目根目录（本文件在 tests/ 下，往上一级），用来找 configs/ 里的配置文件
ROOT = Path(__file__).resolve().parents[1]


# 模块：隔离运行（server）
# 功能：为每组测试启动一个独立的服务进程，用完自动关闭，并检查它是否正常退出。
# 目的：每组测试有自己的服务和配置，可以把队列、缓冲、超时设得很小，可靠地触发边界。
# 步骤：
#   1. 读出 server.json，向系统申请一个空闲端口，再用 changes 覆盖想改的配置项
#   2. 把配置和日志放进临时目录，启动服务进程
#   3. 最多等 8 秒，反复访问 /readyz，返回 200 才算启动成功；进程中途退出就带上日志报错
#   4. 把端口和进程交给测试使用（yield）
#   5. 测试结束后发 SIGINT 请它退出，5 秒内没退出就强杀并判为失败；
#      退出码不是 0 同样判为失败，并附上服务日志
@contextlib.contextmanager
def server(**changes):
    cfg = json.loads((ROOT / 'configs/server.json').read_text())
    # 绑定端口 0 让系统分配一个空闲端口，取出端口号后立刻释放，再交给服务使用
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        cfg['port'] = sock.getsockname()[1]
    cfg.update(changes)
    with tempfile.TemporaryDirectory(prefix='llmserve-test-') as folder:
        config = Path(folder) / 'config.json'
        config.write_text(json.dumps(cfg))
        with open(Path(folder) / 'server.log', 'w+') as log:
            process = subprocess.Popen([BINARY, str(config)], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        raise AssertionError('server startup failed: ' + log.read())
                    try:
                        if call(cfg['port'], path='/readyz')[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.02)
                else:
                    # while 循环正常耗尽（没有 break）才会走到这里：8 秒内一直没就绪
                    raise AssertionError('server startup deadline')
                yield cfg['port'], process
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise AssertionError('shutdown exceeded five seconds')
                if process.returncode != 0:
                    log.seek(0)
                    raise AssertionError(f'server exit {process.returncode}: ' + log.read())


# 模块：发送一条请求（call）
# 功能：向本地服务发一次请求，返回（状态码, 响应正文）。
# 目的：测试可以选择传 JSON（data）或原始字节（raw），观察 HTTP 层实际返回的状态和正文。
# 说明：data 和 raw 都没传时发 GET，否则发 POST；raw 用来发送故意写坏的内容，比如不完整的 JSON。
def call(port, data=None, path='/v1/completions', raw=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=8)
    try:
        body = json.dumps(data).encode() if raw is None and data is not None else raw
        connection.request(
            'GET' if data is None and raw is None else 'POST',
            path,
            body,
            {'Content-Type': 'application/json'},
        )
        response = connection.getresponse()
        return response.status, response.read().decode('utf-8')
    finally:
        connection.close()


# 模块：等待指标（wait_metric）
# 功能：反复读取 /metrics，直到某个指标等于期望值，最多等约 2 秒（200 次 × 10 毫秒）。
# 目的：服务处理请求和测试线程是并行的，轮询到期望的指标再往下走，
#       比用固定睡眠去猜 worker 到了哪一步可靠得多。
def wait_metric(port, key, value):
    for _ in range(200):
        metrics = json.loads(call(port, path='/metrics')[1])
        if metrics[key] == value:
            return metrics
        time.sleep(0.01)
    raise AssertionError(f'{key} never reached {value}: {metrics}')


# 模块：测试集合（HTTPTests）
# 功能：从客户端角度逐项验收服务：指标、预算、配置、协议、故障、过载、超时、断连和退出。
# 目的：每个测试自己启动服务，失败时看对应的测试名就能知道是哪一类行为出了问题。
class HTTPTests(unittest.TestCase):


    # 模块：标准指标（test_prometheus_and_stage_metrics）
    # 功能：成功与失败分开计数，验证累计桶和文本格式。
    # 目的：一个成功请求加一个失败请求之后，完成数和失败数各为 1，耗时直方图只统计成功的那一个，
    #       Prometheus 文本里不泄露请求编号和提示词内容。
    def test_prometheus_and_stage_metrics(self):
        with server(mock_delay_ms=1) as (port, _):
            self.assertEqual(call(port, {'prompt': 'hello', 'max_tokens': 2})[0], 200)
            self.assertEqual(call(port, {'prompt': '__error__', 'max_tokens': 2})[0], 500)
            wait_metric(port, 'completed', 1)
            wait_metric(port, 'failed', 1)
            status, body = call(port, path='/metrics/prometheus')
            self.assertEqual(status, 200)
            # 标准文本格式：类型声明、计数器、耗时直方图（_count 和最后一个 +Inf 桶都只有 1 个成功请求）
            self.assertIn('# TYPE llmserve_completed_total counter', body)
            self.assertIn('llmserve_completed_total 1\n', body)
            self.assertIn('llmserve_failed_total 1\n', body)
            self.assertIn('llmserve_success_execution_seconds_count 1\n', body)
            self.assertIn('llmserve_success_execution_seconds_bucket{le="+Inf"} 1\n', body)
            # 不能用请求编号或提示词当标签，否则每个请求都会多出一条时间序列
            self.assertNotIn('request_id=', body)
            self.assertNotIn('hello', body)
            # JSON 指标里同样只有 1 个成功请求进入直方图，输出 token 数为那个成功请求的 2 个
            metrics = json.loads(call(port, path='/metrics')[1])
            self.assertEqual(metrics['histograms']['execution']['count'], 1)
            self.assertEqual(metrics['output_tokens'], 2)


    # 模块：预算背压（test_queued_output_budget）
    # 功能：活动任务不计入等待预算；超过预算拒绝，消费后容量恢复。
    # 目的：排队预算是 12：正在执行的请求不占预算，排队的请求占 10，再来一个 max_tokens 为 3 的请求，
    #       10+3 超过 12，应被拒绝。
    # 说明：这时队列里还有空位（容量 4，只有 1 个在等），所以这次 429 只可能来自预算超限，
    #       不是因为队列满了。这正是专门测输出预算时需要保留空位的原因。
    def test_queued_output_budget(self):
        with server(mock_delay_ms=25, queued_output_budget=12) as (port, _):
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                active = pool.submit(call, port, {'prompt': 'active', 'max_tokens': 12})
                wait_metric(port, 'active', 1)
                waiting = pool.submit(call, port, {'prompt': 'waiting', 'max_tokens': 10})
                wait_metric(port, 'queued_output_tokens', 10)
                self.assertEqual(call(port, {'prompt': 'over budget', 'max_tokens': 3})[0], 429)
                self.assertEqual(active.result()[0], 200)
                self.assertEqual(waiting.result()[0], 200)
            # 前面的请求都结束后，预算应当归零，新请求又能正常提交
            self.assertEqual(call(port, {'prompt': 'recovered', 'max_tokens': 2})[0], 200)
            wait_metric(port, 'queued_output_tokens', 0)


    # 模块：配置拒绝（test_new_config_validation）
    # 功能：新增的两个开关（前缀缓存、排队预算）同样执行类型与范围检查。
    # 目的：错误配置不能启动服务，必须以退出码 1 结束，并在标准错误里留下 startup_error。
    # 说明：这里依次试了前缀缓存取 1 和 -1，排队预算取 -1 和 1.5（不是整数）。
    #       前缀缓存取 1 就被拒绝，说明开启时可能有最小值要求，具体范围看 Config::load。
    def test_new_config_validation(self):
        for changes in [
            {'prefix_cache_tokens': 1},
            {'prefix_cache_tokens': -1},
            {'queued_output_budget': -1},
            {'queued_output_budget': 1.5},
        ]:
            with tempfile.TemporaryDirectory() as folder:
                config = json.loads((ROOT / 'configs/mock.json').read_text())
                config.update(changes)
                path = Path(folder) / 'invalid.json'
                path.write_text(json.dumps(config))
                result = subprocess.run([BINARY, str(path)], capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1, changes)
                self.assertIn(b'startup_error', result.stderr)


    # 模块：协议验收（test_json_stream_and_validation）
    # 功能：分别验证普通 JSON、SSE 流式响应，以及各种非法请求的状态码。
    # 目的：普通 JSON 和 SSE 有各自的成功结束方式；逐段核对响应，才能发现“返回 200 却少了尾段”的问题。
    def test_json_stream_and_validation(self):
        with server(mock_delay_ms=50) as (port, _):
            # 普通 JSON：文字被拼成完整的一句，并带有 token 用量
            status, body = call(port, {'prompt': '你好', 'max_tokens': 8})
            result = json.loads(body)
            self.assertEqual((status, result['text']), (200, '你好，这是测试。'))
            self.assertEqual(result['usage']['completion_tokens'], 8)
            # 流式：先读到第一段，再确认服务端此时还在继续生成（active 为 1），
            # 说明文字是边生成边送出来的，不是生成完才一起发。
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=8)
            conn.request(
                'POST',
                '/v1/completions',
                json.dumps({'prompt': 'hi', 'max_tokens': 32, 'stream': True}),
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            start = time.monotonic()
            first = response.readline().decode('utf-8')
            self.assertIn('delta', first)
            # 收到第一段 delta 之后，生成仍在进行
            self.assertEqual(json.loads(call(port, path='/metrics')[1])['active'], 1)
            # 读完剩下的内容，流结束标记 [DONE] 必须恰好出现一次
            remaining = response.read().decode('utf-8')
            self.assertEqual(remaining.count('[DONE]'), 1)
            conn.close()
            # 非法请求一律返回 400，逐项对应 main.cpp 的校验：
            #   提示词为空 / max_tokens 为 0 / 超过配置上限 128 / stream 不是布尔 / 多出未知字段 /
            #   max_tokens 是小数 / 提示词加生成预留超出上下文长度 / max_tokens 太大装不进 int（两种）/ 提示词含空字符
            for request in [
                {'prompt': ''},
                {'prompt': 'x', 'max_tokens': 0},
                {'prompt': 'x', 'max_tokens': 129},
                {'prompt': 'x', 'stream': 1},
                {'prompt': 'x', 'extra': True},
                {'prompt': 'x', 'max_tokens': 1.5},
                {'prompt': 'x' * 1000, 'max_tokens': 128},
                {'prompt': 'x', 'max_tokens': 2**64},
                {'prompt': 'x', 'max_tokens': 2**32 + 1},
                {'prompt': '\0'},
            ]:
                self.assertEqual(call(port, request)[0], 400, request)
            # 不是合法 JSON（缺少后半部分）或不是对象：400；请求体超过 16384 字节：413；不存在的路径：404
            self.assertEqual(call(port, raw=b'{')[0], 400)
            self.assertEqual(call(port, raw=b'[')[0], 400)
            self.assertEqual(call(port, raw=b'x' * 17000)[0], 413)
            self.assertEqual(call(port, path='/missing')[0], 404)


    # 模块：故障与恢复（test_faults_and_recovery）
    # 功能：分别模拟“一开始就失败”和“输出一部分后失败”，再检查服务还能继续工作。
    # 目的：验证两种失败的反馈方式不同：
    #   一开始就失败，响应头还没发出，能返回真实的 500；
    #   输出途中失败，响应头已经是 200，只能在流里放一个 error 事件，而且不能追加 [DONE]，
    #   否则客户端会把失败当成正常结束。
    def test_faults_and_recovery(self):
        with server(mock_delay_ms=2) as (port, _):
            self.assertEqual(call(port, {'prompt': '__error__'})[0], 500)
            status, body = call(
                port, {'prompt': '__error_after__', 'max_tokens': 8, 'stream': True}
            )
            self.assertEqual(status, 200)
            self.assertIn('event: error', body)
            self.assertNotIn('[DONE]', body)
            self.assertEqual(call(port, {'prompt': 'recovery', 'max_tokens': 1})[0], 200)


    # 模块：满队列与健康探针（test_overload_health_and_fifo）
    # 功能：让 1 个任务在执行、4 个任务排队，再提交第 6 个请求，检查拒绝、健康探针和先进先出。
    # 目的：只有让 worker 有活、队列有人等时再提交，才真正覆盖拒绝路径。
    #       此时服务还能不能快速回答健康检查，是另一项独立的要求。
    # 说明：1 个执行加 4 个排队，正好占满接入名额（queue_capacity+1），
    #       所以第 6 个请求在接入层就会被 429 拒绝，不会进入调度器。
    def test_overload_health_and_fifo(self):
        with server(mock_delay_ms=30) as (port, _):
            with concurrent.futures.ThreadPoolExecutor(5) as pool:
                futures = [pool.submit(call, port, {'prompt': 'active', 'max_tokens': 30})]
                wait_metric(port, 'active', 1)
                # 每提交一个排队请求，就等队列深度加 1，确保它们按这个顺序入队
                for i in range(4):
                    futures.append(
                        pool.submit(call, port, {'prompt': f'queued{i}', 'max_tokens': 1})
                    )
                    wait_metric(port, 'queue_depth', i + 1)
                self.assertEqual(call(port, {'prompt': 'overflow', 'max_tokens': 1})[0], 429)
                # 满载时健康检查也要在 0.5 秒内返回，说明 HTTP 线程没有被占满
                t = time.monotonic()
                self.assertEqual(call(port, path='/healthz')[0], 200)
                self.assertLess(time.monotonic() - t, 0.5)
                # 5 个请求全部成功，并且请求编号 1 到 5 与提交顺序一致，说明是先到先服务
                results = [f.result() for f in futures]
                self.assertTrue(all(r[0] == 200 for r in results))
                self.assertEqual(
                    [json.loads(r[1])['request_id'] for r in results], ['1', '2', '3', '4', '5']
                )


    # 模块：超时（test_timeouts）
    # 功能：分别验证生成超时和排队超时，都应返回 504。
    # 目的：排队超时发生在拿到 worker 以前，生成超时发生在开始推理以后。
    #       两种场景需要不同的负载，才能分别验证。
    def test_timeouts(self):
        # 生成超时：每步延迟 100 毫秒，生成最多只给 40 毫秒，一个任务单独运行就会超时
        with server(mock_delay_ms=100, generation_timeout_ms=40) as (port, _):
            self.assertEqual(call(port, {'prompt': 'slow', 'max_tokens': 3})[0], 504)
        # 排队超时：先让一个长任务占住 worker，第二个请求排队超过 40 毫秒就超时；长任务本身不受影响
        with server(mock_delay_ms=20, queue_timeout_ms=40) as (port, _):
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                active = pool.submit(call, port, {'prompt': 'active', 'max_tokens': 10})
                wait_metric(port, 'active', 1)
                self.assertEqual(call(port, {'prompt': 'queued', 'max_tokens': 1})[0], 504)
                self.assertEqual(active.result()[0], 200)


    # 模块：验证退出与恢复（test_disconnect_and_shutdown）
    # 功能：先让客户端中途断开，再在有任务时关闭服务，检查其余任务能否结束。
    # 目的：客户端提前关闭后，服务应能收回该任务并继续响应；
    #       有任务时关闭进程，要唤醒所有等待者，并在 5 秒内以退出码 0 干净退出。
    def test_disconnect_and_shutdown(self):
        with server(mock_delay_ms=15) as (port, process):
            # 第一部分：发起流式请求，读到第一行就把连接关掉，模拟客户端中途离开
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            conn.request(
                'POST',
                '/v1/completions',
                json.dumps({'prompt': 'disconnect', 'stream': True, 'max_tokens': 128}),
            )
            response = conn.getresponse()
            response.readline()
            response.close()
            conn.close()
            # 服务应发现断开并取消任务：执行中的任务回到 0，取消计数至少为 1，之后还能正常服务
            wait_metric(port, 'active', 0)
            metrics = json.loads(call(port, path='/metrics')[1])
            self.assertGreaterEqual(metrics['cancelled'], 1)
            self.assertEqual(call(port, {'prompt': 'after_disconnect', 'max_tokens': 1})[0], 200)
            # 第二部分：1 个执行中、1 个排队，此时发 SIGINT 请求退出。
            # 执行中的请求以 500 或 503 结束（取决于被打断时的状态），排队的一律 503，
            # 进程随后以退出码 0 结束。
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                active = pool.submit(call, port, {'prompt': 'active', 'max_tokens': 128})
                wait_metric(port, 'active', 1)
                queued = pool.submit(call, port, {'prompt': 'queued', 'max_tokens': 1})
                wait_metric(port, 'queue_depth', 1)
                process.send_signal(signal.SIGINT)
                self.assertIn(active.result()[0], (500, 503))
                self.assertEqual(queued.result()[0], 503)
                self.assertEqual(process.wait(5), 0)


    # 模块：HTTP 指标范围（test_http_metrics_scope）
    # 功能：发成功、无效参数、引擎错误各一次，对照 HTTP 层和调度器两边的计数。
    # 目的：HTTP 计数含三次请求；而调度器只接到通过入参校验的两次（空提示词在接入层就被 400 拦下，没进调度器）。
    def test_http_metrics_scope(self):
        with server(mock_delay_ms=1) as (port, _):
            self.assertEqual(call(port, {'prompt': 'x', 'max_tokens': 1})[0], 200)
            self.assertEqual(call(port, {'prompt': ''})[0], 400)
            self.assertEqual(call(port, {'prompt': '__error__'})[0], 500)
            metrics = wait_metric(port, 'http_responses_logged', 3)
            self.assertEqual(metrics['http_requests_started'], 3)
            self.assertEqual(metrics['http_status_4xx'], 1)
            self.assertEqual(metrics['http_status_5xx'], 1)
            self.assertEqual(metrics['accepted'], 2)
            self.assertEqual(metrics['completed'], 1)


    # 模块：配置拒绝（test_bad_config）
    # 功能：逐个把配置里的某一项改成不合法的值，确认服务拒绝启动。
    # 目的：错误配置要在启动阶段就以退出码 1 失败并留下 startup_error，而不是带着错误配置跑起来。
    # 逐项含义：线程数为 0；未知引擎名；监听地址不是本机回环地址；端口过小；
    #           等待队列容量过大；HTTP 线程数过少（说明 http_threads 有下限，具体下限看 Config::load）；
    #           多出未知配置项；线程数是小数；端口太大，绕回 8080 的写法也必须被拒绝，不能悄悄溢出。
    def test_bad_config(self):
        base = json.loads((ROOT / 'configs/server.json').read_text())
        for key, value in [
            ('threads', 0),
            ('engine', 'bad'),
            ('host', '0.0.0.0'),
            ('port', 12),
            ('queue_capacity', 100),
            ('http_threads', 2),
            ('typo', 1),
            ('threads', 1.5),
            ('port', 2**32 + 8080),
        ]:
            cfg = dict(base, **{key: value})
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as config:
                json.dump(cfg, config)
                config.flush()
                result = subprocess.run([BINARY, config.name], capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                self.assertIn(b'startup_error', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
