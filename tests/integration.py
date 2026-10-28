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

BINARY = str(Path(sys.argv.pop(1)).resolve())
ROOT = Path(__file__).resolve().parents[1]

@contextlib.contextmanager
def server(**changes):
    cfg = json.loads((ROOT / 'configs/server.json').read_text())
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
                    time.sleep(.02)
                else:
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

def call(port, data=None, path='/v1/completions', raw=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=8)
    try:
        body = json.dumps(data).encode() if raw is None and data is not None else raw
        connection.request('GET' if data is None and raw is None else 'POST', path, body,
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, response.read().decode('utf-8')
    finally:
        connection.close()

def wait_metric(port, key, value):
    for _ in range(200):
        metrics = json.loads(call(port, path='/metrics')[1])
        if metrics[key] == value:
            return metrics
        time.sleep(.01)
    raise AssertionError(f'{key} never reached {value}: {metrics}')

class HTTPTests(unittest.TestCase):
    def test_json_stream_and_validation(self):
        with server(mock_delay_ms=3) as (port, _):
            status, body = call(port, {'prompt': '你好', 'max_tokens': 8})
            result = json.loads(body)
            self.assertEqual((status, result['text']), (200, '你好，这是测试。'))
            self.assertEqual(result['usage']['completion_tokens'], 8)
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=8)
            conn.request('POST', '/v1/completions', json.dumps({'prompt': 'hi', 'max_tokens': 8, 'stream': True}))
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            start = time.monotonic()
            first = response.readline().decode('utf-8')
            self.assertIn('delta', first)
            remaining = response.read().decode('utf-8')
            self.assertGreater(time.monotonic() - start, .005)
            self.assertEqual(remaining.count('[DONE]'), 1)
            conn.close()
            for request in [{'prompt': ''}, {'prompt': 'x', 'max_tokens': 0}, {'prompt': 'x', 'max_tokens': 129},
                            {'prompt': 'x', 'stream': 1}, {'prompt': 'x', 'extra': True},
                            {'prompt': 'x', 'max_tokens': 1.5}, {'prompt': 'x'*1000, 'max_tokens': 128},
                            {'prompt': 'x', 'max_tokens': 2**64}, {'prompt': 'x', 'max_tokens': 2**32+1}, {'prompt': '\0'}]:
                self.assertEqual(call(port, request)[0], 400, request)
            self.assertEqual(call(port, raw=b'{')[0], 400)
            self.assertEqual(call(port, raw=b'[')[0], 400)
            self.assertEqual(call(port, raw=b'x'*17000)[0], 413)
            self.assertEqual(call(port, path='/missing')[0], 404)

    def test_faults_and_recovery(self):
        with server(mock_delay_ms=2) as (port, _):
            self.assertEqual(call(port, {'prompt': '__error__'})[0], 500)
            status, body = call(port, {'prompt': '__error_after__', 'max_tokens': 8, 'stream': True})
            self.assertEqual(status, 200)
            self.assertIn('event: error', body)
            self.assertNotIn('[DONE]', body)
            self.assertEqual(call(port, {'prompt': 'recovery', 'max_tokens': 1})[0], 200)

    def test_overload_health_and_fifo(self):
        with server(mock_delay_ms=30) as (port, _):
            with concurrent.futures.ThreadPoolExecutor(5) as pool:
                futures = [pool.submit(call, port, {'prompt': 'active', 'max_tokens': 30})]
                wait_metric(port, 'active', 1)
                for i in range(4):
                    futures.append(pool.submit(call, port, {'prompt': f'queued{i}', 'max_tokens': 1}))
                    wait_metric(port, 'queue_depth', i+1)
                self.assertEqual(call(port, {'prompt': 'overflow', 'max_tokens': 1})[0], 429)
                t = time.monotonic()
                self.assertEqual(call(port, path='/healthz')[0], 200)
                self.assertLess(time.monotonic()-t, .5)
                results = [f.result() for f in futures]
                self.assertTrue(all(r[0] == 200 for r in results))
                self.assertEqual([json.loads(r[1])['request_id'] for r in results], ['1','2','3','4','5'])

    def test_timeouts(self):
        with server(mock_delay_ms=100, generation_timeout_ms=40) as (port, _):
            self.assertEqual(call(port, {'prompt': 'slow', 'max_tokens': 3})[0], 504)
        with server(mock_delay_ms=20, queue_timeout_ms=40) as (port, _):
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                active = pool.submit(call, port, {'prompt': 'active', 'max_tokens': 10})
                wait_metric(port, 'active', 1)
                self.assertEqual(call(port, {'prompt': 'queued', 'max_tokens': 1})[0], 504)
                self.assertEqual(active.result()[0], 200)

    def test_disconnect_and_shutdown(self):
        with server(mock_delay_ms=15) as (port, process):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            conn.request('POST', '/v1/completions', json.dumps({'prompt': 'disconnect', 'stream': True, 'max_tokens': 128}))
            response = conn.getresponse()
            response.readline()
            response.close()
            conn.close()
            wait_metric(port, 'active', 0)
            metrics = json.loads(call(port, path='/metrics')[1])
            self.assertGreaterEqual(metrics['cancelled'], 1)
            self.assertEqual(call(port, {'prompt': 'after_disconnect', 'max_tokens': 1})[0], 200)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                active = pool.submit(call, port, {'prompt': 'active', 'max_tokens': 128})
                wait_metric(port, 'active', 1)
                queued = pool.submit(call, port, {'prompt': 'queued', 'max_tokens': 1})
                wait_metric(port, 'queue_depth', 1)
                process.send_signal(signal.SIGINT)
                self.assertIn(active.result()[0], (500,503))
                self.assertEqual(queued.result()[0], 503)
                self.assertEqual(process.wait(5), 0)

    def test_bad_config(self):
        base = json.loads((ROOT/'configs/server.json').read_text())
        for key, value in [('threads', 0), ('engine', 'bad'), ('host', '0.0.0.0'), ('port', 12),
                           ('queue_capacity', 100), ('http_threads', 2), ('typo', 1), ('threads', 1.5), ('port', 2**32+8080)]:
            cfg = dict(base, **{key: value})
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as config:
                json.dump(cfg, config)
                config.flush()
                result = subprocess.run([BINARY, config.name], capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                self.assertIn(b'startup_error', result.stderr)

if __name__ == '__main__':
    unittest.main(verbosity=2)
