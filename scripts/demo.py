# LLMServe-Lite 本地推理服务 | scripts/demo.py
# 功能：一次完整的本地客户端演示。用临时端口启动 CPU 服务，等 /readyz 成功，
# 再实际请求生成并读取响应。演示结束时关闭本脚本启动的进程，日志保存在 results。
# 目的：用临时端口跑一次实际 CPU 请求，让演示结束后能回收自己启动的服务。


# 模块：演示客户端
# 功能：调用本地推理接口
"""One-command local CPU demonstration; starts and stops only its own server."""
from pathlib import Path
import http.client, json, signal, socket, subprocess, tempfile, time

ROOT = Path(__file__).resolve().parents[1]


# 模块：执行入口（main）
# 功能：读取参数并组织该工具的完整流程
# 绑定端口 0 是让系统先挑可用端口，再把选出的端口写到临时配置。演示请求跟着这份配置连接，原有 cpu.json 不需要为每次演示改端口。
# 这个脚本扮演客户端。阅读时跟着 prompt 从 JSON 请求体到终端输出，实际生成仍在另一个服务进程。
def main():
    config = json.loads((ROOT / 'configs/cpu.json').read_text())
    # 端口填 0 让操作系统选择一个空闲端口，再写入临时配置，减少和已运行服务撞端口的机会。
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        config['port'] = sock.getsockname()[1]
    binary = ROOT / 'build/cpu/llmserve'
    if not binary.is_file():
        raise RuntimeError('Build the CPU preset first: bash scripts/check.sh cpu')
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='llmserve-demo-') as folder:
        path = Path(folder) / 'config.json'
        path.write_text(json.dumps(config))
        with (results / 'demo.server.log').open('w') as log:
            process = subprocess.Popen([str(binary), str(path)], cwd=ROOT, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Server startup failed; see results/demo.server.log')
                    conn = http.client.HTTPConnection('127.0.0.1', config['port'], timeout=2)
                    try:
                        conn.request('GET', '/readyz')
                        response = conn.getresponse()
                        response.read()
                        # 探针通过才发正式请求；这样“模型还在加载”和“生成请求失败”在日志中容易区分。
                        if response.status == 200:
                            break
                    except OSError:
                        pass
                    finally:
                        conn.close()
                    if time.monotonic() > deadline:
                        raise RuntimeError('Server readiness timeout')
                    time.sleep(0.05)
                conn = http.client.HTTPConnection('127.0.0.1', config['port'], timeout=150)
                try:
                    prompt = '你好，请用一句话介绍你自己。'
                    conn.request(
                        'POST',
                        '/v1/completions',
                        json.dumps({'prompt': prompt, 'max_tokens': 32}),
                        {'Content-Type': 'application/json'},
                    )
                    response = conn.getresponse()
                    record = json.loads(response.read())
                    assert response.status == 200, record
                    print('提示词：' + prompt)
                    print('实际模型输出：' + record['text'])
                    print(
                        json.dumps(
                            {'usage': record['usage'], 'timings': record['timings']},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    (results / 'demo.json').write_text(
                        json.dumps(record, ensure_ascii=False, indent=2) + '\n'
                    )
                finally:
                    conn.close()
            finally:
                if process.poll() is None:
                    # 先让服务按自己的退出流程回收 worker；超过等待上限才处理这个脚本刚创建的子进程。
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                assert process.returncode == 0, process.returncode


if __name__ == '__main__':
    main()
