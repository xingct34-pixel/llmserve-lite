#!/usr/bin/env python3
# LLMServe-Lite 本地推理服务 | tests/real_model.py
# 核对本地 GGUF 的摘要，然后启动 CPU 服务，交替发送普通和流式请求。
# 一次成功要看到实际文本或符合协议的流结束标志。缺少已验证模型时用返回码 77 告知 CTest 跳过。
#
# 阅读地图（按源码顺序）
#   1 验证两种响应


"""Optional real GGUF smoke test. Never downloads weights; 77 means skipped."""
import hashlib  # 按文件或字节内容计算指纹
import http.client  # 从客户端发送 HTTP 请求、读取状态与响应体
import json  # 在 Python 对象和 JSON 文本之间转换
from pathlib import Path  # 拼接目录、文件名并读取文件
import signal  # 向子进程传递退出信号
import socket  # 创建套接字或选择本地监听端口
import subprocess  # 启动外部程序，获取输出或退出码
import sys  # 访问解释器参数、模块搜索路径等运行环境
import tempfile  # 为一次测试创建临时文件或目录
import time  # 记录时间、设置等待间隔或截止时刻

ROOT=Path(__file__).resolve().parents[1]
def main():
    binary=str(Path(sys.argv[1]).resolve())
    manifest_path=ROOT/'models/manifest.json'
    if not manifest_path.exists():
        print('SKIP: no model manifest')
        # 返回 77 表示测试先决条件缺失，CTest 可以据此标记跳过。实际模型已加载并生成才算通过。
        return 77
    manifest=json.loads(manifest_path.read_text())
    model=ROOT/'models'/manifest.get('filename','missing.gguf')
    if manifest.get('status')!='verified' or not model.is_file():
        print('SKIP: official model unavailable / not verified')
        return 77
    digest=hashlib.sha256()
    with model.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    assert digest.hexdigest()==manifest['expected_sha256']
    config=json.loads((ROOT/'configs/cpu.json').read_text())
    config['model_path']=str(model)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        config['port']=sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='llmserve-real-') as directory:
        path=Path(directory)/'cpu.json'
        path.write_text(json.dumps(config))
        with open(ROOT/'results/real-smoke.server.log','w') as log:
            process=subprocess.Popen([binary,str(path)],cwd=ROOT,stdout=log,stderr=log)
            results=[]
            try:
                for _ in range(600):
                    if process.poll() is not None:
                        raise RuntimeError('real engine failed startup; inspect real-smoke.server.log')
                    try:
                        conn=http.client.HTTPConnection('127.0.0.1',config['port'],timeout=1)
                        conn.request('GET','/readyz')
                        ready=conn.getresponse().status==200
                        conn.close()
                        if ready:
                            break
                    except OSError:
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('real engine startup timeout')


                # 模块 1：验证两种响应
                # 同一提示词发十次，用 i 的奇偶交替选择 JSON 和 SSE，并分别检查结束条件。


                # 在同一个已启动的 CPU 服务上交替走流式和普通接口，观察真实权重能否完成两种返回路径。
                for i in range(10):
                    conn=http.client.HTTPConnection('127.0.0.1',config['port'],timeout=150)
                    conn.request('POST','/v1/completions',json.dumps({'prompt':'你好，请用一句话介绍自己。','max_tokens':32,'stream':bool(i%2)}))
                    response=conn.getresponse()
                    body=response.read().decode('utf-8')
                    assert response.status==200,(response.status,body)
                    if i%2:
                        assert body.count('[DONE]')==1 and 'event: error' not in body
                    else:
                        result=json.loads(body)
                        assert result['text'] and 0<result['usage']['completion_tokens']<=32
                    results.append({'index':i,'status':response.status,'body':body})
                    conn.close()
                (ROOT/'results/real-smoke.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(10)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait()
                assert process.returncode==0,process.returncode
    print('PASS: ten real-model requests, including five UTF-8 SSE streams')
    return 0

if __name__=='__main__':
    sys.exit(main())
