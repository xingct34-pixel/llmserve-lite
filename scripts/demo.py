"""One-command local CPU demonstration; starts and stops only its own server."""
from pathlib import Path
import http.client,json,signal,socket,subprocess,tempfile,time

ROOT=Path(__file__).resolve().parents[1]
def main():
    config=json.loads((ROOT/'configs/cpu.json').read_text())
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));config['port']=sock.getsockname()[1]
    binary=ROOT/'build/cpu/llmserve'
    if not binary.is_file():raise RuntimeError('Build the CPU preset first: bash scripts/check.sh cpu')
    results=ROOT/'results';results.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='llmserve-demo-') as folder:
        path=Path(folder)/'config.json';path.write_text(json.dumps(config))
        with (results/'demo.server.log').open('w') as log:
            process=subprocess.Popen([str(binary),str(path)],cwd=ROOT,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+60
                while True:
                    if process.poll() is not None:raise RuntimeError('Server startup failed; see results/demo.server.log')
                    conn=http.client.HTTPConnection('127.0.0.1',config['port'],timeout=2)
                    try:
                        conn.request('GET','/readyz');response=conn.getresponse();response.read()
                        if response.status==200:break
                    except OSError:pass
                    finally:conn.close()
                    if time.monotonic()>deadline:raise RuntimeError('Server readiness timeout')
                    time.sleep(.05)
                conn=http.client.HTTPConnection('127.0.0.1',config['port'],timeout=150)
                try:
                    prompt='你好，请用一句话介绍你自己。'
                    conn.request('POST','/v1/completions',json.dumps({'prompt':prompt,'max_tokens':32}),{'Content-Type':'application/json'})
                    response=conn.getresponse();record=json.loads(response.read());assert response.status==200,record
                    print('提示词：'+prompt);print('实际模型输出：'+record['text'])
                    print(json.dumps({'usage':record['usage'],'timings':record['timings']},ensure_ascii=False,indent=2))
                    (results/'demo.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
                finally:conn.close()
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(15)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                assert process.returncode==0,process.returncode
if __name__=='__main__':main()
