#!/usr/bin/env python3
"""Optional real GGUF smoke test. Never downloads weights; 77 means skipped."""
import hashlib
import http.client
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
def main():
    binary=str(Path(sys.argv[1]).resolve())
    manifest_path=ROOT/'models/manifest.json'
    if not manifest_path.exists():
        print('SKIP: no model manifest')
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
