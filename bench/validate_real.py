"""Validation only: existing service, cold first request plus repeat CPU measurements."""
from pathlib import Path
import hashlib
import http.client
import importlib.util
import json
import signal
import socket
import statistics
import subprocess
import tempfile
import threading
import time

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('benchmark',root/'bench/run.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
config=json.loads((root/'configs/cpu.json').read_text())
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));config['port']=sock.getsockname()[1]
binary=root/'build/cpu/llmserve'
output=root/'results/real-validation.json'
report={'engine':'llama','backend':'CPU','config':config,
        'model':json.loads((root/'models/manifest.json').read_text()),
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'prompt':'你好，请用一句话介绍你自己。','max_tokens':32,'requests':[],'resources':[]}
stop=threading.Event()
def one():
    start=time.monotonic();result={'success':False}
    conn=http.client.HTTPConnection('127.0.0.1',config['port'],timeout=150)
    try:
        conn.request('POST','/v1/completions',json.dumps({'prompt':report['prompt'],'max_tokens':32,'stream':True}),{'Content-Type':'application/json'})
        response=conn.getresponse();result['http_status']=response.status
        chunks=[];terminal=None;done=False
        if response.status!=200:raise RuntimeError(response.read().decode())
        for line in response:
            text=line.decode('utf-8').strip()
            if not text.startswith('data: '):continue
            value=text[6:]
            if value=='[DONE]':done=True;continue
            event=json.loads(value)
            if event.get('type')=='delta':
                if event['text'] and 'ttft_ms' not in result:result['ttft_ms']=(time.monotonic()-start)*1000
                chunks.append(event['text'])
            elif event.get('type')=='error':raise RuntimeError(event)
            elif event.get('type')=='done':terminal=event
        assert terminal and done and chunks
        result.update(success=True,text=''.join(chunks),usage=terminal['usage'],timings=terminal['timings'],finish_reason=terminal['finish_reason'])
        result['decode_tokens_per_second']=terminal['usage']['completion_tokens']/(terminal['timings']['decode_ms']/1000)
    except Exception as exc:result['error']=str(exc)
    finally:result['latency_ms']=(time.monotonic()-start)*1000;conn.close()
    return result
with tempfile.TemporaryDirectory(prefix='llmserve-real-validation-') as tmp:
    path=Path(tmp)/'cpu.json';path.write_text(json.dumps(config))
    with (root/'results/real-validation.server.log').open('w') as log:
        started=time.monotonic();process=subprocess.Popen([str(binary),str(path)],cwd=root,stdout=log,stderr=log)
        def sample():
            while not stop.wait(.1):
                try:report['resources'].append(bench.snapshot(process.pid))
                except FileNotFoundError:break
        monitor=threading.Thread(target=sample);monitor.start()
        try:
            for _ in range(1200):
                if process.poll() is not None:raise RuntimeError('real model startup failed')
                try:
                    bench.get(config['port'],'/readyz');break
                except OSError:time.sleep(.05)
            else:raise RuntimeError('startup timeout')
            report['model_load_to_ready_ms']=(time.monotonic()-started)*1000
            for index in range(21):
                result=one();result['index']=index;result['phase']='cold_first' if index==0 else 'warm_repeat'
                report['requests'].append(result)
                print(json.dumps({k:v for k,v in result.items() if k not in ('text','timings')},ensure_ascii=False),flush=True)
                if not result['success']:raise RuntimeError('real request failed')
                if report['resources'] and report['resources'][-1]['MemAvailable']<700*1024:
                    raise RuntimeError('WSL memory margin below 700 MiB; stopped')
            report['metrics']=bench.get(config['port'],'/metrics')
        except Exception as exc:report['error']=str(exc)
        finally:
            stop.set();monitor.join()
            try:report['resources'].append(bench.snapshot(process.pid))
            except FileNotFoundError:pass
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:process.wait(10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            report['exit_code']=process.returncode
            good=[r for r in report['requests'][1:] if r['success']]
            report['summary']={'successes':sum(r['success'] for r in report['requests']),
                               'failures':sum(not r['success'] for r in report['requests']),
                               'warm_samples':len(good),
                               'warm_ttft_p50_ms':bench.percentile([r['ttft_ms'] for r in good],.5),
                               'warm_ttft_p95_ms':bench.percentile([r['ttft_ms'] for r in good],.95),
                               'warm_decode_tokens_per_second_median':statistics.median([r['decode_tokens_per_second'] for r in good]) if good else None,
                               'peak_rss_kib':max((r['VmHWM'] for r in report['resources']),default=0),
                               'peak_process_swap_kib':max((r['VmSwap'] for r in report['resources']),default=0)}
            output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report['summary'],indent=2),flush=True)
assert not report.get('error') and report['exit_code']==0 and report['summary']['successes']==21,report.get('error')
