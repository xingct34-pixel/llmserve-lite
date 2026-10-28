"""Controlled real-model validation. Standard library only; no downloads."""
import argparse,concurrent.futures,contextlib,hashlib,http.client,importlib.util,json,signal,socket,subprocess,tempfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bench',ROOT/'bench/run.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

@contextlib.contextmanager
def server(binary,base,out,label,**changes):
    cfg=dict(base,**changes)
    with socket.socket() as s:s.bind(('127.0.0.1',0));cfg['port']=s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='llmserve-matrix-') as tmp:
        path=Path(tmp)/'config.json';path.write_text(json.dumps(cfg));samples=[];stop=threading.Event()
        with (out/(label+'.log')).open('w') as log:
            p=subprocess.Popen([str(binary),str(path)],cwd=ROOT,stdout=log,stderr=log)
            def monitor():
                while not stop.wait(.2):
                    try:
                        sample=bench.snapshot(p.pid);samples.append(sample)
                        if sample['MemAvailable']<700*1024:p.send_signal(signal.SIGINT);break
                    except (FileNotFoundError,ProcessLookupError):break
            watcher=threading.Thread(target=monitor);watcher.start()
            resource={'samples':samples}
            try:
                for _ in range(600):
                    if p.poll() is not None:raise RuntimeError(label+' startup exited')
                    try:bench.get(cfg['port'],'/readyz');break
                    except OSError:time.sleep(.05)
                else:raise RuntimeError('startup timeout')
                yield cfg['port'],resource
            finally:
                try:samples.append(bench.snapshot(p.pid))
                except FileNotFoundError:pass
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
                    try:p.wait(15)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                stop.set();watcher.join();resource['exit_code']=p.returncode
                resource['peak_rss_kib']=max((s.get('VmHWM',0) for s in samples),default=0)
                resource['peak_swap_kib']=max((s.get('VmSwap',0) for s in samples),default=0)
                resource['min_available_kib']=min((s['MemAvailable'] for s in samples),default=0)
                assert p.returncode==0,(label,p.returncode)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--soak-seconds',type=int,default=600)
    args=parser.parse_args();assert 0<=args.soak_seconds<=3600
    out=ROOT/'results/cpu-matrix';out.mkdir(parents=True,exist_ok=True)
    binary=ROOT/'build/cpu/llmserve';base=json.loads((ROOT/'configs/cpu.json').read_text())
    manifest=json.loads((ROOT/'models/manifest.json').read_text());model=ROOT/base['model_path']
    actual_hash=hashlib.sha256(model.read_bytes()).hexdigest()
    assert actual_hash=='74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
    report={'engine':'llama','backend':'CPU','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'model_sha256':actual_hash,'context_tokens':base['context_tokens'],'matrix':[],'concurrency':[]}
    short='请按序号列出十个编程学习建议。'
    long=('背景资料：服务器需要管理请求队列、超时、取消和有限的内存。'*20)+'请总结这些要求。'
    try:
        for threads in [1,2,4,6]:
            with server(binary,base,out,'threads-'+str(threads),threads=threads) as (port,resources):
                assert bench.request(port,short,8)['success']
                group={'threads':threads,'cases':[],'resources':resources}
                for label,prompt in [('short',short),('long',long)]:
                    for limit in [16,64]:
                        rows=[bench.request(port,prompt,limit) for _ in range(2)]
                        assert all(r['success'] for r in rows),rows
                        group['cases'].append({'prompt_kind':label,'output_limit':limit,'requests':rows})
                report['matrix'].append(group)
            print('matrix threads',threads,'passed',flush=True)
        with server(binary,base,out,'concurrency',threads=4) as (port,resources):
            report['concurrency_resources']=resources
            assert bench.request(port,short,8)['success']
            for concurrency in [1,2,4,8]:
                begin=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                    rows=list(pool.map(lambda _:bench.request(port,short,32),range(12)))
                assert all(r['success'] or r['status']==429 for r in rows),rows
                elapsed=time.monotonic()-begin;good=[r for r in rows if r['success']]
                report['concurrency'].append({'clients':concurrency,'samples':len(rows),'success':len(good),
                    'rejected_429':sum(r['status']==429 for r in rows),'elapsed_seconds':elapsed,
                    'successful_requests_per_second':len(good)/elapsed,'successful_ttft_p95_ms':bench.percentile([r['ttft_ms'] for r in good],.95),
                    'requests':rows})
                print('concurrency',concurrency,'success',len(good),'rejected',sum(r['status']==429 for r in rows),flush=True)
            # Close an actual SSE response and prove the worker can accept another request.
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=30)
            conn.request('POST','/v1/completions',json.dumps({'prompt':short,'max_tokens':128,'stream':True}))
            response=conn.getresponse();assert response.status==200;response.readline();response.close();conn.close()
            for _ in range(300):
                metrics=bench.get(port,'/metrics')
                if not metrics['active']:break
                time.sleep(.05)
            else:raise RuntimeError('disconnect did not release worker')
            assert metrics['cancelled']>=1,metrics
            assert bench.request(port,short,8)['success'];report['disconnect_recovery_passed']=True
        with server(binary,base,out,'timeout',threads=4,generation_timeout_ms=10) as (port,resources):
            row=bench.request(port,long,64);assert not row['success'] and row['status'] in [200,504],row
            for _ in range(100):
                metrics=bench.get(port,'/metrics')
                if metrics['timed_out']>=1 and not metrics['active']:break
                time.sleep(.02)
            assert metrics['timed_out']>=1 and not metrics['active'];report['real_generation_timeout_passed']=True
        if args.soak_seconds:
            with server(binary,base,out,'soak',threads=4) as (port,resources):
                begin=time.monotonic();rows=[];next_update=60
                while time.monotonic()-begin<args.soak_seconds:
                    row=bench.request(port,short,16);rows.append(row);assert row['success'],row
                    assert bench.get(port,'/healthz')['status']=='ok'
                    elapsed=time.monotonic()-begin
                    if elapsed>=next_update:print('real soak seconds',round(elapsed),'requests',len(rows),flush=True);next_update+=60
                    time.sleep(.1)
                report['soak']={'elapsed_seconds':time.monotonic()-begin,'success':len(rows),'failures':0,'requests':rows,'resources':resources}
        report['passed']=True
    except Exception as exc:
        report['passed']=False;report['error']=str(exc);raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('real matrix complete',flush=True)
if __name__=='__main__':main()
