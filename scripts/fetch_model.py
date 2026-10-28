#!/usr/bin/env python3
"""Fetch only the approved official Q4_K_M file, pin revision and verify LFS SHA-256."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Qwen/Qwen2.5-0.5B-Instruct-GGUF'
NAME = 'qwen2.5-0.5b-instruct-q4_k_m.gguf'
LIMIT = 600_000_000
MIN_BYTES = 100_000_000

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()

def main():
    folder = ROOT/'models'
    folder.mkdir(exist_ok=True)
    manifest_path = folder/'manifest.json'
    record = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        if shutil.disk_usage(folder).free < 2*1024**3:
            raise RuntimeError('need at least 2 GiB free disk')
        memory = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if memory['MemAvailable'] < 1024**2:
            raise RuntimeError('need at least 1 GiB WSL MemAvailable')
        if not record.get('revision'):
            api = f'https://huggingface.co/api/models/{REPO}/revision/main?blobs=true'
            with urllib.request.urlopen(api, timeout=10) as response:
                data = response.read(2_000_001)
            if len(data)>2_000_000:
                raise RuntimeError('metadata unexpectedly large')
            metadata = json.loads(data)
            item = next(v for v in metadata['siblings'] if v['rfilename']==NAME)
            record = {'repo':REPO, 'filename':NAME, 'revision':metadata['sha'],
                      'expected_bytes':item['size'], 'expected_sha256':item['lfs']['sha256'],
                      'license':metadata.get('cardData',{}).get('license','see repository LICENSE'),
                      'status':'metadata_pinned'}
            if not MIN_BYTES < record['expected_bytes'] <= LIMIT:
                raise RuntimeError('refusing file outside approved small-model size range')
            manifest_path.write_text(json.dumps(record,indent=2)+'\n')
        if record.get('repo') != REPO or record.get('filename') != NAME or not record.get('expected_sha256'):
            raise RuntimeError('unexpected manifest contents')
        if not MIN_BYTES < record['expected_bytes'] <= LIMIT:
            raise RuntimeError('manifest size exceeds approved model range')
        final = folder/NAME
        if final.exists():
            if final.stat().st_size==record['expected_bytes'] and digest(final)==record['expected_sha256']:
                record.update(status='verified',actual_bytes=final.stat().st_size,actual_sha256=record['expected_sha256'])
                manifest_path.write_text(json.dumps(record,indent=2)+'\n')
                print('Already verified:', final)
                return
            raise RuntimeError('existing model fails verification; refusing to replace it automatically')
        url = f'https://huggingface.co/{REPO}/resolve/{record["revision"]}/{NAME}'
        partial = folder/(NAME+'.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset>record['expected_bytes']:
            raise RuntimeError('partial file exceeds expected size')
        if offset != record['expected_bytes']:
            headers = {'Range':f'bytes={offset}-'} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=20) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError('server does not honor resume range; partial file preserved')
                started = time.monotonic()
                with partial.open('ab' if partial.exists() else 'xb') as handle:
                    total = offset
                    while True:
                        if time.monotonic()-started>1200:
                            raise RuntimeError('download exceeded 20 minute deadline; partial file preserved')
                        chunk = response.read(1024*1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > min(LIMIT,record['expected_bytes']):
                            raise RuntimeError('download exceeded approved size')
                        handle.write(chunk)
        size = partial.stat().st_size
        sha = digest(partial)
        if size != record['expected_bytes'] or sha != record['expected_sha256']:
            raise RuntimeError('download failed byte count / official LFS SHA-256 verification')
        partial.rename(final)
        record.update(status='verified',actual_bytes=size,actual_sha256=sha)
        record.pop('last_error',None)
        manifest_path.write_text(json.dumps(record,indent=2)+'\n')
        print('Verified:',final)
    except Exception as exc:
        record.update(status='blocked', repo=REPO, filename=NAME, last_error=str(exc),
                      last_attempt_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        manifest_path.write_text(json.dumps(record,indent=2)+'\n')
        print('Model unavailable:',exc,file=sys.stderr)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
