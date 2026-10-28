"""Offline download safety tests. Payload is fake bytes in temporary directories, never a model."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

spec=importlib.util.spec_from_file_location('fetch_model',Path(__file__).resolve().parents[1]/'scripts/fetch_model.py')
fetch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
PAYLOAD=b'GGUF-unit-test-fake-bytes'
REV='a'*40
SHA=hashlib.sha256(PAYLOAD).hexdigest()

class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data)
        self.status=status
        self.headers=headers or {}

class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='llmserve-download-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.folder=self.root/'models'
        self.folder.mkdir()
        for name,value in [('ROOT',self.root),('LIMIT',1000),('MIN_BYTES',1)]:
            patcher=patch.object(fetch,name,value)
            patcher.start();self.addCleanup(patcher.stop)

    def network(self,url,timeout):
        if isinstance(url,str):
            self.assertEqual(url,f'https://huggingface.co/api/models/{fetch.REPO}/revision/main?blobs=true')
            metadata={'sha':REV,'siblings':[{'rfilename':fetch.NAME,'size':len(PAYLOAD),'lfs':{'sha256':SHA}}],
                      'cardData':{'license':'apache-2.0'}}
            return Response(json.dumps(metadata).encode())
        self.assertEqual(url.full_url,f'https://huggingface.co/{fetch.REPO}/resolve/{REV}/{fetch.NAME}')
        byte_range=url.get_header('Range')
        if byte_range:
            offset=int(byte_range.split('=')[1].split('-')[0])
            return Response(PAYLOAD[offset:],206,{'Content-Range':f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}'})
        return Response(PAYLOAD)

    def run_fetch(self,network=None,expected_code=0):
        with patch.object(fetch.urllib.request,'urlopen',side_effect=network or self.network), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            if expected_code:
                with self.assertRaises(SystemExit) as exc:
                    fetch.main()
                self.assertEqual(exc.exception.code,expected_code)
            else:
                fetch.main()
        return json.loads((self.folder/'manifest.json').read_text())

    def test_full_download_and_metadata_pin(self):
        record=self.run_fetch()
        self.assertEqual(record['status'],'verified')
        self.assertEqual(record['revision'],REV)
        self.assertEqual(record['actual_sha256'],SHA)
        self.assertEqual((self.folder/fetch.NAME).read_bytes(),PAYLOAD)
        self.assertFalse((self.folder/(fetch.NAME+'.part')).exists())

    def test_partial_resume_and_empty_retry(self):
        part=self.folder/(fetch.NAME+'.part')
        part.write_bytes(PAYLOAD[:5])
        self.assertEqual(self.run_fetch()['status'],'verified')
        (self.folder/fetch.NAME).unlink()
        part.write_bytes(b'')
        self.assertEqual(self.run_fetch()['status'],'verified')

    def test_checksum_mismatch_never_becomes_model(self):
        def wrong(url,timeout):
            response=self.network(url,timeout)
            return response if isinstance(url,str) else Response(b'X'*len(PAYLOAD))
        record=self.run_fetch(wrong,1)
        self.assertEqual(record['status'],'blocked')
        self.assertFalse((self.folder/fetch.NAME).exists())
        self.assertTrue((self.folder/(fetch.NAME+'.part')).exists())

    def test_refuse_server_ignoring_resume(self):
        (self.folder/(fetch.NAME+'.part')).write_bytes(PAYLOAD[:5])
        def ignored(url,timeout):
            return self.network(url,timeout) if isinstance(url,str) else Response(PAYLOAD)
        self.assertIn('resume range',self.run_fetch(ignored,1)['last_error'])
        self.assertEqual((self.folder/(fetch.NAME+'.part')).read_bytes(),PAYLOAD[:5])

    def test_metadata_failure_and_size_cap(self):
        def unavailable(url,timeout):
            raise urllib.error.URLError('offline test')
        self.assertEqual(self.run_fetch(unavailable,1)['status'],'blocked')
        with patch.object(fetch,'LIMIT',5):
            self.assertIn('size range',self.run_fetch(expected_code=1)['last_error'])
        self.assertFalse((self.folder/fetch.NAME).exists())

    def test_existing_corrupt_file_is_preserved(self):
        final=self.folder/fetch.NAME
        final.write_bytes(b'user-existing-bytes')
        record=self.run_fetch(expected_code=1)
        self.assertIn('refusing to replace',record['last_error'])
        self.assertEqual(final.read_bytes(),b'user-existing-bytes')

if __name__=='__main__':
    unittest.main(verbosity=2)
