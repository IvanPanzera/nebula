import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'qwen')]
from download import download
from prepare_model import repack,verified_part,validate_recipe


class InstallerTests(unittest.TestCase):
    def test_pinned_recipe(self):
        validate_recipe(json.loads((HERE/'model_recipe.json').read_text()))

    def test_resumed_http_and_bad_hash(self):
        payload=b'0123456789'*4096
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);requests.append(start)
                self.send_response(206 if start else 200)
                if start:self.send_header('Content-Range',f'bytes {start}-{len(payload)-1}/{len(payload)}')
                self.send_header('Content-Length',str(len(payload)-start));self.end_headers();self.wfile.write(payload[start:])
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'weights';path.with_suffix('.download').write_bytes(payload[:713])
                url=f'http://127.0.0.1:{server.server_port}/weights'
                download(url,path,hashlib.sha256(payload).hexdigest(),len(payload),lambda _:None)
                self.assertEqual(path.read_bytes(),payload);self.assertEqual(requests,[713])
                download(url,path,hashlib.sha256(payload).hexdigest(),len(payload),lambda _:None)
                self.assertEqual(requests,[713])
                with self.assertRaises(ValueError):download(url,Path(d)/'bad','0'*64,len(payload),lambda _:None)
                self.assertFalse((Path(d)/'bad.download').exists())
        finally:server.shutdown();server.server_close();thread.join()

    def test_repack_checkpoint_and_provenance(self):
        class Identity:
            def convert(self,*args):raise AssertionError('Copy-only tensor must not be requantized')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.write_bytes(bytes(range(256))*4)
            part=dict(number=0,bytes=272,tensors=[dict(name='a',dims=[4,1,1,1],type=0,source_type=0,
                offset=0,source_offset=32,bytes=16,source_bytes=16),dict(name='b',dims=[4,1,1,1],type=0,
                source_type=0,offset=256,source_offset=96,bytes=16,source_bytes=16)])
            dest=root/'part.bin';digest=hashlib.sha256(source.read_bytes()[32:48]).hexdigest()
            dest.with_suffix('.partial').write_bytes(source.read_bytes()[32:48]+b'incomplete tail')
            dest.with_suffix('.resume.json').write_text(json.dumps(dict(recipe_sha256='test',done=[dict(name='a',sha256=digest)])))
            stamp=repack(source,part,dest,Identity(),'test')
            self.assertEqual(dest.read_bytes(),source.read_bytes()[32:48]+bytes(240)+source.read_bytes()[96:112])
            self.assertEqual(verified_part(dest,'test'),stamp)
            dest.with_suffix('.json').unlink() # crash between rename and final stamp
            self.assertEqual(verified_part(dest,'test')['sha256'],stamp['sha256'])
            dest.write_bytes(b'X'*272)
            with self.assertRaises(ValueError):verified_part(dest,'test')

if __name__=='__main__':unittest.main()
