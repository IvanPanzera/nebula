#!/usr/bin/env python3
"""Offline Q4 candidate for target Q8 matrices; preserve F32 norms and routers.

The routed experts stay unchanged. This candidate permits a larger prefill and
expert cache, but must pass full-model quality comparisons before adoption.
"""
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from quant import dequant,row_size

ROOT=Path(__file__).resolve().parent.parent;WORK=ROOT/'work/qwen'
OUT=ROOT/'models/qwen/derived/target-core-Q4.bin'


def main():
    assets=[a for a in json.loads((WORK/'assets_manifest.json').read_text())['files'] if a['name'].startswith('UD-Q4_K_XL/')]
    tensors=[]
    for asset in assets:
        source=ROOT/'models/qwen'/asset['name'];stat=source.stat()
        stamp=json.loads(source.with_suffix('.gguf.verified.json').read_text())
        if stamp['sha256']!=asset['sha256'] or stat.st_size!=asset['size'] or stat.st_mtime_ns!=stamp['mtime_ns']:
            raise ValueError('unverified target source: '+asset['name'])
        h=json.loads((WORK/'headers'/(Path(asset['name']).name+'.json')).read_text())
        tensors.extend((asset,t) for t in h['tensors'] if t['type']==8 and '_exps.' not in t['name'] and 'ffn_gate_inp' not in t['name'])
    if OUT.exists() and OUT.with_suffix('.json').exists():
        from derived import core_q4
        _,info=core_q4(assets,verify_hash=True);print('Verified cached core Q4:',info['size'],flush=True);return
    library=WORK/'upstream/build/bin/libggml-base.so';ggml=C.CDLL(str(library));p,i,q,z=C.c_void_p,C.c_int,C.c_int64,C.c_size_t
    ggml.ggml_quantize_init.argtypes=[i];ggml.ggml_quantize_init.restype=None
    ggml.ggml_quantize_chunk.argtypes=[i,p,p,q,q,q,p];ggml.ggml_quantize_chunk.restype=z
    for kind in (12,20):ggml.ggml_quantize_init(kind)
    def convert(item):
        packed,rows,cols,kind=item;original=dequant(packed,8,rows,cols)
        if not np.isfinite(original).all():raise ValueError('non-finite source tensor')
        out=np.empty(rows*row_size(kind,cols),np.uint8)
        wrote=ggml.ggml_quantize_chunk(kind,original.ctypes.data,out.ctypes.data,0,rows,cols,None)
        if wrote!=out.nbytes:raise ValueError('quantizer size mismatch')
        delta=dequant(out,kind,rows,cols)-original
        return out.tobytes(),float(np.square(delta,dtype=np.float64).sum()),float(np.square(original,dtype=np.float64).sum()),float(np.max(abs(delta)))
    OUT.parent.mkdir(exist_ok=True);temporary=OUT.with_suffix('.partial');digest=hashlib.sha256();records=[];total=0
    begin=time.monotonic()
    with temporary.open('wb') as dest,ThreadPoolExecutor(max_workers=4) as pool:
        for index,(asset,t) in enumerate(tensors):
            rows=math.prod(t['dims'][1:]);cols=t['dims'][0];kind=12 if cols%256==0 else 20
            start=time.monotonic();sse=energy=maximum=0.;written=0
            def chunks(f):
                f.seek(t['file_offset'])
                for offset in range(0,rows,256):
                    nr=min(256,rows-offset);size=nr*row_size(8,cols);packed=f.read(size)
                    if len(packed)!=size:raise EOFError(t['name'])
                    yield packed,nr,cols,kind
            with (ROOT/'models/qwen'/asset['name']).open('rb') as source:
                for data,error,norm,peak in pool.map(convert,chunks(source),buffersize=4):
                    dest.write(data);digest.update(data);written+=len(data);sse+=error;energy+=norm;maximum=max(maximum,peak)
            record=dict(t,type=kind,format='Q4_K' if kind==12 else 'IQ4_NL',bytes=written,file_offset=total,offset=total,
                source_asset=asset['name'],source_tensor=t,rmse=(sse/(rows*cols))**.5,
                relative_rmse=(sse/energy)**.5 if energy else 0,max_abs_error=maximum,seconds=time.monotonic()-start)
            records.append(record);total+=written
            print(f'Core Q4 {index+1}/{len(tensors)}: {t["name"]}, relative RMSE {record["relative_rmse"]:.4f}',flush=True)
    temporary.replace(OUT)
    info=dict(tensors=records,source_assets=assets,size=total,mtime_ns=OUT.stat().st_mtime_ns,sha256=digest.hexdigest(),
        quantizer_commit='d1a92352cbd417fd840b4e765c0b82f5fe3d1d89',quantizer_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        requantized_from='Q8_0',routed_experts_changed=False,norms_and_routers_changed=False,
        source_bytes=sum(t['bytes'] for a,t in tensors),seconds=time.monotonic()-begin,quality_validated=False)
    stamp=OUT.with_suffix('.json');temp=stamp.with_suffix('.tmp');temp.write_text(json.dumps(info,indent=2));temp.replace(stamp)
    print(json.dumps({k:v for k,v in info.items() if k not in ('tensors','source_assets')},indent=2),flush=True)


if __name__=='__main__':main()
