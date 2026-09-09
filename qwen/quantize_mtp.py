#!/usr/bin/env python3
"""Convert only the MTP routed down projection from Q8_0 to IQ4_NL.

Uses the pinned external GGML quantizer offline. Production needs only the
resulting packed tensor and the existing C/CUDA IQ4_NL reader. Target weights
remain unchanged; draft acceptance must still be measured end to end.
"""
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from quant import dequant,row_size

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent/'work/qwen'
OUT=ROOT.parent/'models/qwen/derived'
NAME='mtp-down-IQ4_NL'


def main():
    OUT.mkdir(exist_ok=True)
    h=json.loads((WORK/'headers/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf.json').read_text())
    asset=h['asset'];source=ROOT.parent/'models/qwen'/asset['name'];stamp=json.loads(source.with_suffix('.gguf.verified.json').read_text())
    if stamp['sha256']!=asset['sha256'] or source.stat().st_size!=asset['size'] or source.stat().st_mtime_ns!=stamp['mtime_ns']:
        raise ValueError('MTP source is not verified')
    t=next(t for t in h['tensors'] if t['name']=='blk.48.ffn_down_exps.weight')
    if t['type']!=8 or t['dims']!=[640,2560,512]:raise ValueError('unexpected source down projection')
    dest=OUT/(NAME+'.bin');record=OUT/(NAME+'.json');rows=2560*512;cols=640
    if not (dest.exists() and record.exists()):
        library=WORK/'upstream/build/bin/libggml-base.so'
        ggml=C.CDLL(str(library));p,i,q,z=C.c_void_p,C.c_int,C.c_int64,C.c_size_t
        ggml.ggml_quantize_init.argtypes=[i];ggml.ggml_quantize_init.restype=None
        ggml.ggml_quantize_chunk.argtypes=[i,p,p,q,q,q,p];ggml.ggml_quantize_chunk.restype=z
        ggml.ggml_quantize_init(20)
        def convert(item):
            packed,nrows=item;original=dequant(packed,8,nrows,cols)
            if not np.isfinite(original).all():raise ValueError('non-finite source weights')
            out=np.empty(nrows*row_size(20,cols),np.uint8)
            written=ggml.ggml_quantize_chunk(20,original.ctypes.data,out.ctypes.data,0,nrows,cols,None)
            if written!=out.nbytes:raise ValueError('quantizer returned an unexpected byte count')
            restored=dequant(out,20,nrows,cols);delta=restored-original
            return out.tobytes(),float(np.square(delta,dtype=np.float64).sum()),float(np.square(original,dtype=np.float64).sum()),float(np.max(np.abs(delta)))
        def chunks(f):
            f.seek(t['file_offset'])
            for start in range(0,rows,4096):
                nrows=min(rows-start,4096);packed=f.read(nrows*row_size(8,cols))
                if len(packed)!=nrows*row_size(8,cols):raise EOFError('MTP tensor')
                yield packed,nrows
        start=time.monotonic();digest=hashlib.sha256();sse=energy=maximum=0.;written=0;next_progress=0
        temporary=dest.with_suffix('.partial')
        with source.open('rb') as f,temporary.open('wb') as out,ThreadPoolExecutor(max_workers=4) as pool:
            for data,error,norm,peak in pool.map(convert,chunks(f),buffersize=4):
                out.write(data);digest.update(data);sse+=error;energy+=norm;maximum=max(maximum,peak);written+=len(data)
                progress=100*written/(rows*row_size(20,cols))
                if progress>=next_progress:print(f'MTP down IQ4_NL: {progress:.0f}%',flush=True);next_progress+=10
        temporary.replace(dest);info=dict(tensor=dict(t,type=20,format='IQ4_NL',bytes=written,offset=0,file_offset=0),
            source_asset=asset,source_tensor=t,sha256=digest.hexdigest(),size=written,mtime_ns=dest.stat().st_mtime_ns,
            quantizer_commit='d1a92352cbd417fd840b4e765c0b82f5fe3d1d89',quantizer_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            requantized_from='Q8_0',target_weights_changed=False,rmse=(sse/(rows*cols))**0.5,
            relative_rmse=(sse/energy)**0.5,max_abs_error=maximum,seconds=time.monotonic()-start)
        temp_record=record.with_suffix('.tmp');temp_record.write_text(json.dumps(info,indent=2));temp_record.replace(record)
    info=json.loads(record.read_text())
    if dest.stat().st_size!=info['size'] or dest.stat().st_mtime_ns!=info['mtime_ns'] or info['source_asset']!=asset:
        raise ValueError('derived MTP tensor stamp/source mismatch')
    # Keep component diagnostics separate from the verified full-target index.
    probe_index=WORK/'mtp_probe/probe.index'
    if probe_index.exists():
        original=probe_index.read_text().splitlines();lines=[];inserted=False
        for line in original:
            if line.startswith('TENSOR\t') and not inserted:
                lines.append(f'FILE\t3\t{info["size"]}\t{dest}');inserted=True
            if line.startswith('TENSOR\t'+t['name']+'\t'):
                line=f'TENSOR\t{t["name"]}\t3\t0\t{info["size"]}\t20\t640\t2560\t512\t1'
            lines.append(line)
        (WORK/'mtp_probe/probe-iq4.index').write_text('\n'.join(lines)+'\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
