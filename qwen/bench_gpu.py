#!/usr/bin/env python3
"""Synchronised component timings with real packed matrices, not model tok/s."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
from quant import row_size
from test_gpu import Device,call

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent/'work/qwen'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',default='gpu_benchmark.json');args=ap.parse_args()
    header=json.loads((WORK/'headers/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf.json').read_text())
    tensors={t['name']:t for t in header['tensors']}
    output=json.loads((WORK/'mtp_probe/output.weight.json').read_text())['tensor']
    cases=[('output.weight',dict(output,file_offset=0),WORK/'mtp_probe/output.weight.bin')]
    for name in ('blk.48.attn_q.weight','blk.48.ffn_gate_exps.weight','blk.48.ffn_down_exps.weight','blk.48.nextn.eh_proj.weight'):
        cases.append((name,tensors[name],ROOT.parent/'models/qwen'/header['asset']['name']))
    records=[];rng=np.random.default_rng(9021)
    for name,t,path in cases:
        cols,rows=t['dims'][:2];nbytes=rows*row_size(t['type'],cols)
        begin=time.monotonic()
        with path.open('rb') as f:
            f.seek(t['file_offset']);packed=f.read(nbytes)
        read_seconds=time.monotonic()-begin
        if len(packed)!=nbytes:raise EOFError(name)
        begin=time.monotonic();dw=Device(np.frombuffer(packed,np.uint8));call('sync')
        upload_seconds=time.monotonic()-begin
        for nt in (1,3,16):
            dx=Device(rng.normal(size=(nt,cols)).astype(np.float32));out=Device(np.zeros((nt,rows),np.float32))
            call('matmul',out.p,dw.p,t['type'],rows,cols,dx.p,nt);call('sync')
            samples=[]
            for _ in range(5):
                start=time.monotonic()
                for _ in range(3):call('matmul',out.p,dw.p,t['type'],rows,cols,dx.p,nt)
                call('sync');samples.append((time.monotonic()-start)/3)
            record=dict(name=name,format=t['format'],rows=rows,cols=cols,tokens=nt,
                        median_ms=1000*float(np.median(samples)),min_ms=1000*min(samples),
                        bytes=nbytes,read_seconds=read_seconds,upload_seconds=upload_seconds)
            records.append(record);print(json.dumps(record),flush=True)
        del dw
    (WORK/args.report).write_text(json.dumps(dict(component_only=True,records=records),indent=2))


if __name__=='__main__':main()
