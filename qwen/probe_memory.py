#!/usr/bin/env python3
"""Allocate and touch the audited VRAM plan; this does not run target inference."""
import ctypes as C
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from derived import mtp_iq4

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent/'work/qwen'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mtp-iq4',action='store_true')
    parser.add_argument('--core-q4',action='store_true')
    parser.add_argument('--resident-experts',type=int,default=16)
    parser.add_argument('--prefill-batch',type=int,default=512);settings=parser.parse_args()
    if not 10<=settings.resident_experts<=128:parser.error('resident-experts must be 10..128')
    lib=C.CDLL(str(ROOT/'build/libqwen_gpu.so'));p,z,i=C.c_void_p,C.c_size_t,C.c_int
    for name,args,res in [('alloc',[z],p),('free',[p],None),('zero',[p,z],i),('sync',[],i),('memory',[p,p],i),('error',[i],C.c_char_p)]:
        f=getattr(lib,'qg_'+name);f.argtypes,f.restype=args,res
    def check(rc):
        if rc:raise RuntimeError(lib.qg_error(rc).decode())
    def memory():
        free,total=z(),z();check(lib.qg_memory(C.byref(free),C.byref(total)));return dict(free_bytes=free.value,total_bytes=total.value)
    def system_gpu():
        return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip()
    context=8192;resident=settings.resident_experts;allocations=[];cache=[0]*48
    manifest=json.loads((WORK/'assets_manifest.json').read_text())
    for asset in manifest['files']:
        h=json.loads((WORK/'headers'/(Path(asset['name']).name+'.json')).read_text())
        for t in h['tensors']:
            if settings.core_q4 and asset['name'].startswith('UD-Q4_K_XL/') and t['type']==8 and '_exps.' not in t['name'] and 'ffn_gate_inp' not in t['name']:
                cols=t['dims'][0];size=(cols//256*144 if cols%256==0 else cols//32*18)*math.prod(t['dims'][1:])
                t=dict(t,bytes=size)
            if settings.mtp_iq4 and t['name']=='blk.48.ffn_down_exps.weight':
                _,info=mtp_iq4(asset);t=info['tensor']
            if t['name']=='per_layer_token_embd.weight':continue
            layer=re.match(r'blk\.(\d+)\.',t['name'])
            if layer and int(layer[1])<48 and '_exps.' in t['name']:cache[int(layer[1])]+=t['bytes']//512*resident
            else:allocations.append(('weights',t['bytes']))
    allocations.extend(('weights',n) for n in cache)
    for il in range(49):
        if il<48 and il%4!=3:
            allocations.extend([('states',48*128*128*4)]*2+[('states',3*10240*4)]*2)
        else:
            allocations.extend([('states',context*512*2)]*2)
            if il<48:allocations.extend([('states',(context+3)//4*128*4)]+[('states',4*128*4)]*2)
        if il==1:allocations.extend([('states',9*10240*4)]*2)
    # Use the same allocation planner as the C loader, without opening a model.
    planner=C.CDLL(str(ROOT/'build/libqwen_probe.so')).qwen_probe_workspace_bytes
    planner.argtypes=[i,i];planner.restype=C.c_uint64
    workspace=planner(context,settings.prefill_batch)
    if not workspace:raise ValueError('invalid workspace configuration')
    allocations.append(('workspace',workspace))
    totals={key:sum(n for group,n in allocations if group==key) for key in ('weights','states','workspace')}
    arenas=[sum((n+255)//256*256 for group,n in allocations if group=='weights'),totals['states'],workspace]
    before=memory();system_before=system_gpu();pointers=[]
    if sum(totals.values())+256*1024**2>before['free_bytes']:raise MemoryError('insufficient free VRAM for allocation probe')
    try:
        for group,n in enumerate(arenas):
            ptr=lib.qg_alloc(n)
            if not ptr:raise MemoryError(f'CUDA allocation failed: {group}, {n} bytes')
            pointers.append(ptr);check(lib.qg_zero(ptr,n))
        check(lib.qg_sync());peak=memory();system_peak=system_gpu()
    finally:
        for ptr in pointers:lib.qg_free(ptr)
    result=dict(allocation_only=True,model_inference_test=False,context=context,resident_per_layer=resident,mtp_down_iq4=settings.mtp_iq4,
        core_q4_plan=settings.core_q4,prefill_batch=settings.prefill_batch,allocations=len(arenas),allocation_strategy='three native aligned arenas; workspace planned at runtime',bytes=totals,arena_bytes=arenas,total_GiB=sum(totals.values())/1024**3,
        cuda_before=before,cuda_peak=peak,cuda_after=memory(),system_before_csv=system_before,system_peak_csv=system_peak)
    report='memory_allocation_mtp_iq4.json' if settings.mtp_iq4 else 'memory_allocation_probe.json'
    if settings.core_q4:report=f'memory_core_q4_batch{settings.prefill_batch}_cache{resident}.json'
    (WORK/report).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
