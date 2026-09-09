#!/usr/bin/env python3
"""Actual MTP weights, synthetic target features: independent full-graph check.

This validates integration and native cache updates; it is not an acceptance,
language-quality or end-to-end speed benchmark for the target model.
"""
import ctypes as C
import argparse
import json
import mmap
import os
from pathlib import Path
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','14')
import numpy as np
from quant import dequant_chunked,row_size
from native import Stats

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent/'work/qwen'


def sig(x):return 1/(1+np.exp(-x))
def rms(x,w):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w
def rope(x,pos):
    out=x.copy();angle=pos*1e7**(-np.arange(32,dtype=np.float64)/32)
    cs,sn=np.cos(angle).astype(np.float32),np.sin(angle).astype(np.float32)
    out[...,:32]=x[...,:32]*cs-x[...,32:64]*sn
    out[...,32:64]=x[...,:32]*sn+x[...,32:64]*cs
    return out


class Weights:
    def __init__(self,component=None,mtp_override=None):
        self.maps=[];self.tensors={};self.cache={}
        if component is not None:
            h=json.loads(component.with_suffix('.json').read_text())
            f=component.with_suffix('.bin').open('rb');mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);f.close();self.maps.append(mapped)
            for t in h['tensors']:self.tensors[t['name']]=(t,mapped)
            return
        probe=WORK/'mtp_probe'
        for name in ('token_embd.weight','output.weight'):
            stamp=json.loads((probe/(name+'.json')).read_text());path=probe/(name+'.bin')
            f=path.open('rb');mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);f.close();self.maps.append(mapped)
            self.tensors[name]=(dict(stamp['tensor'],file_offset=0),mapped)
        h=json.loads((WORK/'headers/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf.json').read_text())
        f=(ROOT.parent/'models/qwen'/h['asset']['name']).open('rb');mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);f.close();self.maps.append(mapped)
        for t in h['tensors']:self.tensors[t['name']]=(t,mapped)
        if mtp_override is not None:
            info=json.loads(mtp_override.with_suffix('.json').read_text());t=info['tensor']
            f=mtp_override.with_suffix('.bin').open('rb');mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);f.close();self.maps.append(mapped)
            self.tensors[t['name']]=(t,mapped)
    def matrix(self,name,expert=None):
        key=(name,expert)
        if key not in self.cache:
            t,mapped=self.tensors[name];cols=t['dims'][0];rows=t['dims'][1] if len(t['dims'])>1 else 1
            size=row_size(t['type'],cols)*rows;offset=t['file_offset']+(expert or 0)*size
            self.cache[key]=dequant_chunked(memoryview(mapped)[offset:offset+size],t['type'],rows,cols)
        return self.cache[key]
    def embedding(self,token):
        t,mapped=self.tensors['token_embd.weight'];size=row_size(t['type'],2560);offset=token*size
        return dequant_chunked(memoryview(mapped)[offset:offset+size],t['type'],1,2560).ravel()
    def close(self):
        self.cache.clear();self.tensors.clear()
        for mapped in self.maps:mapped.close()


class Reference:
    def __init__(self,w):self.w=w;self.keys=[];self.values=[]
    def mm(self,name,x,expert=None):return x@self.w.matrix(name,expert).T
    def mix(self,h,prefix):
        hn=rms(h,self.w.matrix(prefix+'_norm.weight').reshape(4,2560))
        low=self.mm(prefix+'_down.weight',hn.ravel())/4
        gate=sig(self.mm(prefix+'_up.weight',low*sig(low))).reshape(4,2560)
        mixed=(hn*gate).mean(0)
        inject=self.mm(prefix+'_inject.weight',hn.ravel()) if prefix+'_inject.weight' in self.w.tensors else None
        return mixed,inject
    def step(self,token,h,pos):
        p='blk.48.'
        e=rms(self.w.embedding(token),self.w.matrix(p+'nextn.enorm.weight').ravel())
        hn=rms(h,self.w.matrix(p+'nextn.hnorm.weight').reshape(4,2560))
        combined=np.concatenate([np.broadcast_to(e,hn.shape),hn],axis=-1)
        residual=self.mm(p+'nextn.eh_proj.weight',combined)
        x,inject=self.mix(residual,p+'hc_attn')
        qfull=self.mm(p+'attn_q.weight',x).reshape(24,2,256)
        q=rope(rms(qfull[:,0],self.w.matrix(p+'attn_q_norm.weight').ravel()),pos)
        k=rope(rms(self.mm(p+'attn_k.weight',x).reshape(2,256),self.w.matrix(p+'attn_k_norm.weight').ravel()),pos)
        v=self.mm(p+'attn_v.weight',x).reshape(2,256)
        self.keys.append(k.astype(np.float16));self.values.append(v.astype(np.float16))
        keys=np.array(self.keys).astype(np.float32);values=np.array(self.values).astype(np.float32)
        attn=[]
        for head in range(24):
            scores=keys[:,head//12]@q[head]/16
            probs=np.exp(scores-scores.max());probs/=probs.sum()
            attn.append(probs@values[:,head//12])
        attn=np.array(attn)*sig(qfull[:,1])
        result=self.mm(p+'attn_output.weight',attn.ravel())
        residual+=result[None,:]*2*sig(inject[:,None]/4)
        residual,ids=self.ffn(residual,p)
        head,_=self.mix(residual,p+'nextn.hc_head')
        print('  reference output projection',flush=True)
        logits=self.mm('output.weight',head)
        return logits,residual,ids

    def ffn(self,residual,p):
        x,inject=self.mix(residual,p+'hc_ffn')
        router=self.mm(p+'ffn_gate_inp.weight',x)
        ids=np.argsort(-router,kind='stable')[:10]
        weights=np.exp(router[ids]-router[ids].max());weights/=weights.sum()
        moe=np.zeros(2560,np.float32)
        for expert,weight in zip(ids,weights):
            gate=self.mm(p+'ffn_gate_exps.weight',x,int(expert))
            up=self.mm(p+'ffn_up_exps.weight',x,int(expert))
            moe+=self.mm(p+'ffn_down_exps.weight',gate*sig(gate)*up,int(expert))*weight
        sg=self.mm(p+'ffn_gate_shexp.weight',x);su=self.mm(p+'ffn_up_shexp.weight',x)
        shared=self.mm(p+'ffn_down_shexp.weight',sg*sig(sg)*su)
        moe+=shared*sig(self.mm(p+'ffn_gate_inp_shexp.weight',x))[0]
        residual+=moe[None,:]*2*sig(inject[:,None]/4)
        return residual,ids


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mtp-iq4',action='store_true')
    parser.add_argument('--prefill-tokens',type=int,choices=(512,2048),default=512);settings=parser.parse_args()
    results_dir=WORK/'mtp_probe'/('iq4' if settings.mtp_iq4 else '')
    results_dir.mkdir(exist_ok=True)
    lib=C.CDLL(str(ROOT/'build/libqwen_probe.so'));p,i=C.c_void_p,C.c_int
    signatures={'probe_open':([C.c_char_p,i],p),'probe_hidden':([p,p],i),'probe_get_hidden':([p,p],i),
                'mtp_step':([p,i,i,i,p],i),'logits':([p,p,i,i],i),'close':([p],None),'last_error':([],C.c_char_p),
                'checkpoint':([p],i),'restore':([p],i),'reset':([p],i),'probe_batch':([p,p,i,i,p],i),'get_stats':([p,p],None)}
    for n,(args,res) in signatures.items():f=getattr(lib,'qwen_'+n);f.argtypes,f.restype=args,res
    def check(code):
        if code:raise RuntimeError(lib.qwen_last_error().decode())
    index=WORK/'mtp_probe'/('probe-iq4.index' if settings.mtp_iq4 else 'probe.index')
    load_start=time.monotonic();handle=lib.qwen_probe_open(str(index).encode(),settings.prefill_tokens)
    if not handle:raise RuntimeError(lib.qwen_last_error().decode())
    load_seconds=time.monotonic()-load_start;print(f'MTP component load: {load_seconds:.3f}s',flush=True)
    override=ROOT.parent/'models/qwen/derived/mtp-down-IQ4_NL' if settings.mtp_iq4 else None
    w=Weights(mtp_override=override);ref=Reference(w);rng=np.random.default_rng(4838);records=[]
    try:
        for pos,token in enumerate((248045,74455,198)):
            h=(rng.normal(size=(4,2560))*0.8).astype(np.float32)
            check(lib.qwen_probe_hidden(handle,h.ctypes.data))
            begin=time.monotonic();actual_id=C.c_int();check(lib.qwen_mtp_step(handle,token,pos,1,C.byref(actual_id)))
            gpu_seconds=time.monotonic()-begin
            logits=np.empty(248320,np.float32);hidden=np.empty((4,2560),np.float32)
            check(lib.qwen_logits(handle,logits.ctypes.data,0,1));check(lib.qwen_probe_get_hidden(handle,hidden.ctypes.data))
            print(f'MTP step {pos}: GPU {gpu_seconds:.3f}s, computing independent reference',flush=True)
            begin=time.monotonic();expected,expected_h,ids=ref.step(token,h,pos)
            cpu_seconds=time.monotonic()-begin
            record=dict(position=pos,token=token,actual_top1=actual_id.value,reference_top1=int(expected.argmax()),
                        hidden_max_abs=float(np.max(np.abs(hidden-expected_h))),logits_max_abs=float(np.max(np.abs(logits-expected))),
                        hidden_rmse=float(np.sqrt(np.mean((hidden-expected_h)**2))),logits_rmse=float(np.sqrt(np.mean((logits-expected)**2))),
                        gpu_seconds=gpu_seconds,reference_seconds=cpu_seconds,reference_experts=ids.tolist())
            records.append(record);print(json.dumps(record),flush=True)
            (results_dir/'results.json').write_text(json.dumps(dict(synthetic_target_features=True,quality_benchmark=False,mtp_down_iq4=settings.mtp_iq4,steps=records),indent=2))
            np.testing.assert_allclose(hidden,expected_h,rtol=3e-3,atol=3e-3)
            np.testing.assert_allclose(logits,expected,rtol=3e-3,atol=1e-2)
            if actual_id.value!=int(expected.argmax()):raise AssertionError('MTP top-1 differs from independent reference')
        print('PASS: complete MTP CUDA graph agrees with independent NumPy on all three cache steps',flush=True)
        check(lib.qwen_checkpoint(handle))
        def trial(token):
            out=C.c_int();check(lib.qwen_mtp_step(handle,token,3,1,C.byref(out)))
            logits=np.empty(248320,np.float32);check(lib.qwen_logits(handle,logits.ctypes.data,0,1))
            return logits
        expected=trial(17)
        check(lib.qwen_restore(handle));trial(200)
        check(lib.qwen_restore(handle));actual=trial(17)
        np.testing.assert_array_equal(actual,expected)
        print('PASS: MTP checkpoint discards rejected KV and reproduces logits bit for bit',flush=True)
        samples=[];before=Stats();lib.qwen_get_stats(handle,C.byref(before))
        for pos in range(4,24):
            start=time.monotonic();out=C.c_int();check(lib.qwen_mtp_step(handle,17,pos,0,C.byref(out)))
            samples.append(time.monotonic()-start)
        after=Stats();lib.qwen_get_stats(handle,C.byref(after));read_bytes=after.metadata_read_bytes-before.metadata_read_bytes
        if read_bytes!=20*4:raise AssertionError('steady MTP unexpectedly reads routing IDs on CPU')
        (results_dir/'steady_timing.json').write_text(json.dumps(dict(component_only=True,
            median_ms=float(np.median(samples))*1000,steps_ms=[x*1000 for x in samples],metadata_read_bytes=read_bytes,
            routing_read_bytes=0),indent=2))
        print(f'MTP consecutive component steps: median {np.median(samples)*1000:.3f}ms',flush=True)
        tokens=np.array([248045,74455,198,17,200,34,268],np.int32)
        features=(rng.normal(size=(len(tokens),4,2560))*0.8).astype(np.float32)
        check(lib.qwen_reset(handle));start=time.monotonic()
        check(lib.qwen_probe_batch(handle,tokens.ctypes.data,len(tokens),0,features.ctypes.data))
        batch_seconds=time.monotonic()-start
        batch=np.empty((len(tokens),248320),np.float32)
        for row in range(len(tokens)):check(lib.qwen_logits(handle,batch[row].ctypes.data,row,1))
        check(lib.qwen_reset(handle));sequential=[];start=time.monotonic()
        for pos,token in enumerate(tokens):
            check(lib.qwen_probe_hidden(handle,features[pos].ctypes.data));out=C.c_int()
            check(lib.qwen_mtp_step(handle,int(token),pos,1,C.byref(out)))
            logits=np.empty(248320,np.float32);check(lib.qwen_logits(handle,logits.ctypes.data,0,1));sequential.append(logits)
        sequential_seconds=time.monotonic()-start
        np.testing.assert_array_equal(batch,np.array(sequential))
        (results_dir/'integration.json').write_text(json.dumps(dict(component_only=True,
            load_seconds=load_seconds,rollback_bit_exact=True,batch_tokens=len(tokens),batch_bit_exact=True,
            batch_seconds=batch_seconds,sequential_with_logits_readback_seconds=sequential_seconds),indent=2))
        print('PASS: seven-token MTP batch with compacted expert inputs matches sequential logits bit for bit',flush=True)
        count=settings.prefill_tokens;tokens=rng.integers(0,200000,size=count,dtype=np.int32)
        features=(rng.normal(size=(count,4,2560))*0.8).astype(np.float32)
        check(lib.qwen_reset(handle));start=time.monotonic()
        check(lib.qwen_probe_batch(handle,tokens.ctypes.data,count,0,features.ctypes.data))
        large_seconds=time.monotonic()-start;large_h=np.empty((4,2560),np.float32)
        check(lib.qwen_probe_get_hidden(handle,large_h.ctypes.data))
        out=C.c_int();check(lib.qwen_mtp_step(handle,17,count,0,C.byref(out)))
        next_logits=np.empty(248320,np.float32);check(lib.qwen_logits(handle,next_logits.ctypes.data,0,1))
        check(lib.qwen_reset(handle));start=time.monotonic()
        for pos in range(0,count,16):
            check(lib.qwen_probe_batch(handle,tokens[pos:pos+16].ctypes.data,16,pos,features[pos:pos+16].ctypes.data))
        chunks_seconds=time.monotonic()-start;chunks_h=np.empty_like(large_h)
        check(lib.qwen_probe_get_hidden(handle,chunks_h.ctypes.data));np.testing.assert_array_equal(large_h,chunks_h)
        check(lib.qwen_mtp_step(handle,17,count,0,C.byref(out)))
        replay_logits=np.empty(248320,np.float32);check(lib.qwen_logits(handle,replay_logits.ctypes.data,0,1))
        np.testing.assert_array_equal(next_logits,replay_logits)
        (results_dir/'large_prefill.json').write_text(json.dumps(dict(component_only=True,tokens=count,
            last_hidden_bit_exact=True,next_logits_bit_exact=True,large_seconds=large_seconds,
            chunks_with_extra_output_projections_seconds=chunks_seconds),indent=2))
        print(f'PASS: MTP prefill {count} matches {count//16} chunks of 16, including the next draft logits',flush=True)
    finally:
        lib.qwen_close(handle);w.close()


if __name__=='__main__':main()
