#!/usr/bin/env python3
"""Real target GDN/QSA+MoE layers, independent NumPy and rollback checks."""
import ctypes as C
import argparse
import json
from pathlib import Path
import time
from probe_mtp import Weights,Reference,rms,sig,rope,np
from native import Stats

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'work/qwen/layer_probe'


class LayerReference(Reference):
    def __init__(self,w,layer):
        super().__init__(w);self.layer=layer;self.conv=[];self.state=np.zeros((48,128,128),np.float32)
    def step(self,h,pos):
        p=f'blk.{self.layer}.';residual=h.copy();x,inject=self.mix(residual,p+'hc_attn')
        if self.layer%4!=3:
            raw=self.mm(p+'attn_qkv.weight',x)
            full=np.array([np.zeros_like(raw)]*(3-len(self.conv))+self.conv+[raw])
            convolved=(full.T*self.w.matrix(p+'ssm_conv1d.weight')).sum(-1)
            convolved*=sig(convolved);self.conv=(self.conv+[raw])[-3:]
            q=convolved[:2048].reshape(16,128);k=convolved[2048:4096].reshape(16,128)
            q=q/np.sqrt((q*q).sum(-1,keepdims=True)+1e-6);k=k/np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
            q=q[np.arange(48)%16];k=k[np.arange(48)%16];v=convolved[4096:].reshape(48,128)
            alpha=self.mm(p+'ssm_alpha.weight',x);beta=sig(self.mm(p+'ssm_beta.weight',x))
            decay=np.exp(self.w.matrix(p+'ssm_a').ravel()*np.logaddexp(0,alpha+self.w.matrix(p+'ssm_dt.bias').ravel()))
            self.state*=decay[:,None,None]
            delta=(v-np.einsum('hkv,hk->hv',self.state,k))*beta[:,None]
            self.state+=k[:,:,None]*delta[:,None,:]
            y=np.einsum('hkv,hk->hv',self.state,q)/np.sqrt(np.float32(128))
            y=rms(y,self.w.matrix(p+'ssm_norm.weight').ravel())
            # Official Flash-Next config overrides generic GDN's output gate.
            y=y.ravel()*sig(self.mm(p+'attn_gate.weight',x));result=self.mm(p+'ssm_out.weight',y)
        else:
            # At these short positions QSA retains every completed block and
            # its causal tail. Long selective masks have a separate CUDA test.
            qfull=self.mm(p+'attn_q.weight',x).reshape(24,2,256)
            q=rope(rms(qfull[:,0],self.w.matrix(p+'attn_q_norm.weight').ravel()),pos)
            k=rope(rms(self.mm(p+'attn_k.weight',x).reshape(2,256),self.w.matrix(p+'attn_k_norm.weight').ravel()),pos)
            v=self.mm(p+'attn_v.weight',x).reshape(2,256)
            self.keys.append(k.astype(np.float16));self.values.append(v.astype(np.float16));attn=[]
            keys=np.array(self.keys).astype(np.float32);values=np.array(self.values).astype(np.float32)
            for head in range(24):
                scores=keys[:,head//12]@q[head]/16;probs=np.exp(scores-scores.max());probs/=probs.sum()
                attn.append(probs@values[:,head//12])
            y=np.array(attn)*sig(qfull[:,1]);result=self.mm(p+'attn_output.weight',y.ravel())
        residual+=result[None,:]*2*sig(inject[:,None]/4)
        return self.ffn(residual,p)[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefill-benchmark',action='store_true')
    ap.add_argument('--long-context',action='store_true')
    ap.add_argument('--prefill-tokens',type=int,choices=(512,2048),default=512);args=ap.parse_args()
    lib=C.CDLL(str(ROOT/'build/libqwen_probe.so'));p,i=C.c_void_p,C.c_int
    signatures={'probe_layer_open':([C.c_char_p,i,i],p),'probe_layer':([p,p,p,i],i),
        'checkpoint':([p],i),'restore':([p],i),'reset':([p],i),'close':([p],None),
        'last_error':([],C.c_char_p),'get_stats':([p,p],None)}
    for n,(sig_args,res) in signatures.items():f=getattr(lib,'qwen_'+n);f.argtypes,f.restype=sig_args,res
    def check(code):
        if code:raise RuntimeError(lib.qwen_last_error().decode())
    results=[]
    for layer in (12,15):
        start=time.monotonic();m=lib.qwen_probe_layer_open(str(OUT/f'layer{layer}.index').encode(),layer,args.prefill_tokens)
        if not m:raise RuntimeError(lib.qwen_last_error().decode())
        load_seconds=time.monotonic()-start;w=Weights(OUT/f'layer{layer}');ref=LayerReference(w,layer)
        rng=np.random.default_rng(947+layer);h=(rng.normal(size=(7,4,2560))*0.8).astype(np.float32)
        def forward(inputs):
            out=np.empty_like(inputs);check(lib.qwen_probe_layer(m,out.ctypes.data,inputs.ctypes.data,len(inputs)));return out
        try:
            start=time.monotonic();actual=forward(h);batch_seconds=time.monotonic()-start
            print(f'Layer {layer}: GPU batch done, independent reference running',flush=True)
            expected=np.array([ref.step(x,t) for t,x in enumerate(h)])
            error=float(np.max(np.abs(actual-expected)));rmse=float(np.sqrt(np.mean((actual-expected)**2)))
            np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=3e-3)
            check(lib.qwen_reset(m));sequential=np.concatenate([forward(x[None,:]) for x in h])
            np.testing.assert_array_equal(actual,sequential)
            check(lib.qwen_reset(m));forward(h[:2]);check(lib.qwen_checkpoint(m))
            forward(h[2:]);check(lib.qwen_restore(m));replay=forward(h[2:5])
            np.testing.assert_array_equal(replay,actual[2:5])
            stats=Stats();lib.qwen_get_stats(m,C.byref(stats))
            record=dict(layer=layer,load_seconds=load_seconds,batch_seconds=batch_seconds,max_abs=error,rmse=rmse,
                batch_bit_exact=True,rollback_bit_exact=True,native={n:getattr(stats,n) for n,_ in Stats._fields_})
            if args.prefill_benchmark:
                inputs=(rng.normal(size=(args.prefill_tokens,4,2560))*0.8).astype(np.float32)
                check(lib.qwen_reset(m));forward(inputs[:32]);measurements=[];reference=None
                for batch_size in ((16,64,128,512) if args.prefill_tokens==512 else (128,512,2048)):
                    check(lib.qwen_reset(m));begin=time.monotonic();parts=[]
                    before=Stats();lib.qwen_get_stats(m,C.byref(before))
                    for start in range(0,len(inputs),batch_size):parts.append(forward(inputs[start:start+batch_size]))
                    seconds=time.monotonic()-begin;after=Stats();lib.qwen_get_stats(m,C.byref(after));combined=np.concatenate(parts)
                    if reference is None:reference=combined
                    else:np.testing.assert_array_equal(combined,reference)
                    measurement=dict(batch_size=batch_size,seconds=seconds,tokens_per_second=len(inputs)/seconds,
                        handoffs=after.expert_handoffs-before.expert_handoffs,upload_bytes=after.expert_upload_bytes-before.expert_upload_bytes)
                    measurements.append(measurement);print(json.dumps(dict(layer=layer,**measurement)),flush=True)
                record['prefill_benchmark']=measurements
            if args.long_context:
                inputs=(rng.normal(size=(8192,4,2560))*0.8).astype(np.float32)
                check(lib.qwen_reset(m));parts=[];begin=time.monotonic()
                for start in range(0,8192,args.prefill_tokens):
                    parts.append(forward(inputs[start:start+args.prefill_tokens]))
                    print(f'Layer {layer}: long reference {start+args.prefill_tokens}/8192',flush=True)
                reference=np.concatenate(parts);del parts
                first_seconds=time.monotonic()-begin
                check(lib.qwen_reset(m));begin=time.monotonic()
                for start in range(0,8188,128):
                    end=min(start+128,8188);actual=forward(inputs[start:end])
                    np.testing.assert_array_equal(actual,reference[start:end])
                check(lib.qwen_checkpoint(m));actual=forward(inputs[8188:])
                np.testing.assert_array_equal(actual,reference[8188:])
                check(lib.qwen_restore(m));actual=forward(inputs[8188:8190])
                np.testing.assert_array_equal(actual,reference[8188:8190])
                record['long_context']=dict(context=8192,reference_batch=args.prefill_tokens,reference_batch_seconds=first_seconds,
                    chunk_128_and_replay_seconds=time.monotonic()-begin,chunking_bit_exact=True,
                    rollback_at_8188_bit_exact=True)
                del inputs,reference
            stats=Stats();lib.qwen_get_stats(m,C.byref(stats))
            record['native']={n:getattr(stats,n) for n,_ in Stats._fields_}
            results.append(record);print(json.dumps(record),flush=True)
            (OUT/'results.json').write_text(json.dumps(dict(component_only=True,synthetic_target_features=True,layers=results),indent=2))
        finally:lib.qwen_close(m);w.close()
    print('PASS: real target GDN and QSA layers agree with independent reference, cache handoff and replay',flush=True)


if __name__=='__main__':main()
