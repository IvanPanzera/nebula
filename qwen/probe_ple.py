#!/usr/bin/env python3
"""PLE hashing/EOS and full numerical injection with real quantized weights."""
import ctypes as C
import json
from pathlib import Path
from test_gpu import Device,call,LIB,P,I,np
from probe_mtp import Weights,rms,sig
from quant import dequant

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'work/qwen/ple_probe'


def main():
    info=json.loads((OUT/'rows.json').read_text());tokens=np.array(info['tokens'],np.int32);nt=len(tokens)
    lib=C.CDLL(str(ROOT/'build/libqwen_probe.so'));fn=lib.qwen_probe_ple_rows
    fn.argtypes=[P,P,I,I,P,P,P];fn.restype=None
    arrays=[np.array(info[k],np.uint64) for k in ('multiplier','vocab','offset')]
    rows=np.zeros((nt,16),np.uint64)
    fn(rows.ctypes.data,tokens.ctypes.data,nt,info['eos'],*[a.ctypes.data for a in arrays])
    np.testing.assert_array_equal(rows,info['rows'])
    # Compare vectorized EOS masks on long/random histories, not just fixtures.
    rng=np.random.default_rng(98123);history=rng.integers(0,248320,8192,dtype=np.int32)
    history[::7]=info['eos'];history[1]=history[2]=info['eos'];positions=np.arange(len(history))
    preceding=np.concatenate([[-1],np.maximum.accumulate(np.where(history==info['eos'],positions,-1))[:-1]])
    shifted=[history.astype(np.uint64)]
    for shift in (1,2):shifted.append(np.where(positions-shift>preceding,history[np.maximum(positions-shift,0)],info['eos']).astype(np.uint64))
    expected=[];mixed=shifted[0]*arrays[0][0]
    for order in (2,3):
        mixed=mixed^(shifted[order-1]*arrays[0][order-1]);sl=slice((order-2)*8,(order-1)*8)
        expected.append(mixed[:,None]%arrays[1][sl]+arrays[2][sl])
    actual=np.empty((len(history),16),np.uint64)
    fn(actual.ctypes.data,history.ctypes.data,len(history),info['eos'],*[a.ctypes.data for a in arrays])
    np.testing.assert_array_equal(actual,np.concatenate(expected,axis=1))
    w=Weights(OUT/'weights');p='blk.1.';packed=np.frombuffer((OUT/'rows.bin').read_bytes(),np.uint8)
    h=(rng.normal(size=(nt,4,2560))*.8).astype(np.float32)
    emb=dequant(packed,20,nt*16,160).reshape(nt,2560)
    key=(emb@w.matrix(p+'ple_key.weight').T).reshape(nt,4,2560)
    value=emb@w.matrix(p+'ple_value.weight').T
    key=rms(key,w.matrix(p+'ple_norm_key.weight').reshape(4,2560))
    query=rms(h,w.matrix(p+'ple_norm_query.weight').reshape(4,2560))
    score=(key*query).sum(-1,keepdims=True)/np.sqrt(np.float32(2560))
    gated=sig(np.sqrt(np.maximum(abs(score),1e-6))*np.sign(score))*value[:,None,:]
    normal=rms(gated,w.matrix(p+'ple_norm_conv.weight').reshape(4,2560))
    convolution=np.zeros_like(normal)
    for t in range(nt):
        for k in range(4):
            source=t-(3-k)*3
            if source>=0:convolution[t]+=normal[source]*w.matrix(p+'ple_conv1d.weight')[:,k].reshape(4,2560)
    expected=h+gated+convolution*sig(convolution)
    device_weights={}
    for name,(t,mapped) in w.tensors.items():
        device_weights[name]=Device(np.frombuffer(mapped,dtype=np.uint8,count=t['bytes'],offset=t['file_offset']))
    def weights(suffix):return device_weights[p+suffix].p
    LIB.qg_dequant.argtypes=[P,P,I,I,I];LIB.qg_dequant.restype=I
    dpacked=Device(packed);de=Device(np.empty((nt,2560),np.float32));dh=Device(h)
    dk,dq,dg,dn,dc=[Device(np.empty_like(h)) for _ in range(5)];dv=Device(np.empty((nt,2560),np.float32))
    state=Device(np.zeros((9,10240),np.float32))
    call('dequant',de.p,dpacked.p,20,nt*16,160)
    call('matmul',dk.p,weights('ple_key.weight'),8,10240,2560,de.p,nt)
    call('matmul',dv.p,weights('ple_value.weight'),8,2560,2560,de.p,nt)
    call('norm',dk.p,dk.p,weights('ple_norm_key.weight'),2560,4,nt,1,0,1e-6)
    call('norm',dq.p,dh.p,weights('ple_norm_query.weight'),2560,4,nt,1,0,1e-6)
    call('ple_gate',dg.p,dk.p,dq.p,dv.p,nt)
    call('norm',dn.p,dg.p,weights('ple_norm_conv.weight'),2560,4,nt,1,0,1e-6)
    call('conv',dc.p,dn.p,weights('ple_conv1d.weight'),state.p,10240,4,3,nt,1)
    call('binary',dh.p,dh.p,dg.p,nt*10240,0);call('binary',dh.p,dh.p,dc.p,nt*10240,0)
    actual=dh.get();np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
    # Zero dot product must have exactly gate=0.5, despite the magnitude clamp.
    call('zero',dk.p,dk.x.nbytes);call('ple_gate',dg.p,dk.p,dq.p,dv.p,nt)
    np.testing.assert_array_equal(dg.get(),np.broadcast_to(dv.get()[:,None,:]*.5,h.shape))
    result=dict(real_weights=True,full_target=False,hash_positions_checked=8192+nt,
        eos_boundaries_exact=True,max_abs=float(np.max(abs(actual-expected))),rmse=float(np.sqrt(np.mean((actual-expected)**2))),zero_gate_exact=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    del device_weights;w.close()


if __name__=='__main__':main()
