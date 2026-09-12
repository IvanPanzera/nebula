#!/usr/bin/env python3
"""Independent NumPy equations, real CUDA execution, chunk/rollback invariants."""
import ctypes as C
import json
from pathlib import Path
import time
import unittest
import numpy as np
from quant import dequant

ROOT = Path(__file__).resolve().parent
LIB = C.CDLL(str(ROOT / "build/libqwen_gpu.so"))
P, I, F, Z = C.c_void_p, C.c_int, C.c_float, C.c_size_t
SIGNATURES = {
    "alloc": ([Z], P), "free": ([P], None), "write": ([P,P,Z], I),
    "read": ([P,P,Z], I), "copy": ([P,P,Z], I), "zero": ([P,Z], I),
    "sync": ([], I), "error": ([I], C.c_char_p),
    "prepare_context": ([I], I),
    "matmul": ([P,P,I,I,I,P,I], I), "gather": ([P,P,I,I,I], I),
    "norm": ([P,P,P,I,I,I,I,I,F], I), "unary": ([P,P,I,I,F], I),
    "binary": ([P,P,P,I,I], I), "hc_init": ([P,P,I,I], I),
    "hc_read": ([P,P,P,I,I], I), "hc_write": ([P,P,P,I,I], I),
    "mtp_concat": ([P,P,P,I,I], I), "conv": ([P,P,P,P,I,I,I,I,I], I),
    "gdn": ([P,P,P,P,P,P,P,I,F], I), "gate": ([P,P,I], I),
    "qsplit": ([P,P,P,I], I), "rope": ([P,I,I,I,I,F], I),
    "kv_store": ([P,P,I,I,I], I), "attention": ([P,P,P,P,P,I,I,I], I),
    "index": ([P,P,P,P,P,P,P,I,I,I,F,F], I),
    "route": ([P,P,P,I], I), "ple_gate": ([P,P,P,P,I], I),
    "prune_routes": ([P,P,P,I,C.c_double,P], I),
    "argmax": ([P,P,I,I], I),
    "moe_map": ([P,P,I], I), "moe_gather": ([P,P,P,I], I), "moe_scatter": ([P,P,P,P,P,I,I], I), "moe_reduce": ([P,P,I], I),
    "mtp_moe": ([P,P,P,P,P,P,I,P,I,P,I,I], I),
}
for name, (args, result) in SIGNATURES.items():
    fn = getattr(LIB, "qg_" + name)
    fn.argtypes, fn.restype = args, result


def call(name, *args):
    code = getattr(LIB, "qg_" + name)(*args)
    if code:
        raise RuntimeError(f"{name}: {LIB.qg_error(code).decode()}")


class Device:
    def __init__(self, x):
        self.x = np.ascontiguousarray(x)
        self.p = LIB.qg_alloc(self.x.nbytes)
        if not self.p:
            raise MemoryError("CUDA allocation")
        call("write", self.p, self.x.ctypes.data, self.x.nbytes)

    def get(self):
        out = np.empty_like(self.x)
        call("read", out.ctypes.data, self.p, out.nbytes)
        return out

    def __del__(self):
        if getattr(self, "p", None):
            LIB.qg_free(self.p)
            self.p = None


def sig(x):
    return 1 / (1 + np.exp(-x))


def rms(x, w=1, eps=1e-6):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions, theta=1e7):
    y = x.copy()
    angle = np.asarray(positions)[..., None] * theta**(-np.arange(32, dtype=np.float64)/32)
    while angle.ndim < x.ndim:
        angle = np.expand_dims(angle, -2)
    cs, sn = np.cos(angle), np.sin(angle)
    y[..., :32] = x[..., :32]*cs - x[..., 32:64]*sn
    y[..., 32:64] = x[..., :32]*sn + x[..., 32:64]*cs
    return y


class KernelTests(unittest.TestCase):
    def test_current_mass_pruning_and_strict_threshold(self):
        class Counters(C.Structure):
            _fields_=[(k,C.c_uint64) for k in ('token_layers','missing','skipped','avoided')]+[('mass',C.c_double)]
        ids=np.tile(np.arange(10,dtype=np.int32),(5,1));weights=np.full((5,10),.1,np.float32)
        resident=np.zeros(512,np.uint8);resident[:8]=1
        weights[0,8:]=.04;weights[1,8:]=.05;weights[2,8:]=.06;weights[3,8:]=0;weights[4,8:]=[.01,.10]
        di,dw,dr,dc=Device(ids),Device(weights),Device(resident),Device(np.zeros(C.sizeof(Counters),np.uint8))
        call('prune_routes',di.p,dw.p,dr.p,5,.10,dc.p)
        actual=di.get();self.assertTrue(np.all(actual[[0,3],8:]<=-2))
        np.testing.assert_array_equal(actual[[1,2,4]],ids[[1,2,4]])
        np.testing.assert_array_equal(dw.get()[:,:8],weights[:,:8])
        counts=Counters.from_buffer_copy(dc.get());self.assertEqual((counts.token_layers,counts.missing,counts.skipped,counts.avoided),(5,10,4,2))

    @classmethod
    def setUpClass(cls):
        call('prepare_context',24576)

    def setUp(self):
        self.rng = np.random.default_rng(812603)

    def rand(self, shape, scale=0.2):
        return (self.rng.normal(size=shape)*scale).astype(np.float32)

    def close(self, a, b, atol=2e-5, rtol=2e-5):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    def test_dense_formats_and_batch_tail(self):
        # Random legal packed blocks exercise every bit, independently expanded
        # here using array operations, including Q5's non-contiguous high bits.
        for typ in (0, 1, 30, 2, 6, 7, 8, 12, 13, 14, 20):
            for cols in ((320,640) if typ not in (12,13,14) else (256,768)):
                rows = 13
                if typ in (0,1,30):
                    w = self.rand((rows,cols))
                    if typ==0: packed=w.view(np.uint8)
                    elif typ==1: packed=w.astype(np.float16).view(np.uint8);w=packed.view(np.float16).astype(np.float32)
                    else: packed=(w.view(np.uint32)>>16).astype(np.uint16).view(np.uint8);w=(packed.view(np.uint16).astype(np.uint32)<<16).view(np.float32)
                else:
                    block = 256 if typ in (12,13,14) else 32
                    size = {2:18,6:22,7:24,8:34,12:144,13:176,14:210,20:18}[typ]
                    p = self.rng.integers(0,256,(rows,cols//block,size),dtype=np.uint8)
                    d = np.full((rows,cols//block),0.002,dtype=np.float16)
                    off = 208 if typ==14 else 0
                    p[...,off:off+2]=d[...,None].view(np.uint8)
                    if typ in (7,12,13):p[...,2:4]=np.full_like(d,0.001)[...,None].view(np.uint8)
                    values=[]
                    for i in range(block):
                        if typ==8:v=p[...,2+i].view(np.int8).astype(np.float64)*d
                        elif typ in (2,6,7,20):
                            q=((p[...,(8 if typ==7 else 6 if typ==6 else 2)+i%16]>>((i//16)*4))&15).astype(np.int32)
                            if typ in (6,7):
                                high=np.ascontiguousarray(p[...,(4 if typ==7 else 2):(8 if typ==7 else 6)]).view('<u4').squeeze(-1)
                                q|=((high>>i)&1).astype(np.int32)*16
                            if typ==20:v=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113])[q]*d
                            else:v=(q-(8 if typ==2 else 16 if typ==6 else 0))*d+(np.float16(0.001) if typ==7 else 0)
                        elif typ==14:
                            h,s,l=i//128,(i%128)//32,i%32
                            low=(p[...,h*64+s%2*32+l]>>((s//2)*4))&15
                            high=(p[...,128+h*32+l]>>(s*2))&3
                            scale=p[...,192+h*8+s*2+l//16].view(np.int8).astype(np.int32)
                            v=((low.astype(np.int32)|(high.astype(np.int32)<<4))-32)*scale*d
                        else:
                            g,l=i//32,i%32;sc=p[...,4:16]
                            scale=(sc[...,g]&63) if g<4 else ((sc[...,g+4]&15)|((sc[...,g-4]>>6)<<4))
                            minimum=(sc[...,g+4]&63) if g<4 else ((sc[...,g+4]>>4)|((sc[...,g]>>6)<<4))
                            q=((p[...,(16 if typ==12 else 48)+(g//2)*32+l]>>((g%2)*4))&15).astype(np.int32)
                            if typ==13:q|=((p[...,16+l]>>g)&1).astype(np.int32)*16
                            v=q*scale.astype(np.float64)*d-minimum.astype(np.float64)*float(np.float16(0.001))
                        values.append(v)
                    w=np.stack(values,axis=-1).reshape(rows,cols).astype(np.float32)
                    packed=p.reshape(rows,-1)
                dw=Device(packed)
                self.close(dequant(packed,typ,rows,cols),w)
                for nt in (1,4,16,17,33,512,2048,1):
                    x=self.rand((nt,cols));dx=Device(x);out=Device(np.zeros((nt,rows),np.float32))
                    call('matmul',out.p,dw.p,typ,rows,cols,dx.p,nt)
                    self.close(out.get(),x.astype(np.float64)@w.astype(np.float64).T,atol=8e-5,rtol=4e-5)
                row=Device(np.zeros(cols,np.float32));call('gather',row.p,dw.p,typ,cols,rows-1)
                self.close(row.get(),w[-1])

    def test_mtp_moe_seventeen_rows_against_numpy(self):
        # Two small synthetic experts use the production Q4_K / IQ4_NL formats.
        # Repeated IDs exercise all ten slots; no large model is opened.
        weights=[];device=[]
        for typ,rows,cols in ((12,640,2560),(12,640,2560),(20,2560,640)):
            block,size=(256,144) if typ==12 else (32,18)
            packed=self.rng.integers(0,256,(2*rows,cols//block,size),dtype=np.uint8)
            packed[...,:2]=np.full((*packed.shape[:2],1),.0002,dtype=np.float16).view(np.uint8)
            if typ==12:packed[...,2:4]=np.full((*packed.shape[:2],1),.0001,dtype=np.float16).view(np.uint8)
            weights.append(dequant(packed,typ,2*rows,cols).reshape(2,rows,cols))
            device.append(Device(packed))
        nt=17;x=self.rand((nt,2560));ids=self.rng.integers(0,2,(nt,10),dtype=np.int32)
        scales=self.rng.uniform(.01,.2,(nt,10)).astype(np.float32)
        dx,di,dw=Device(x),Device(ids),Device(scales)
        mid=Device(np.zeros((nt,10,640),np.float32));slots=Device(np.zeros((nt,10,2560),np.float32))
        call('mtp_moe',slots.p,mid.p,dx.p,di.p,dw.p,device[0].p,12,device[1].p,12,device[2].p,20,nt)
        expected=np.empty((nt,10,2560),np.float32)
        for e in range(2):
            g=x.astype(np.float64)@weights[0][e].astype(np.float64).T
            u=x.astype(np.float64)@weights[1][e].astype(np.float64).T
            y=((g*sig(g))*u)@weights[2][e].astype(np.float64).T
            for t,k in zip(*np.nonzero(ids==e)):expected[t,k]=y[t]*scales[t,k]
        self.close(slots.get(),expected,atol=2e-5,rtol=5e-5)
        self.assertNotEqual(LIB.qg_mtp_moe(slots.p,mid.p,dx.p,di.p,dw.p,device[0].p,12,device[1].p,12,device[2].p,20,18),0)

    def test_hyper_connections_and_mtp_streams(self):
        nt,d=3,320
        h=self.rand((nt,4,d));w=self.rand((4,d),0.5)+1;gate=self.rand(h.shape)
        dh,dw,dg=Device(h),Device(w),Device(gate);norm=Device(np.zeros_like(h));out=Device(np.zeros((nt,d),np.float32))
        call('norm',norm.p,dh.p,dw.p,d,4,nt,1,0,1e-6)
        self.close(norm.get(),rms(h,w))
        call('hc_read',out.p,norm.p,dg.p,d,nt)
        mixed=(rms(h,w)*sig(gate)).mean(axis=1);self.close(out.get(),mixed)
        inject=self.rand((nt,4));di=Device(inject)
        call('hc_write',dh.p,out.p,di.p,d,nt)
        self.close(dh.get(),h+mixed[:,None,:]*2*sig(inject[:,:,None]/4))
        emb=self.rand((nt,d));de=Device(emb);concat=Device(np.zeros((nt,4,2*d),np.float32))
        call('mtp_concat',concat.p,de.p,norm.p,d,nt)
        self.close(concat.get(),np.concatenate([np.broadcast_to(emb[:,None,:],h.shape),rms(h,w)],axis=-1))

    def test_convolution_chunking_and_rollback(self):
        for dilation in (1,3):
            nt,c,k=7,69,4;nh=(k-1)*dilation
            x=self.rand((nt,c));w=self.rand((c,k));hist=self.rand((nh,c));dh=Device(hist)
            dx,dw=Device(x),Device(w);out=Device(np.zeros_like(x))
            call('conv',out.p,dx.p,dw.p,dh.p,c,k,dilation,nt,1)
            full=np.concatenate([hist,x]);expected=np.stack([sum(w[:,j]*full[nh+t-(k-1-j)*dilation] for j in range(k)) for t in range(nt)])
            self.close(out.get(),expected*sig(expected));self.close(dh.get(),full[-nh:])
            replay=Device(hist);parts=[]
            for chunk in (x[:2],x[2:5],x[5:]):
                dc=Device(chunk);do=Device(np.zeros_like(chunk));call('conv',do.p,dc.p,dw.p,replay.p,c,k,dilation,len(chunk),1);parts.append(do.get())
            self.close(np.concatenate(parts),out.get());self.close(replay.get(),dh.get())
            # Restore pre-block history and keep only the first three tokens.
            call('write',dh.p,hist.ctypes.data,hist.nbytes)
            call('conv',out.p,dx.p,dw.p,dh.p,c,k,dilation,3,1)
            self.close(dh.get(),np.concatenate([hist,x[:3]])[-nh:])

    def test_gdn_against_recurrence_and_chunking(self):
        nt=4;qkv=self.rand((nt,10240));qkv[0,:4096]*=1e-5
        alpha,beta=self.rand((nt,48)),self.rand((nt,48));a=-np.exp(self.rand((48,)));dt=self.rand((48,))
        initial=self.rand((48,128,128),0.01);state=initial.astype(np.float64).transpose(0,2,1).copy();expected=[]
        for t in range(nt):
            q=qkv[t,:2048].reshape(16,128).astype(np.float64);k=qkv[t,2048:4096].reshape(16,128).astype(np.float64)
            q=q/np.sqrt((q*q).sum(-1,keepdims=True)+1e-6);k=k/np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
            q=q[np.arange(48)%16];k=k[np.arange(48)%16];v=qkv[t,4096:].reshape(48,128)
            decay=np.exp(a.astype(np.float64)*np.logaddexp(0,alpha[t]+dt))
            state*=decay[:,None,None];mem=np.einsum('hkv,hk->hv',state,k)
            delta=(v-mem)*sig(beta[t])[:,None];state+=k[:,:,None]*delta[:,None,:]
            expected.append(np.einsum('hkv,hk->hv',state,q)/np.sqrt(128))
        buffers=[Device(x) for x in (qkv,alpha,beta,a,dt,initial)]
        output=Device(np.zeros((nt,6144),np.float32))
        call('gdn',output.p,*[x.p for x in buffers],nt,1e-6)
        self.close(output.get(),np.array(expected).reshape(nt,6144),atol=2e-6)
        self.close(buffers[-1].get(),state.transpose(0,2,1),atol=2e-6)
        ds=Device(initial);parts=[]
        for t in range(nt):
            inputs=[Device(x[t:t+1]) for x in (qkv,alpha,beta)];o=Device(np.zeros((1,6144),np.float32))
            call('gdn',o.p,*[x.p for x in inputs],buffers[3].p,buffers[4].p,ds.p,1,1e-6);parts.append(o.get())
        self.close(np.concatenate(parts),output.get(),atol=2e-6);self.close(ds.get(),buffers[-1].get(),atol=2e-6)

    def test_flash_next_sigmoid_output_gate(self):
        # The exported model sets output_gate_type="sigmoid"; using the GDN
        # class's default SiLU would flip the sign of negative-gate outputs.
        x=self.rand((2560,));g=np.linspace(-30,30,2560,dtype=np.float32);dg=Device(g)
        dx=Device(x);call('gate',dx.p,dg.p,x.size);self.close(dx.get(),x*sig(g))

    def test_rope_attention_and_future_mask(self):
        nt,pos,cap=3,79,128
        q=self.rand((nt,24,256));dq=Device(q)
        call('rope',dq.p,256,24,nt,pos,1e7)
        rotated=rope(q,np.arange(pos,pos+nt));self.close(dq.get(),rotated)
        k=self.rand((cap,2,256)).astype(np.float16);v=self.rand((cap,2,256)).astype(np.float16)
        dk,dv=Device(k),Device(v);out=Device(np.zeros_like(q))
        mask=np.ones((nt,cap//4),np.int32);mask[:,2:6]=0;dm=Device(mask)
        call('attention',out.p,dq.p,dk.p,dv.p,dm.p,nt,pos,cap)
        expected=[]
        for t in range(nt):
            heads=[]
            for h in range(24):
                keep=np.flatnonzero(mask[t,np.arange(pos+t+1)//4]);kh=h//12
                score=k[keep,kh].astype(np.float64)@rotated[t,h]/16
                prob=np.exp(score-score.max());prob/=prob.sum()
                heads.append(prob@v[keep,kh].astype(np.float64))
            expected.append(heads)
        self.close(out.get(),np.array(expected))
        v[pos+nt:]=1000;call('write',dv.p,v.ctypes.data,v.nbytes)
        call('attention',out.p,dq.p,dk.p,dv.p,dm.p,nt,pos,cap);self.close(out.get(),expected)

    def test_qsa_complete_blocks_tail_and_chunking(self):
        for pos,cap in ((2078,8192),(8187,8192),(16381,24576),(24571,24576)):
            self.check_qsa_selection(pos,cap)

    def check_qsa_selection(self,pos,cap):
        nt=5;blocks=cap//4
        raw=self.rand((pos+nt,128));queries=self.rand((nt,4,128));kn=self.rand((128,),0.2)+1;qn=self.rand((128,),0.2)+1
        queries[-1]=0  # tied block scores choose the oldest blocks deterministically
        initial=np.zeros((blocks,128),np.float32);ncomplete=pos//4
        initial[:ncomplete]=rope(rms(raw[:ncomplete*4].reshape(-1,4,128).mean(1),kn),np.arange(ncomplete)*4)
        tail=np.zeros((4,128),np.float32);tail[:pos%4]=raw[ncomplete*4:pos]
        keys,dt,dr,dq,dkn,dqn=[Device(x) for x in (initial,tail,raw[pos:],queries,kn,qn)]
        allowed=Device(np.zeros((nt,blocks),np.int32))
        call('index',keys.p,dt.p,allowed.p,dr.p,dq.p,dkn.p,dqn.p,nt,pos,cap,1e7,1e-6)
        masks=[]
        for t in range(nt):
            length=pos+t+1;nb=length//4
            pooled=rope(rms(raw[:nb*4].reshape(-1,4,128).mean(1),kn),np.arange(nb)*4)
            rq=rope(rms(queries[t],qn),length-1)
            scores=np.maximum(rq.astype(np.float64)@pooled.astype(np.float64).T,0).sum(0)
            selected=np.lexsort((np.arange(nb),-scores))[:512]
            mask=np.zeros(blocks,np.int32);mask[selected]=1
            if length%4:mask[nb]=1
            masks.append(mask)
        np.testing.assert_array_equal(allowed.get(),masks)
        replay_keys,replay_tail=Device(initial),Device(tail);replay_masks=[]
        for t in range(nt):
            r,q=Device(raw[pos+t:pos+t+1]),Device(queries[t:t+1]);out=Device(np.zeros((1,blocks),np.int32))
            call('index',replay_keys.p,replay_tail.p,out.p,r.p,q.p,dkn.p,dqn.p,1,pos+t,cap,1e7,1e-6);replay_masks.append(out.get())
        np.testing.assert_array_equal(np.concatenate(replay_masks),allowed.get());self.close(replay_keys.get(),keys.get())

    def test_attention_24k_dense_sparse_and_in_place(self):
        cap,nt=24576,5
        k=self.rand((cap,2,256)).astype(np.float16);v=self.rand((cap,2,256)).astype(np.float16)
        dk,dv=Device(k),Device(v)
        for pos in (12285,16381,24571):
            q=self.rand((nt,24,256));mask=np.zeros((nt,cap//4),np.int32)
            mask[:,::3]=1;mask[:,pos//4:(pos+nt+3)//4]=1;dm=Device(mask)
            for sparse in (False,True):
                dq=Device(q)
                call('attention',dq.p,dq.p,dk.p,dv.p,dm.p if sparse else None,nt,pos,cap)
                expected=[]
                for t in range(nt):
                    keep=np.arange(pos+t+1)
                    if sparse:keep=keep[mask[t,keep//4]!=0]
                    heads=[]
                    for h in range(24):
                        score=k[keep,h//12].astype(np.float64)@q[t,h].astype(np.float64)/16
                        prob=np.exp(score-score.max());prob/=prob.sum()
                        heads.append(prob@v[keep,h//12].astype(np.float64))
                    expected.append(heads)
                self.close(dq.get(),expected)

    def test_attention_24k_prefill_2048_and_capacity_guard(self):
        cap,nt=24576,2048;pos=cap-nt
        # Zero keys give uniform scores. Every output must be the causal mean,
        # checking all 2048 rows without a context-squared reference matrix.
        k=Device(np.zeros((cap,512),np.float16))
        value=(np.arange(cap,dtype=np.float32)%71/71).astype(np.float16)
        v=Device(np.repeat(value[:,None],512,axis=1));q=Device(np.zeros((nt,24,256),np.float32))
        call('attention',q.p,q.p,k.p,v.p,None,nt,pos,cap)
        expected=np.cumsum(value.astype(np.float64))[pos:]/np.arange(pos+1,cap+1)
        self.close(q.get(),np.broadcast_to(expected[:,None,None],q.x.shape),atol=3e-5)
        code=LIB.qg_attention(q.p,q.p,k.p,v.p,None,nt,pos+1,cap)
        self.assertNotEqual(code,0)

    def test_routing_and_argmax(self):
        logits=self.rand((3,512));logits[0]=0
        dl=Device(logits);ids=Device(np.zeros((3,10),np.int32));weights=Device(np.zeros((3,10),np.float32))
        call('route',ids.p,weights.p,dl.p,3)
        selected=np.argsort(-logits,axis=1,kind='stable')[:,:10]
        np.testing.assert_array_equal(ids.get(),selected)
        vals=np.take_along_axis(logits,selected,axis=1);p=np.exp(vals-vals.max(1,keepdims=True));p/=p.sum(1,keepdims=True);self.close(weights.get(),p)
        logits[1,7]=np.nan;call('write',dl.p,logits.ctypes.data,logits.nbytes);call('route',ids.p,weights.p,dl.p,3)
        np.testing.assert_array_equal(ids.get()[1],-1)
        vocab=248320;head=self.rand((2,vocab));head[0,11]=head[0,77]=10;dhead=Device(head);result=Device(np.zeros(2,np.int32))
        call('argmax',result.p,dhead.p,vocab,2);np.testing.assert_array_equal(result.get(),np.argmax(head,axis=1))
        head[1,99]=np.nan;call('write',dhead.p,head.ctypes.data,head.nbytes)
        call('argmax',result.p,dhead.p,vocab,2);np.testing.assert_array_equal(result.get(),[11,-1])

    def test_moe_large_prefill_map(self):
        for nt in (511,513,1025,2048):
            ids=np.stack([self.rng.choice(512,10,replace=False) for _ in range(nt)]).astype(np.int32)
            dids=Device(ids);mapping=Device(np.zeros((512,nt),np.int32))
            call('moe_map',mapping.p,dids.p,nt);actual=mapping.get()
            for expert in range(512):
                expected=np.flatnonzero((ids==expert).any(1))
                np.testing.assert_array_equal(actual[expert,:len(expected)],expected)

    def test_moe_replay_independent_of_cache_order(self):
        # A verification batch can need more experts than the resident cache.
        # Warm/cold scheduling must not change summation order on rollback.
        nt=4
        ids=np.stack([self.rng.choice(27,10,replace=False) for _ in range(nt)]).astype(np.int32)
        weights=np.exp(self.rand((nt,10)));weights/=weights.sum(1,keepdims=True)
        values=self.rand((27,nt,2560),30)
        dids,dw=Device(ids),Device(weights)
        mapping=Device(np.zeros((512,nt),np.int32));call('moe_map',mapping.p,dids.p,nt)
        slots=Device(np.zeros((nt,10,2560),np.float32));out=Device(np.zeros((nt,2560),np.float32))
        expected=np.zeros((nt,2560),np.float32)
        for k in range(10):expected+=values[ids[:,k],np.arange(nt)]*weights[:,k,None]
        results=[]
        for order in (np.unique(ids),self.rng.permutation(np.unique(ids))):
            call('zero',slots.p,slots.x.nbytes)
            for expert in order:
                selected=(ids==expert).any(1);dx=Device(values[expert]);compact=Device(np.zeros_like(values[expert]))
                token_map=mapping.p+int(expert)*nt*4
                call('moe_gather',compact.p,dx.p,token_map,int(selected.sum()))
                np.testing.assert_array_equal(compact.get()[:selected.sum()],values[expert,selected])
                call('moe_scatter',slots.p,compact.p,dw.p,dids.p,token_map,int(expert),int(selected.sum()))
            call('moe_reduce',out.p,slots.p,nt);results.append(out.get())
        np.testing.assert_array_equal(results[0],results[1])
        self.close(results[0],expected)


if __name__=='__main__':
    start=time.time();suite=unittest.defaultTestLoader.loadTestsFromTestCase(KernelTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=ROOT.parent/'work/qwen';out.mkdir(parents=True,exist_ok=True)
    (out/'kernel_tests.json').write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.time()-start),indent=2))
    raise SystemExit(not result.wasSuccessful())
