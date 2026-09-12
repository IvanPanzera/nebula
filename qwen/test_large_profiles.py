"""Small real-CUDA tests for the largest install profiles; no model loaded."""
import ctypes as C
import unittest
import numpy as np
from test_gpu import Device, LIB, call, P, I, F

LIB.qg_index_workspace.argtypes=[P,P,P,P,P,P,P,I,I,I,F,F,P]
LIB.qg_index_workspace.restype=I


class LargeProfiles(unittest.TestCase):
    def test_tiled_matches_original_attention_bitwise(self):
        rng=np.random.default_rng(4824);cap=24576;pos=16003;nt=2
        k=Device((rng.normal(size=(cap,2,256))*.1).astype(np.float16))
        v=Device((rng.normal(size=(cap,2,256))*.1).astype(np.float16))
        q=Device((rng.normal(size=(nt,24,256))*.1).astype(np.float32))
        first=Device(np.zeros((nt,24,256),np.float32));second=Device(np.zeros_like(first.x))
        call('prepare_context',24576);call('attention',first.p,q.p,k.p,v.p,None,nt,pos,cap)
        call('prepare_context',8192);call('attention',second.p,q.p,k.p,v.p,None,nt,pos,cap)
        np.testing.assert_array_equal(first.get(),second.get())

    def test_tiled_attention_and_alias_at_96k(self):
        rng=np.random.default_rng(1981)
        cap=98304;nt=2;pos=cap-nt
        k=(rng.normal(size=(cap,2,256))*.1).astype(np.float16)
        v=(rng.normal(size=k.shape)*.1).astype(np.float16)
        q=(rng.normal(size=(nt,24,256))*.1).astype(np.float32)
        dk,dv=Device(k),Device(v)
        mask=np.zeros((nt,cap//4),np.int32);mask[:,::47]=1;mask[:,-1]=1
        dm=Device(mask);call('prepare_context',cap)
        for sparse in (False,True):
            dq=Device(q)
            call('attention',dq.p,dq.p,dk.p,dv.p,dm.p if sparse else None,nt,pos,cap)
            actual=dq.get()
            for t in range(nt):
                indices=np.arange(pos+t+1)
                if sparse:indices=indices[mask[t,indices//4]!=0]
                # Independent float64 reference for two different KV heads.
                for h in (0,13):
                    scores=k[indices,h//12].astype(np.float64)@q[t,h].astype(np.float64)/16
                    weights=np.exp(scores-scores.max());weights/=weights.sum()
                    expected=weights@v[indices,h//12].astype(np.float64)
                    np.testing.assert_allclose(actual[t,h],expected,atol=2e-7,rtol=2e-4)

    def test_index_global_scratch_causal_ties_96k(self):
        cap=98304;blocks=cap//4
        # Identical scores: exactly the first 512 complete blocks win every tie.
        keys=Device(np.zeros((blocks,128),np.float32));tail=Device(np.zeros((4,128),np.float32))
        raw=Device(np.zeros((2,128),np.float32));query=Device(np.zeros((2,512),np.float32))
        norm=Device(np.ones(128,np.float32));allowed=Device(np.zeros((2,blocks),np.int32))
        scratch=Device(np.zeros(32768*8,np.uint8));call('prepare_context',cap)
        call('index_workspace',keys.p,tail.p,allowed.p,raw.p,query.p,norm.p,norm.p,2,cap-2,cap,1e7,1e-6,scratch.p)
        expected=np.zeros((2,blocks),np.int32);expected[:,:512]=1;expected[0,-1]=1
        np.testing.assert_array_equal(allowed.get(),expected)
        # Strictly ordered scores on a coordinate unaffected by RoPE. Unlike
        # the tie case, the newest completed keys must win the top-512 sort.
        data=np.zeros((blocks,128),np.float32);data[:,64]=np.arange(blocks,dtype=np.float32)/blocks
        keys=Device(data);query_values=np.zeros((2,512),np.float32);query_values[:,64::128]=1
        query=Device(query_values)
        call('index_workspace',keys.p,tail.p,allowed.p,raw.p,query.p,norm.p,norm.p,2,cap-2,cap,1e7,1e-6,scratch.p)
        expected=np.zeros((2,blocks),np.int32);expected[:,blocks-513:blocks-1]=1;expected[0,-1]=1
        np.testing.assert_array_equal(allowed.get(),expected)

    def test_prefill_8192_matmul_and_routes(self):
        nt=8192;rows=17;cols=32
        rng=np.random.default_rng(27)
        weights=rng.normal(size=(rows,cols)).astype(np.float32)
        x=rng.normal(size=(nt,cols)).astype(np.float32)
        dw,dx=Device(weights),Device(x);out=Device(np.zeros((nt,rows),np.float32))
        call('matmul',out.p,dw.p,0,rows,cols,dx.p,nt)
        np.testing.assert_allclose(out.get(),x@weights.T,atol=2e-5,rtol=3e-5)
        ids=Device(np.tile(np.arange(10,dtype=np.int32),(nt,1)));mapping=Device(np.zeros((nt,512),np.int32))
        call('moe_map',mapping.p,ids.p,nt)
        self.assertEqual(LIB.qg_sync(),0)

if __name__=='__main__': unittest.main()
