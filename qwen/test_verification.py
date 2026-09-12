"""Verification boundaries, causal prefix state, API validation and CUDA scores."""
import ctypes as C
import os
import unittest
import numpy as np
from verification import settings, accepted_prefix
from decode import generate, DecodeStats
from test_decode import Backend
from web_server import validate_request

CPU_ONLY = os.environ.get('NEBULA_TEST_CPU_ONLY') == '1'
if not CPU_ONLY:
    from test_gpu import LIB, Device, call
    LIB.qg_verify.argtypes=[C.c_void_p]*4+[C.c_int]*3+[C.c_float]
    LIB.qg_verify.restype=C.c_int


class RelaxedBackend(Backend):
    def verify(self,proposals,top_k,min_ratio):
        return self.top1(),[True]*len(proposals)


class VerificationTests(unittest.TestCase):
    def test_caps_and_first_rejection(self):
        proposals=[10,11,12,13,14,15];best=[1,11,2,13,3,4,99]
        for level,expected in ((3,(0,0)),(2,(2,1)),(1,(6,4))):
            self.assertEqual(accepted_prefix(proposals,best,[True]*6,level),expected)
        self.assertEqual(accepted_prefix(proposals,best,[True,True,False,True,True,True],2),(2,1))
        self.assertEqual(accepted_prefix(proposals,best,[False]*6,1),(0,0))

    def test_relaxed_state_and_eos_boundaries(self):
        for level in range(1,4):
            for context,limit in ((8,50),(128,61),(128,1)):
                b=RelaxedBackend(context,range(200));s=DecodeStats();prompt=[1,2,3]
                output=list(generate(b,prompt,max_new_tokens=limit,draft_max=4,adaptive=False,
                                     eos_ids=(0,),stats=s,verification_level=level))
                self.assertEqual(b.history,(prompt+output)[:b.position])
                self.assertEqual(len(output),min(limit,context-len(prompt)))
                self.assertEqual(s.target_output_tokens+s.draft_output_tokens,len(output))
                self.assertEqual(s.relaxed_output_tokens,len(s.relaxed_trace))
                self.assertEqual(s.verification_scale,3)
                if level==3:self.assertEqual(s.relaxed_output_tokens,0)
            for end in (3,4,6,8,14):
                b=RelaxedBackend(128,eos_position=end)
                output=list(generate(b,[1,2,3],max_new_tokens=50,eos_ids=(0,),verification_level=level))
                self.assertEqual(len(output),end-3);self.assertNotIn(0,output)

    def test_request_validation(self):
        request=dict(messages=[dict(role='user',content='Test')])
        self.assertEqual(validate_request(request)['verification_level'],3)
        for level in range(1,4):self.assertEqual(validate_request(dict(request,verification_scale=3,verification_level=level))['verification_level'],level)
        for level in (True,False,0,4,5,6,'2',2.0,None):
            with self.assertRaises(ValueError):validate_request(dict(request,verification_scale=3,verification_level=level))
            with self.assertRaises(ValueError):list(generate(Backend(128),[1],verification_level=level))
        # A stale tab using old 1/2/3 must not silently change policy.
        for scale in (None,5,'3',3.0,True):
            with self.assertRaises(ValueError):validate_request(dict(request,verification_scale=scale,verification_level=1))
        with self.assertRaises(ValueError):validate_request(dict(request,verification_level=1))

    def cuda_check(self,x,ids,top_k,ratio):
        n=x.shape[1];count=len(ids);gap=np.float32(-np.log(np.float32(ratio)))
        data,proposals,best,eligible,greedy=map(Device,(x,np.array(ids,'i'),np.zeros(count+1,'i'),np.zeros(count,'i'),np.zeros(count+1,'i')))
        call('verify',best.p,eligible.p,data.p,proposals.p,n,count,top_k,float(gap))
        call('argmax',greedy.p,data.p,n,count+1)
        actual=best.get();np.testing.assert_array_equal(actual,greedy.get())
        expected=[]
        for t,d in enumerate(ids):
            row=x[t];score=row[d];order=np.count_nonzero((row>score)|((row==score)&(np.arange(n)<d)))
            expected.append(actual[t]>=0 and np.isfinite(score) and order<top_k and np.float32(row[actual[t]]-score)<=gap)
        np.testing.assert_array_equal(eligible.get(),expected)

    @unittest.skipIf(CPU_ONLY, 'CUDA checks run on an NVIDIA host')
    def test_cuda_full_vocabulary_and_tie_order(self):
        rng=np.random.default_rng(936)
        for count in (1,4,16):
            x=rng.normal(0,2,(count+1,248320)).astype('f')
            ids=[int(np.argsort(-row,kind='stable')[t%12]) for t,row in enumerate(x[:count])]
            for level in (1,2):
                v=settings(level);self.cuda_check(x,ids,v['top_k'],v['min_ratio'])
        x=np.ones((5,512),'f')
        for k in (1,3,5,10):self.cuda_check(x,[0,1,3,10],k,.8)

    @unittest.skipIf(CPU_ONLY, 'CUDA checks run on an NVIDIA host')
    def test_cuda_ratio_boundary_and_nonfinite(self):
        for ratio,k in ((.8,3),(.2,10)):
            gap=np.float32(-np.log(np.float32(ratio)));x=np.full((5,1024),-100,'f');x[:,0]=0
            x[0,1]=-gap+1e-4;x[1,1]=-gap-1e-4;x[2,1]=-np.inf;x[3,1]=np.nan
            self.cuda_check(x,[1]*4,k,ratio)
        x=np.zeros((4,512),'f');x[0,4]=np.inf;x[1,:]=-np.inf
        self.cuda_check(x,[0,0,0],3,.8)


if __name__=='__main__':unittest.main(verbosity=2)
