"""Independent old/new CPU comparison and CUDA prefix/graph invariants."""
import ctypes as C
import json
import statistics
import time
import unittest
from pathlib import Path
import numpy as np
from test_gpu import Device,LIB,call,SIGNATURES
from decode import DraftPolicy,DecodeStats,generate
from test_decode import Backend

ROOT=Path(__file__).resolve().parent
for name,args in {
    'history_prefix':[C.c_void_p]*3+[C.c_int]*3,
    'index_prefix':[C.c_void_p]*3+[C.c_int]*2,
    'graph_begin':[C.c_void_p],'graph_end':[C.c_void_p],'graph_launch':[C.c_void_p],
}.items():
    f=getattr(LIB,'qg_'+name);f.argtypes=args;f.restype=C.c_int
LIB.qg_graph_destroy.argtypes=[C.c_void_p];LIB.qg_graph_destroy.restype=None

class Optimizations(unittest.TestCase):
    def test_compact_prefix_every_rejection_boundary(self):
        rng=np.random.default_rng(711);nt=17
        raw=rng.normal(0,.1,(nt,10240)).astype('f');alpha=rng.normal(0,.1,(nt,48)).astype('f')
        beta=rng.normal(0,.1,(nt,48)).astype('f');a=np.full(48,-.2,'f');dt=np.zeros(48,'f')
        initial=rng.normal(0,.01,(48,128,128)).astype('f')
        x,al,be,aa,dd=map(Device,(raw,alpha,beta,a,dt));output=Device(np.zeros((nt,6144),'f'))
        full,actual,reference=map(Device,(initial,initial,initial))
        call('gdn',output.p,x.p,al.p,be.p,aa.p,dd.p,full.p,nt,1e-6)
        for keep in range(1,nt):
            call('copy',actual.p,full.p,initial.nbytes)
            call('write',actual.p,initial.ctypes.data,initial.nbytes)
            call('write',reference.p,initial.ctypes.data,initial.nbytes)
            call('gdn',output.p,x.p,al.p,be.p,aa.p,dd.p,actual.p,keep,1e-6)
            # Independent token-by-token state update checks chunk length.
            for t in range(keep):
                call('gdn',output.p,x.p+t*10240*4,al.p+t*48*4,be.p+t*48*4,aa.p,dd.p,reference.p,1,1e-6)
            np.testing.assert_array_equal(actual.get(),reference.get())
        for nh,dil in ((3,1),(9,3)):
            width=32;history=rng.normal(size=(nh,width)).astype('f');inp=rng.normal(size=(nt,width)).astype('f')
            saved,state,data,weights,out=map(Device,(history,history,inp,np.ones((width,4),'f'),np.zeros_like(inp)))
            for keep in range(1,nt):
                call('history_prefix',state.p,saved.p,data.p,width,nh,keep)
                expected=np.concatenate((history,inp[:keep]))[-nh:]
                np.testing.assert_array_equal(state.get(),expected)
        tail=rng.normal(size=(4,128)).astype('f');raw=rng.normal(size=(nt,128)).astype('f')
        saved,state,data=map(Device,(tail,tail,raw))
        for pos in (0,1,2,3,2047,24550):
            for keep in range(1,nt):
                expected=tail.copy()
                for t in range(keep):expected[(pos+t)%4]=raw[t]
                call('index_prefix',state.p,saved.p,data.p,pos,keep)
                np.testing.assert_array_equal(state.get(),expected)

    def test_graph_reuses_operations_with_new_inputs(self):
        x=Device(np.arange(1024,dtype='f')*.001);out=Device(np.zeros(1024,'f'));ref=Device(np.zeros(1024,'f'))
        graph=C.c_void_p()
        try:
            call('graph_begin',C.byref(graph))
            call('unary',out.p,x.p,1024,0,.25)
            call('binary',out.p,out.p,x.p,1024,1)
            call('graph_end',graph)
            for i in range(5):
                values=np.arange(1024,dtype='f')*(i+1)*.001
                call('write',x.p,values.ctypes.data,values.nbytes)
                call('graph_launch',graph)
                call('unary',ref.p,x.p,1024,0,.25)
                call('binary',ref.p,ref.p,x.p,1024,1)
                np.testing.assert_array_equal(out.get(),ref.get())
        finally:LIB.qg_graph_destroy(graph)

    def test_timed_controller_accounts_for_cost_and_censoring(self):
        p=DraftPolicy();p.samples=8;p.costs={'target':{5:.04,17:.04},'catchup':{1:.001,17:.001}}
        p.reached=[100.]*16;p.matched=[99.]*16;p.draft_cost=.001;p.checkpoint_cost=.001;p.restore_cost=.001
        fast=p.timed_choice(4,4,dict(target=.04,catchup=.001,draft=.004,checkpoint=.001,restore=0))
        p=DraftPolicy();p.samples=8;p.costs={'target':{5:.04,17:.04},'catchup':{1:.001,17:.001}}
        p.reached=[100.]*16;p.matched=[99.]*16;p.draft_cost=.1;p.checkpoint_cost=.001;p.restore_cost=.001
        slow=p.timed_choice(4,4,dict(target=.04,catchup=.001,draft=.4,checkpoint=.001,restore=0))
        self.assertGreater(fast,slow)
        p=DraftPolicy();p.observe(16,0,dict(target=.1,catchup=.01,draft=.02,checkpoint=.001,restore=.001))
        self.assertEqual(p.reached[1:], [2.]*15)
        self.assertEqual(p.matched[1:], [1.]*15)

    def test_rejections_never_replay_target_positions(self):
        b=Backend(128,range(200));stats=DecodeStats()
        list(generate(b,[1,2,3],max_new_tokens=61,stats=stats,eos_ids=(0,)))
        self.assertEqual(stats.replay_tokens,0);self.assertGreater(stats.retained_prefix_tokens,0)
        self.assertFalse(any(c[0]=='restore' for c in b.calls))
        self.assertTrue(any(c[0]=='commit_prefix' for c in b.calls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
