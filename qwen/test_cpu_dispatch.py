"""CPU portability checks, independent of model files and CUDA allocations."""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
P = C.c_void_p

def bind(lib, prefix):
    fn = getattr(lib, prefix + 'matmul')
    fn.argtypes = [P, P, C.c_int, C.c_int, C.c_int, P, C.c_int, C.c_int]
    fn.restype = C.c_int
    return fn

def encoded(rng, typ, rows, cols):
    blocks, size = {0:(1,4),1:(1,2),30:(1,2),2:(32,18),20:(32,18),
                   6:(32,22),7:(32,24),8:(32,34),12:(256,144),13:(256,176),14:(256,210)}[typ]
    if typ == 0:
        return rng.normal(0, .05, (rows, cols)).astype('f4').view('u1')
    if typ == 1:
        return rng.normal(0, .05, (rows, cols)).astype('f2').view('u1')
    if typ == 30:
        return (rng.normal(0, .05, (rows, cols)).astype('f4').view('u4') >> 16).astype('u2').view('u1')
    data = rng.integers(0, 256, (rows * cols // blocks, size), dtype='u1')
    scale = np.array([.0005], dtype='f2').view('u1')
    data[:, 208 if typ == 14 else 0:210 if typ == 14 else 2] = scale
    if typ in (7, 12, 13):
        data[:, 2:4] = np.array([.0002], dtype='f2').view('u1')
    return data

def main():
    lib = C.CDLL(str(HERE / 'build/libqwen_cpu.so'))
    lib.qc_backend_name.restype = C.c_char_p
    lib.qc_backend_available.argtypes = [C.c_char_p]
    backends = [('scalar', 'qcs_'), ('avx2', 'qc2_'), ('avx512', 'qc5_')]
    variants = [(name, prefix, lib) for name, prefix in backends if lib.qc_backend_available(name.encode())]
    old = os.environ.get('QWEN_CPU_REFERENCE')
    if old:
        variants.insert(0, ('previous', 'qc_', C.CDLL(old)))
    rng = np.random.default_rng(91011)
    matrices = 0
    for typ in (0, 1, 30, 2, 20, 6, 7, 8, 12, 13, 14):
        weights = encoded(rng, typ, 7, 512)
        for nt in (1, 3, 4, 5, 17, 18):
            x = rng.normal(0, .2, (nt, 512)).astype('f4')
            expected = None
            for name, prefix, handle in variants:
                out = np.empty((nt, 7), dtype='f4')
                assert bind(handle, prefix)(out.ctypes.data, weights.ctypes.data, typ, 7, 512, x.ctypes.data, nt, 2) == 0
                if expected is None:
                    expected = out.copy()
                else:
                    np.testing.assert_array_equal(out, expected, err_msg=f'{name}, type={typ}, nt={nt}')
                matrices += 1
    # All half encodings including subnormals, signed zero and infinities.
    halves = np.arange(65536, dtype='u2')
    for name, prefix, handle in variants:
        fn = getattr(handle, prefix + 'weight_at'); fn.argtypes = [P,C.c_int,C.c_int]; fn.restype = C.c_float
        observed = np.array([fn(halves.ctypes.data, 1, i) for i in range(65536)], dtype='f4')
        with np.errstate(invalid='ignore'):
            expected = halves.view('f2').astype('f4')
        np.testing.assert_array_equal(observed, expected, err_msg=name + ' half decode')
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(observed[finite].view('u4'), expected[finite].view('u4'))
    # Same expert kernels, row scheduling, omitted routes, both sides of nt=17.
    matrices_expert = [encoded(rng, 12, 640*2, 2560), encoded(rng, 12, 640*2, 2560), encoded(rng, 20, 2560*2, 640)]
    moe_checks = 0
    for nt in (1, 5, 18):
        x = rng.normal(0, .05, (nt, 2560)).astype('f4')
        ids = np.full((nt,10), -2, dtype='i4');ids[:,0]=0;ids[:,1]=1
        weights = np.zeros((nt,10),dtype='f4');weights[:,0]=.6;weights[:,1]=.3
        expected = None
        for name, prefix, handle in variants:
            create=getattr(handle,prefix+'create');create.argtypes=[C.c_int,C.c_int];create.restype=P
            destroy=getattr(handle,prefix+'destroy');destroy.argtypes=[P]
            moe=getattr(handle,prefix+'moe');moe.argtypes=[P,P,P,P,P,C.c_int,P,C.c_int,P,C.c_int,P,C.c_int];moe.restype=C.c_int
            workspace=create(18,2);assert workspace
            try:
                out=np.empty((nt,2560),dtype='f4')
                assert moe(workspace,out.ctypes.data,x.ctypes.data,ids.ctypes.data,weights.ctypes.data,nt,
                    matrices_expert[0].ctypes.data,12,matrices_expert[1].ctypes.data,12,matrices_expert[2].ctypes.data,20)==0
                if expected is None:expected=out.copy()
                else:np.testing.assert_array_equal(out,expected,err_msg=f'MoE {name} nt={nt}')
                ids[0,1]=0
                assert moe(workspace,out.ctypes.data,x.ctypes.data,ids.ctypes.data,weights.ctypes.data,nt,
                    matrices_expert[0].ctypes.data,12,matrices_expert[1].ctypes.data,12,matrices_expert[2].ctypes.data,20)==-1
                ids[0,1]=1;moe_checks+=1
            finally:destroy(workspace)
    # Dispatch is fixed once per process. An invalid forced ISA must fail safely.
    probe='import ctypes as c; x=c.CDLL('+repr(str(HERE/'build/libqwen_cpu.so'))+'); x.qc_backend_name.restype=c.c_char_p; print(x.qc_backend_name().decode())'
    for name in [n for n,_,_ in variants if n!='previous']+['invalid']:
        result=subprocess.check_output([sys.executable,'-c',probe],env={**os.environ,'QWEN_CPU_BACKEND':name},text=True).strip()
        assert result==('unsupported' if name=='invalid' else name)
    print(json.dumps({'backend':lib.qc_backend_name().decode(),'variants':[n for n,_,_ in variants],
                      'matmul_comparisons':matrices,'half_encodings_per_variant':65536,'moe_comparisons':moe_checks,
                      'comparison':'bit exact including previous library' if old else 'bit exact between available backends'}))

if __name__ == '__main__':main()
