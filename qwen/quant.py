"""Bounded NumPy decoding of the GGUF formats used by Qwen's audited exports.

Used for independent component references and conversion preparation. Inference
uses the native CUDA kernels and does not materialize full float weight copies.
"""
import numpy as np

FORMATS={0:(1,4),1:(1,2),30:(1,2),2:(32,18),6:(32,22),7:(32,24),
         8:(32,34),12:(256,144),13:(256,176),14:(256,210),20:(32,18)}


def row_size(kind,cols):
    block,size=FORMATS[kind]
    if cols%block:raise ValueError('quant block does not divide the row')
    return cols//block*size


def dequant(data,kind,rows,cols):
    if kind==0:return np.frombuffer(data,dtype='<f4',count=rows*cols).reshape(rows,cols)
    if kind==1:return np.frombuffer(data,dtype='<f2',count=rows*cols).astype(np.float32).reshape(rows,cols)
    if kind==30:return (np.frombuffer(data,dtype='<u2',count=rows*cols).astype(np.uint32)<<16).view(np.float32).reshape(rows,cols)
    block,size=FORMATS[kind]
    p=np.frombuffer(data,dtype=np.uint8,count=rows*row_size(kind,cols)).reshape(rows,cols//block,size)
    def half(offset):return np.ascontiguousarray(p[...,offset:offset+2]).view('<f2').astype(np.float32)
    if kind==8:out=p[...,2:].view(np.int8).astype(np.float32)*half(0)
    elif kind in (2,6,7,20):
        start=8 if kind==7 else 6 if kind==6 else 2
        qs=p[...,start:start+16]
        q=np.concatenate([qs&15,qs>>4],axis=-1).astype(np.int16)
        if kind in (6,7):
            offset=4 if kind==7 else 2
            high=np.ascontiguousarray(p[...,offset:offset+4]).view('<u4')
            q|=(((high>>np.arange(32,dtype=np.uint32))&1)<<4).astype(np.int16)
        if kind==20:q=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113],np.float32)[q]
        else:q=q-(8 if kind==2 else 16 if kind==6 else 0)
        out=q*half(0)+(half(2) if kind==7 else 0)
    elif kind==14:
        i=np.arange(256);h,s,l=i//128,(i%128)//32,i%32
        lo=(p[...,h*64+(s%2)*32+l]>>((s//2)*4))&15
        hi=(p[...,128+h*32+l]>>(s*2))&3
        q=(lo.astype(np.int16)|(hi.astype(np.int16)<<4))-32
        scales=p[...,192+i//16].view(np.int8)
        out=q.astype(np.float32)*scales*half(208)
    else:
        sc=p[...,4:16]
        scale=np.concatenate([sc[...,:4]&63,(sc[...,8:12]&15)|((sc[...,:4]>>6)<<4)],axis=-1).astype(np.float32)
        minimum=np.concatenate([sc[...,4:8]&63,(sc[...,8:12]>>4)|((sc[...,4:8]>>6)<<4)],axis=-1).astype(np.float32)
        start=16 if kind==12 else 48
        qs=p[...,start:start+128].reshape(rows,cols//block,4,32)
        q=np.stack([qs&15,qs>>4],axis=-2).reshape(rows,cols//block,8,32).astype(np.int16)
        if kind==13:
            high=(p[...,16:48][...,None,:]>>np.arange(8,dtype=np.uint8)[None,None,:,None])&1
            q|=high.astype(np.int16)*16
        out=q*(half(0)*scale)[...,None]-(half(2)*minimum)[...,None]
    return out.reshape(rows,cols).astype(np.float32,copy=False)


def dequant_chunked(data,kind,rows,cols,chunk_rows=512):
    result=np.empty((rows,cols),np.float32);stride=row_size(kind,cols)
    for start in range(0,rows,chunk_rows):
        stop=min(start+chunk_rows,rows)
        result[start:stop]=dequant(memoryview(data)[start*stride:stop*stride],kind,stop-start,cols)
    return result
