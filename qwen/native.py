"""Small ctypes binding; all model operators and recurrent state live in C/CUDA."""
import ctypes as C
import fcntl
import hashlib
import json
from pathlib import Path
from config import CONTEXT, DRAFT_MAX
from model_registry import GPU_LOCK

ROOT = Path(__file__).resolve().parent


class Stats(C.Structure):
    _fields_ = [(n,C.c_uint64) for n in (
        'gpu_weight_bytes','gpu_state_bytes','gpu_workspace_bytes','gpu_arena_bytes','target_tokens','mtp_tokens',
        'expert_handoffs','expert_upload_bytes','expert_hits','expert_requests','metadata_read_bytes',
        'ple_read_bytes','host_weight_bytes','host_staging_bytes')]+[
        (n,C.c_double) for n in ('target_seconds','mtp_seconds','expert_handoff_host_seconds','expert_dma_seconds','ple_lookup_seconds','core_load_seconds','host_load_seconds')]+[
        (n,C.c_int) for n in ('target_position','mtp_position','context','resident_per_layer')]


class HandoffProfile(C.Structure):
    _fields_ = [('layer_calls',C.c_uint64*48),('layer_handoffs',C.c_uint64*48)]


class Native:
    def __init__(self,index=None,context=CONTEXT,resident=24,prefill_batch=2048):
        self.gpu_lock=GPU_LOCK.open('a+b')
        try:
            fcntl.flock(self.gpu_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._open(index,context,resident,prefill_batch)
        except BlockingIOError:
            self.close()
            raise RuntimeError('another Qwen process holds the model lock') from None
        except Exception:
            self.close()
            raise

    def _open(self,index,context,resident,prefill_batch):
        if not 1<=prefill_batch<=2048:raise ValueError('prefill batch must be 1..2048')
        self.lib=C.CDLL(str(ROOT/'build/libqwen.so'))
        try:
            capacity=self.lib.qwen_max_draft
        except AttributeError:
            raise RuntimeError('Qwen native library is outdated; run make -C qwen before restarting the WebUI') from None
        capacity.argtypes,capacity.restype=[],C.c_int
        if capacity()!=DRAFT_MAX:
            raise RuntimeError('Qwen Python/native draft capacities differ; rebuild the native library')
        p,i=C.c_void_p,C.c_int
        signatures={
            'open':([C.c_char_p,i,i,i],p),'close':([p],None),'last_error':([],C.c_char_p),
            'reset':([p],i),'target':([p,p,i,i],i),'mtp_catchup':([p,p,i,i],i),
            'mtp_step':([p,i,i,i,p],i),'logits':([p,p,i,i],i),'top1':([p,p,i,i],i),
            'checkpoint':([p],i),'restore':([p],i),'get_stats':([p,p],None),
            'get_handoff_profile':([p,p],None),
        }
        for name,(args,res) in signatures.items():
            fn=getattr(self.lib,'qwen_'+name);fn.argtypes,fn.restype=args,res
        index=Path(index or ROOT.parent/'work/qwen/model.index')
        manifest=json.loads(index.with_name('model_index_manifest.json').read_text())
        if not manifest['weights_verified'] or hashlib.sha256(index.read_bytes()).hexdigest()!=manifest['index_sha256']:
            raise ValueError('Qwen index has not been generated from verified weights')
        if not manifest.get('sources'):raise ValueError('regenerate Qwen index with source stamps')
        for source in manifest['sources']:
            stamp=Path(source['path']).stat()
            if stamp.st_size!=source['size'] or stamp.st_mtime_ns!=source['mtime_ns']:
                raise ValueError('model asset changed after verification: '+source['path'])
        self.handle=self.lib.qwen_open(str(index).encode(),context,resident,prefill_batch)
        if not self.handle:
            raise RuntimeError(self.lib.qwen_last_error().decode())
        self.context=context;self.prefill_batch=prefill_batch;self.history=[];self.last_rows=0;self.saved=None;self.mtp_current=True

    def close(self):
        if getattr(self,'handle',None):
            self.lib.qwen_close(self.handle);self.handle=None
        if getattr(self,'gpu_lock',None):
            self.gpu_lock.close();self.gpu_lock=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def call(self,name,*args):
        rc=getattr(self.lib,'qwen_'+name)(self.handle,*args)
        if rc:
            raise RuntimeError(self.lib.qwen_last_error().decode())

    @property
    def position(self):return len(self.history)

    def reset(self):
        self.call('reset');self.history=[];self.last_rows=0;self.saved=None;self.mtp_current=True

    def target(self,tokens,logits=True):
        ids=(C.c_int*len(tokens))(*tokens)
        mode=2 if logits=='last' else int(logits)
        self.call('target',ids,len(tokens),mode)
        self.history.extend(tokens);self.last_rows=1 if mode==2 else len(tokens) if mode else 0

    def catchup(self,tokens,position):
        ids=(C.c_int*len(tokens))(*tokens)
        self.call('mtp_catchup',ids,len(tokens),position)

    def prefill(self,prompt,mtp=True):
        # Reuse a live prefix only when both model states are compatible with
        # the requested decoding mode; an MTP cache cannot be invented later.
        if self.history!=prompt[:len(self.history)] or (mtp and not self.mtp_current):
            common=0
            for a,b in zip(self.history,prompt):
                if a!=b:break
                common+=1
            if self.saved is not None and self.saved<=common and (not mtp or self.mtp_current):
                self.restore()
            else:
                self.call('reset');self.history=[];self.last_rows=0;self.saved=None
        if len(self.history)==len(prompt) and self.last_rows:
            self.mtp_current=mtp
            return
        if len(self.history)==len(prompt):
            self.call('reset');self.history=[];self.saved=None
        while len(self.history)<len(prompt):
            position=len(self.history);tokens=prompt[position:position+self.prefill_batch]
            self.target(tokens,logits='last' if position+len(tokens)==len(prompt) else False)
            if mtp:self.catchup(tokens,position)
        self.mtp_current=mtp

    def draft(self,token,position,target_hidden):
        out=C.c_int();self.call('mtp_step',token,position,int(target_hidden),C.byref(out));return out.value

    def top1(self):
        out=(C.c_int*self.last_rows)();self.call('top1',out,self.last_rows,0);return list(out)

    def logits(self,row=-1):
        if row<0:row+=self.last_rows
        out=(C.c_float*248320)();self.call('logits',out,row,0);return out

    def checkpoint(self):
        self.call('checkpoint');self.saved=len(self.history)

    def restore(self):
        self.call('restore');del self.history[self.saved:];self.last_rows=0

    def stats(self):
        value=Stats();self.lib.qwen_get_stats(self.handle,C.byref(value))
        return {name:getattr(value,name) for name,_ in Stats._fields_}

    def handoff_profile(self):
        value=HandoffProfile();self.lib.qwen_get_handoff_profile(self.handle,C.byref(value))
        return {name:list(getattr(value,name)) for name,_ in HandoffProfile._fields_}
