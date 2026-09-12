"""Small ctypes binding; all model operators and recurrent state live in C/CUDA."""
import ctypes as C
import fcntl
import hashlib
import json
from pathlib import Path
from config import CONTEXT, DRAFT_MAX, RESIDENT_EXPERTS, CPU_THREADS, HOTLIST_SHA256, PREFILL_BATCH
from model_registry import GPU_LOCK
from storage import load_storage, load_ranking

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


class CPUStats(C.Structure):
    _fields_ = [(n,C.c_uint64) for n in ('layer_handoffs','token_layers','missing_selections','activation_bytes')]+[
        (n,C.c_double) for n in ('compute_seconds','handoff_seconds')]


class OptimizationStats(C.Structure):
    _fields_ = [(n,C.c_uint64) for n in ('retained_tokens','prefix_restores','graph_launches','graph_captures')]+[
        ('prefix_restore_seconds',C.c_double)]


class PruneStats(C.Structure):
    _fields_ = [(n,C.c_uint64) for n in ('token_layers','missing_selections','skipped_selections','avoided_token_handoffs')]+[
        ('skipped_mass',C.c_double)]


class StorageStats(C.Structure):
    _fields_ = [(n,C.c_uint64) for n in ('ram_expert_bytes','transient_peak_bytes','expert_read_bytes','expert_reads')]+[
        (n,C.c_double) for n in ('expert_io_seconds','expert_wait_seconds')]+[
        (n,C.c_uint64) for n in ('ple_disk_bytes','ple_cache_hits','ple_cache_misses')]+[
        (n,C.c_int) for n in ('ram_experts_per_layer','ple_on_ssd')]


def load_hotlist(index, resident):
    """Fail before loading large weights if the selected profile is incompatible."""
    raw=(ROOT/'hotlist.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=HOTLIST_SHA256:
        raise ValueError('The official hotlist has changed: restore qwen/hotlist.json.')
    hot=json.loads(raw)
    if resident!=RESIDENT_EXPERTS or hot['resident_per_layer']!=resident:
        raise ValueError(f'The official configuration requires {RESIDENT_EXPERTS} fixed experts per layer.')
    if hot['model_index_sha256']!=hashlib.sha256(index.read_bytes()).hexdigest():
        raise ValueError('The hotlist does not match the verified Qwen weights.')
    rows=hot['experts']
    if len(rows)!=48 or any(len(row)!=resident or len(set(row))!=resident or
                           any(type(e) is not int or not 0<=e<512 for e in row) for row in rows):
        raise ValueError('Invalid hotlist: expected 48 layers with distinct experts.')
    return [e for row in rows for e in row]


class Native:
    def __init__(self,index=None,context=CONTEXT,resident=RESIDENT_EXPERTS,prefill_batch=PREFILL_BATCH,storage_config=None):
        self.gpu_lock=GPU_LOCK.open('a+b')
        try:
            fcntl.flock(self.gpu_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.storage=load_storage(storage_config,resident)
            self._open(index,context,resident,prefill_batch)
        except BlockingIOError:
            self.close()
            raise RuntimeError('another Qwen process holds the model lock') from None
        except Exception:
            self.close()
            raise

    def _open(self,index,context,resident,prefill_batch):
        if not 1<=prefill_batch<=8192:raise ValueError('prefill batch must be 1..8192')
        self.lib=C.CDLL(str(ROOT/'build/libqwen.so'))
        try:
            capacity=self.lib.qwen_max_draft
        except AttributeError:
            raise RuntimeError('Qwen native library is outdated; run make -C qwen before restarting the WebUI') from None
        capacity.argtypes,capacity.restype=[],C.c_int
        if capacity()!=DRAFT_MAX:
            raise RuntimeError('Qwen Python/native draft capacities differ; rebuild the native library')
        if not hasattr(self.lib,'qwen_handoff_version') or self.lib.qwen_handoff_version()!=1:
            raise RuntimeError('Rebuild the native engine: the official CPU MoE handoff API is required.')
        if not hasattr(self.lib,'qwen_ple_ram_bytes'):
            raise RuntimeError('Rebuild the native engine: the N-gram RAM API is required.')
        if not hasattr(self.lib,'qwen_set_cpu_threads'):
            raise RuntimeError('Rebuild the native engine: CPU thread control is missing.')
        if not hasattr(self.lib,'qwen_commit_prefix') or not hasattr(self.lib,'qwen_get_optimization_stats'):
            raise RuntimeError('Rebuild the native engine: prefix recovery and CUDA Graph counters are missing.')
        if not hasattr(self.lib,'qwen_verify'):
            raise RuntimeError('Rebuild the native engine: verification level control is missing.')
        if not hasattr(self.lib,'qwen_open_storage'):
            raise RuntimeError('Rebuild Qwen: the native storage API is missing')
        p,i=C.c_void_p,C.c_int
        signatures={
            'open':([C.c_char_p,i,i,i],p),'close':([p],None),'last_error':([],C.c_char_p),
            'reset':([p],i),'target':([p,p,i,i],i),'mtp_catchup':([p,p,i,i],i),
            'mtp_step':([p,i,i,i,p],i),'logits':([p,p,i,i],i),'top1':([p,p,i,i],i),
            'verify':([p,p,i,i,C.c_float,p,p],i),
            'checkpoint':([p],i),'restore':([p],i),'get_stats':([p,p],None),
            'get_handoff_profile':([p,p],None),'commit_prefix':([p,i],i),'get_optimization_stats':([p,p],None),
            'set_hotlist':([p,p,i,i],i),'get_cpu_stats':([p,p],None),'get_prune_stats':([p,p],i),
            'ple_ram_bytes':([p],C.c_uint64),
            'set_cpu_threads':([p,i],i),'cpu_threads':([p],i),
            'open_storage':([C.c_char_p,i,i,i,p,i,i,C.c_uint64,i],p),
            'get_storage_stats':([p,p],None),
        }
        for name,(args,res) in signatures.items():
            fn=getattr(self.lib,'qwen_'+name);fn.argtypes,fn.restype=args,res
        index=Path(index or ROOT.parent/'work/qwen/model.index')
        manifest=json.loads(index.with_name('model_index_manifest.json').read_text())
        if not manifest['weights_verified'] or hashlib.sha256(index.read_bytes()).hexdigest()!=manifest['index_sha256']:
            raise ValueError('Qwen index has not been generated from verified weights')
        if not manifest.get('layer2_q4'):
            raise ValueError('The official version requires layer 2 gate and up matrices in Q4_K: update the index and hotlist.')
        if not manifest.get('all_experts_q4'):
            raise ValueError('The official version requires all expert matrices at 4 bits: update the index and hotlist.')
        if not manifest.get('sources'):raise ValueError('regenerate Qwen index with source stamps')
        for source in manifest['sources']:
            stamp=Path(source['path']).stat()
            if stamp.st_size!=source['size'] or stamp.st_mtime_ns!=source['mtime_ns']:
                raise ValueError('model asset changed after verification: '+source['path'])
        hot=load_hotlist(index,resident)
        cfg=self.storage
        ranking=load_ranking(hot) if cfg['ram_experts_per_layer']<512 else None
        ranks=(C.c_int*len(ranking))(*ranking) if ranking else None
        self.handle=self.lib.qwen_open_storage(str(index).encode(),context,resident,prefill_batch,ranks,
            cfg['ram_experts_per_layer'],int(cfg['ngram']=='ssd'),cfg['buffer_mib']*2**20,cfg['io_threads'])
        if not self.handle:
            raise RuntimeError(self.lib.qwen_last_error().decode())
        if cfg['ngram']=='ram' and not self.lib.qwen_ple_ram_bytes(self.handle):
            raise RuntimeError('The N-gram table was not loaded into RAM.')
        ids=(C.c_int*len(hot))(*hot)
        self.call('set_hotlist',ids,len(hot),CPU_THREADS)
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

    @property
    def cpu_threads(self):return self.lib.qwen_cpu_threads(self.handle)

    def set_cpu_threads(self,threads):
        """Call only when no generation is running; no model reload is needed."""
        if type(threads) is not int or not 1<=threads<=28:raise ValueError('CPU threads must be 1..28')
        self.call('set_cpu_threads',threads)
        if self.cpu_threads!=threads:raise RuntimeError('CPU thread configuration was not applied')

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

    def verify(self,proposals,top_k,min_ratio):
        count=len(proposals)
        ids=(C.c_int*count)(*proposals);best=(C.c_int*(count+1))();eligible=(C.c_int*count)()
        self.call('verify',ids,count,top_k,min_ratio,best,eligible)
        return list(best),list(eligible)

    def checkpoint(self):
        self.call('checkpoint');self.saved=len(self.history)

    def restore(self):
        self.call('restore');del self.history[self.saved:];self.last_rows=0

    def commit_prefix(self, keep):
        self.call('commit_prefix',keep)
        del self.history[self.saved+keep:]
        self.last_rows=keep

    def optimization_stats(self):
        value=OptimizationStats();self.lib.qwen_get_optimization_stats(self.handle,C.byref(value))
        return {name:getattr(value,name) for name,_ in OptimizationStats._fields_}

    def stats(self):
        value=Stats();self.lib.qwen_get_stats(self.handle,C.byref(value))
        result={name:getattr(value,name) for name,_ in Stats._fields_}
        result['ple_host_bytes']=self.lib.qwen_ple_ram_bytes(self.handle)
        return result

    def storage_stats(self):
        value=StorageStats();self.lib.qwen_get_storage_stats(self.handle,C.byref(value))
        return {name:getattr(value,name) for name,_ in StorageStats._fields_}

    def handoff_profile(self):
        value=HandoffProfile();self.lib.qwen_get_handoff_profile(self.handle,C.byref(value))
        return {name:list(getattr(value,name)) for name,_ in HandoffProfile._fields_}

    def cpu_stats(self):
        value=CPUStats();self.lib.qwen_get_cpu_stats(self.handle,C.byref(value))
        return {name:getattr(value,name) for name,_ in CPUStats._fields_}

    def prune_stats(self):
        value=PruneStats();self.call('get_prune_stats',C.byref(value))
        return {name:getattr(value,name) for name,_ in PruneStats._fields_}
