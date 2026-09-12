"""Prepare the official quantization from pinned GGUF assets, one shard at a time."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'qwen'))
from profiles import atomic_json
from download import download, digest


def log(message): print(message,flush=True)


class Quantizer:
    def __init__(self, library):
        self.lib=C.CDLL(str(library))
        self.lib.ggml_quantize_init.argtypes=[C.c_int]
        self.lib.ggml_quantize_init.restype=None
        self.lib.ggml_quantize_chunk.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int64,C.c_int64,C.c_int64,C.c_void_p]
        self.lib.ggml_quantize_chunk.restype=C.c_size_t
        for kind in (12,20):self.lib.ggml_quantize_init(kind)

    def convert(self,data,source_type,kind,rows,cols):
        import numpy as np
        from quant import dequant,row_size
        if source_type==kind:return data
        values=np.ascontiguousarray(dequant(data,source_type,rows,cols),dtype=np.float32)
        if not np.isfinite(values).all():raise ValueError('Non-finite source weight')
        out=np.empty(rows*row_size(kind,cols),np.uint8)
        count=self.lib.ggml_quantize_chunk(kind,values.ctypes.data,out.ctypes.data,0,rows,cols,None)
        if count!=out.nbytes:raise ValueError('Quantizer returned an unexpected size')
        return out.tobytes()


def build_quantizer(recipe, cache, jobs):
    info=recipe['quantizer'];archive=cache/'quantizer.tar.gz'
    download(info['url'],archive,info['archive_sha256'],progress=log)
    source=cache/('llama.cpp-'+info['commit']);build=cache/'quantizer-build'
    if not source.is_dir():
        with tarfile.open(archive,'r:gz') as tar:tar.extractall(cache,filter='data')
    # Only the CPU quantization library is used, never its inference engine.
    subprocess.run(['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=ON','-DGGML_NATIVE=OFF','-DGGML_AVX=OFF','-DGGML_AVX2=OFF',
        '-DGGML_AVX512=OFF','-DGGML_CUDA=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_SERVER=OFF'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','ggml-base','-j',str(jobs)],check=True)
    return build/'bin/libggml-base.so'


def validate_recipe(recipe):
    from quant import row_size
    if recipe.get('version')!=1 or len(recipe['parts'])!=5:raise ValueError('Unsupported model recipe')
    names=set()
    for part in recipe['parts']:
        end=0
        for t in part['tensors']:
            rows=math.prod(t['dims'][1:]);cols=t['dims'][0]
            if t['name'] in names or t['offset']<end:raise ValueError('Duplicated or overlapping tensor')
            names.add(t['name']);end=t['offset']+t['bytes']
            if t['bytes']!=rows*row_size(t['type'],cols) or t['source_bytes']!=rows*row_size(t['source_type'],cols):
                raise ValueError('Incorrect tensor size')
            if t['source_offset']<0 or t['source_offset']+t['source_bytes']>part['asset']['size']:
                raise ValueError('Tensor exceeds source asset')
        if max(256,end)!=part['bytes']:raise ValueError('Incorrect part size')
    if len(names)!=1256:raise ValueError('Incomplete model recipe')


def repack(source_path, part, destination, quantizer, recipe_hash):
    from quant import row_size
    partial=destination.with_suffix('.partial');checkpoint=destination.with_suffix('.resume.json')
    state={'recipe_sha256':recipe_hash,'done':[]}
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        if state['recipe_sha256']!=recipe_hash:raise ValueError('Conversion checkpoint belongs to a different release')
    if destination.exists():raise ValueError('Refusing to overwrite an unverified existing weight part')
    if state['done']:
        if not partial.is_file():raise ValueError('Missing partial conversion')
        with partial.open('rb') as f:
            for number,entry in enumerate(state['done']):
                t=part['tensors'][number];f.seek(t['offset']);h=hashlib.sha256();remaining=t['bytes']
                while remaining:
                    data=f.read(min(remaining,8*2**20))
                    if not data:raise ValueError('Truncated conversion checkpoint')
                    h.update(data);remaining-=len(data)
                if entry!=dict(name=t['name'],sha256=h.hexdigest()):raise ValueError('Damaged conversion checkpoint')
                log('Resuming verified tensor '+t['name'])
    mode='r+b' if partial.exists() else 'w+b'
    with source_path.open('rb') as source,partial.open(mode) as dest:
        for number,t in enumerate(part['tensors']):
            if number<len(state['done']):continue
            previous=part['tensors'][number-1] if number else None
            committed_end=previous['offset']+previous['bytes'] if previous else 0
            dest.seek(committed_end);dest.truncate()
            dest.write(bytes(t['offset']-committed_end))
            source.seek(t['source_offset']);remaining=t['source_bytes'];h=hashlib.sha256();written=0
            cols=t['dims'][0];rows=math.prod(t['dims'][1:]);step=8*2**20
            if t['source_type']!=t['type']:step=256*row_size(t['source_type'],cols)
            started=last=time.monotonic()
            while remaining:
                data=source.read(min(remaining,step))
                if not data:raise EOFError('Source tensor truncated')
                remaining-=len(data)
                if t['source_type']!=t['type']:
                    nr=len(data)//row_size(t['source_type'],cols)
                    data=quantizer.convert(data,t['source_type'],t['type'],nr,cols)
                dest.write(data);h.update(data);written+=len(data)
                if time.monotonic()-last>5:
                    log(f'Preparing {t["name"]}: {written/t["bytes"]:.0%}');last=time.monotonic()
            if written!=t['bytes']:raise ValueError('Converted tensor size mismatch')
            dest.flush();os.fsync(dest.fileno())
            state['done'].append(dict(name=t['name'],sha256=h.hexdigest()))
            atomic_json(checkpoint,state)
            log(f'Prepared {number+1}/{len(part["tensors"])}: {t["name"]} ({time.monotonic()-started:.1f}s)')
        dest.truncate(part['bytes'])
    sha=digest(partial,log)
    # Commit provenance before rename so an interrupted commit can be recovered.
    stamp=dict(recipe_sha256=recipe_hash,size=part['bytes'],sha256=sha)
    atomic_json(destination.with_suffix('.pending.json'),stamp)
    partial.replace(destination)
    stamp['mtime_ns']=destination.stat().st_mtime_ns
    atomic_json(destination.with_suffix('.json'),stamp)
    return stamp


def verified_part(path,recipe_hash):
    stamp_path=path.with_suffix('.json');pending=path.with_suffix('.pending.json')
    if not path.exists():return None
    if not stamp_path.exists() and pending.exists():
        stamp=json.loads(pending.read_text())
        if stamp['recipe_sha256']!=recipe_hash or digest(path,log)!=stamp['sha256']:raise ValueError('Incomplete part commit failed verification')
        stamp['mtime_ns']=path.stat().st_mtime_ns;atomic_json(stamp_path,stamp)
    if not stamp_path.exists():raise ValueError('Existing part has no verification record')
    stamp=json.loads(stamp_path.read_text());stat=path.stat()
    if stamp['recipe_sha256']!=recipe_hash or stat.st_size!=stamp['size']:raise ValueError('Weight part metadata changed')
    if stat.st_mtime_ns!=stamp['mtime_ns']:
        if digest(path,log)!=stamp['sha256']:raise ValueError('Weight part checksum mismatch')
        stamp['mtime_ns']=stat.st_mtime_ns;atomic_json(stamp_path,stamp)
    return stamp


def prepare(model_dir, work, cache, jobs=4):
    recipe_path=HERE/'model_recipe.json';recipe=json.loads(recipe_path.read_text());validate_recipe(recipe)
    recipe_hash=digest(recipe_path);model_dir.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    marker=model_dir/'.nebula-weights.json'
    if marker.exists():
        if json.loads(marker.read_text())['recipe_sha256']!=recipe_hash:raise ValueError('Model directory contains another model release')
    elif any(model_dir.iterdir()):raise ValueError('Model directory is not empty and is not managed by Nebula Setup')
    else:atomic_json(marker,dict(recipe_sha256=recipe_hash))
    quantizer=None;sources=[];lines=list(recipe['preamble']);tensor_lines=[]
    for part in recipe['parts']:
        target=model_dir/f'nebula-{part["number"]+1:02d}.bin';stamp=verified_part(target,recipe_hash)
        if not stamp:
            if quantizer is None:quantizer=Quantizer(build_quantizer(recipe,cache,jobs))
            asset=part['asset'];source=cache/Path(asset['name']).name
            partial=target.with_suffix('.partial')
            converted=min(part['bytes'],partial.stat().st_size) if partial.exists() else 0
            needed=part['bytes']-converted+max(0,asset['size']-(source.stat().st_size if source.exists() else (source.with_suffix(source.suffix+'.download').stat().st_size if source.with_suffix(source.suffix+'.download').exists() else 0)))+2*2**30
            missing=needed-shutil.disk_usage(model_dir).free
            if missing>0:raise RuntimeError(f'Free another {missing/2**30:.2f} GiB on the installation drive, then run Setup again.')
            url=f'https://huggingface.co/{recipe["repo"]}/resolve/{recipe["revision"]}/{asset["name"]}'
            download(url,source,asset['sha256'],asset['size'],log)
            stamp=repack(source,part,target,quantizer,recipe_hash)
            # Delete only our verified temporary source, after committing its replacement.
            source.unlink()
        log(f'Verified model part {part["number"]+1}/5')
        leftover=cache/Path(part['asset']['name']).name
        if leftover.exists():leftover.unlink() # verified final part supersedes our temporary download
        sources.append(dict(path=str(target),size=stamp['size'],mtime_ns=stamp['mtime_ns'],sha256=stamp['sha256']))
        lines.append(f'FILE\t{part["number"]}\t{part["bytes"]}\t{target}')
        for t in part['tensors']:
            tensor_lines.append('\t'.join(map(str,['TENSOR',t['name'],part['number'],t['offset'],t['bytes'],t['type'],*t['dims']])))
    text='\n'.join(lines+tensor_lines)+'\n';index=work/'model.index'
    temp=index.with_suffix('.tmp');temp.write_text(text);temp.replace(index)
    atomic_json(work/'model_index_manifest.json',dict(revision=recipe['revision'],weights_verified=True,
        index_sha256=hashlib.sha256(text.encode()).hexdigest(),tensors=len(tensor_lines),sources=sources,
        mtp_down_iq4=True,core_q4=True,layer2_q4=True,all_experts_q4=True,recipe_sha256=recipe_hash))
    return index


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args();prepare(args.model_dir,ROOT/'work/qwen',args.cache,max(1,min(args.jobs,8)))
