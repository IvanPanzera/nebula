"""Validate offline quantization candidates and their provenance."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def ple_cache(source_assets,verify_hash=False):
    info=json.loads((ROOT/'work/qwen/ple_cache.json').read_text())
    path=Path(info['path']);stamp=path.stat();t=info['tensor'];original=info['source_tensor']
    if (info['source_asset'] not in source_assets or info['weights_changed'] or
        t!=dict(original,file_offset=0,offset=0) or t['name']!='per_layer_token_embd.weight' or
        t['dims']!=[160,320001536] or t['type']!=20 or t['bytes']!=28800138240 or
        stamp.st_size!=t['bytes'] or stamp.st_size!=info['size'] or stamp.st_mtime_ns!=info['mtime_ns']):
        raise ValueError('PLE cache provenance or tensor mismatch')
    if verify_hash:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda:f.read(8*1024**2),b''):digest.update(data)
        if digest.hexdigest()!=info['sha256']:raise ValueError('PLE cache SHA256 mismatch')
    return path,info


def mtp_iq4(source_asset,verify_hash=False):
    path=ROOT/'models/qwen/derived/mtp-down-IQ4_NL.bin'
    info=json.loads(path.with_suffix('.json').read_text());stamp=path.stat();t=info['tensor']
    if (info['source_asset']!=source_asset or info['target_weights_changed'] or
        t['name']!='blk.48.ffn_down_exps.weight' or t['dims']!=[640,2560,512] or
        t['type']!=20 or t['bytes']!=471859200 or t['file_offset']!=0 or
        stamp.st_size!=t['bytes'] or stamp.st_mtime_ns!=info['mtime_ns']):
        raise ValueError('derived MTP tensor or provenance mismatch')
    if verify_hash:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda:f.read(8*1024**2),b''):digest.update(data)
        if digest.hexdigest()!=info['sha256']:raise ValueError('derived MTP SHA256 mismatch')
    return path,info


def core_q4(source_assets,verify_hash=False):
    import math
    path=ROOT/'models/qwen/derived/target-core-Q4.bin'
    info=json.loads(path.with_suffix('.json').read_text());stamp=path.stat()
    if (info['source_assets']!=source_assets or info['routed_experts_changed'] or info['norms_and_routers_changed'] or
        stamp.st_size!=info['size'] or stamp.st_mtime_ns!=info['mtime_ns']):
        raise ValueError('derived core provenance mismatch')
    offset=0;names=set()
    for t in info['tensors']:
        original=t['source_tensor'];kind=12 if t['dims'][0]%256==0 else 20
        if (t['name'] in names or t['name']!=original['name'] or t['dims']!=original['dims'] or
            original['type']!=8 or '_exps.' in t['name'] or 'ffn_gate_inp' in t['name'] or t['name'].startswith('blk.48.') or
            t['type']!=kind or t['file_offset']!=offset or
            t['bytes']!=(t['dims'][0]//256*144 if kind==12 else t['dims'][0]//32*18)*math.prod(t['dims'][1:])):
            raise ValueError('invalid derived core tensor: '+t['name'])
        names.add(t['name']);offset+=t['bytes']
    if offset!=info['size'] or not names:raise ValueError('incomplete derived core')
    if verify_hash:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda:f.read(8*1024**2),b''):digest.update(data)
        if digest.hexdigest()!=info['sha256']:raise ValueError('derived core SHA256 mismatch')
    return path,info
