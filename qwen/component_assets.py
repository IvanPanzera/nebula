#!/usr/bin/env python3
"""Bounded lexical tensor extraction for the MTP diagnostic, not a full target.

Prefer ranges already present in the ongoing download. Range hashes are recorded;
whole-shard SHA verification remains pending until the target download completes.
The production loader still requires all target tensors and verified full shards.
"""
import hashlib
import json
from pathlib import Path
from assets import WORK,MODELS,fetch_range,atomic_json,MTP

OUT=WORK/'mtp_probe';OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((WORK/'assets_manifest.json').read_text())
target_headers=[]
for asset in manifest['files']:
    h=json.loads((WORK/'headers'/(Path(asset['name']).name+'.json')).read_text())
    target_headers.append((asset,h))
md=next(h['metadata'] for a,h in target_headers if '00001-of' in a['name'])
entries=[]
sources=[]
for name in ('token_embd.weight','output.weight'):
    asset,t=next((a,t) for a,h in target_headers for t in h['tensors'] if t['name']==name)
    dest=OUT/(name+'.bin');stamp=dest.with_suffix('.json')
    if not dest.exists() or not stamp.exists() or dest.stat().st_size!=t['bytes']:
        partial=dest.with_suffix('.partial');start=partial.stat().st_size if partial.exists() else 0
        with partial.open('ab') as f:
            while start<t['bytes']:
                end=min(start+64*1024**2,t['bytes'])
                lo,hi=t['file_offset']+start,t['file_offset']+end
                local=MODELS/asset['name'];local=local if local.exists() else local.with_suffix('.gguf.partial')
                if local.exists() and local.stat().st_size>=hi:
                    with local.open('rb') as src:src.seek(lo);data=src.read(hi-lo)
                else:data=fetch_range(asset,lo,hi-1)
                if len(data)!=hi-lo:raise EOFError('short tensor component')
                f.write(data);f.flush();start=end
                print(f'{name}: {start/t["bytes"]:.0%}',flush=True)
        partial.replace(dest)
        digest=hashlib.sha256()
        with dest.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
        atomic_json(stamp,dict(asset=asset,tensor=t,sha256=digest.hexdigest(),full_shard_verified=False))
    sources.append(json.loads(stamp.read_text()))
    entries.append((dest,[dict(t,file_offset=0)]))
mtp=next((a,h) for a,h in target_headers if a['name']==MTP)
mtp_path=MODELS/mtp[0]['name']
checked=json.loads(mtp_path.with_suffix('.gguf.verified.json').read_text())
if checked['sha256']!=mtp[0]['sha256'] or mtp_path.stat().st_size!=mtp[0]['size']:raise ValueError('MTP not verified')
entries.append((mtp_path,mtp[1]['tensors']))
lines=['COB_QWEN_INDEX_V1','VERIFIED 1']
for key in ('layer_multipliers','head_offsets','head_vocab_sizes'):lines.append(' '.join(map(str,md['qwen4exp.ple.'+key])))
lines.append(str(md['qwen4exp.ple.eos_token_id']))
for i,(path,tensors) in enumerate(entries):lines.append(f'FILE\t{i}\t{path.stat().st_size}\t{path}')
for i,(path,tensors) in enumerate(entries):
    for t in tensors:
        dims=t['dims']+[1]*(4-len(t['dims']))
        lines.append(f'TENSOR\t{t["name"]}\t{i}\t{t["file_offset"]}\t{t["bytes"]}\t{t["type"]}\t'+'\t'.join(map(str,dims)))
(OUT/'probe.index').write_text('\n'.join(lines)+'\n')
atomic_json(OUT/'manifest.json',dict(diagnostic_only=True,target_components=sources,mtp_verified=checked))
print('MTP diagnostic components ready; not a runnable full target',flush=True)
