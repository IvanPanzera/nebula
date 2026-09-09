#!/usr/bin/env python3
"""Extract bounded PLE weights and actual hashed rows for independent checks."""
import hashlib
import json
from pathlib import Path
from assets import WORK,MODELS,fetch_range,atomic_json


def main():
    manifest=json.loads((WORK/'assets_manifest.json').read_text())
    headers=[(a,json.loads((WORK/'headers'/(Path(a['name']).name+'.json')).read_text())) for a in manifest['files']]
    md=next(h['metadata'] for a,h in headers if '00001-of' in a['name'])
    out=WORK/'ple_probe';out.mkdir(exist_ok=True)
    def read(asset,offset,size):
        path=MODELS/asset['name'];path=path if path.exists() else path.with_suffix('.gguf.partial')
        if path.exists() and path.stat().st_size>=offset+size:
            with path.open('rb') as src:src.seek(offset);data=src.read(size)
        else:data=fetch_range(asset,offset,offset+size-1)
        if len(data)!=size:raise EOFError('short PLE diagnostic range')
        return data
    if not (out/'weights.json').exists():
        entries=[];offset=0;digest=hashlib.sha256()
        with (out/'weights.bin').open('wb') as f:
            for asset,h in headers:
                for t in h['tensors']:
                    if not t['name'].startswith('blk.1.ple_'):continue
                    data=read(asset,t['file_offset'],t['bytes']);f.write(data);digest.update(data)
                    entries.append(dict(t,file_offset=offset,source_asset=asset['name'],source_file_offset=t['file_offset']))
                    offset+=len(data)
        atomic_json(out/'weights.json',dict(diagnostic_only=True,full_shard_verified=False,
            tensors=entries,size=offset,sha256=digest.hexdigest()))
    eos=md['qwen4exp.ple.eos_token_id'];mul=md['qwen4exp.ple.layer_multipliers']
    vocab=md['qwen4exp.ple.head_vocab_sizes'];offset=md['qwen4exp.ple.head_offsets']
    tokens=[248045,74455,198,17,0,eos,42,99,eos,eos,248319,1,248046,12,19,400,598,8000,9,7,6,5,4,3,2,1,eos,15,78,34,0,29]
    # Independent segment-based formulation, also preserving history for the
    # current EOS itself (only a preceding EOS starts a new segment).
    rows=[]
    for p,token in enumerate(tokens):
        boundary=max((i for i in range(p) if tokens[i]==eos),default=-1)
        previous=[tokens[p-j] if p-j>boundary else eos for j in range(3)]
        mixed=token*mul[0];row=[]
        for order in (2,3):
            mixed^=previous[order-1]*mul[order-1]
            row.extend(mixed%vocab[h]+offset[h] for h in range((order-2)*8,(order-1)*8))
        rows.append(row)
    asset,table=next((a,t) for a,h in headers for t in h['tensors'] if t['name']=='per_layer_token_embd.weight')
    packed=b''.join(read(asset,table['file_offset']+row*90,90) for token_rows in rows for row in token_rows)
    (out/'rows.bin').write_bytes(packed)
    atomic_json(out/'rows.json',dict(tokens=tokens,rows=rows,multiplier=mul,vocab=vocab,offset=offset,eos=eos,
        sha256=hashlib.sha256(packed).hexdigest(),source_asset=asset['name'],full_shard_verified=False))
    print('PLE: six projections/norms/conv and 512 actual hashed rows extracted',flush=True)


if __name__=='__main__':main()
