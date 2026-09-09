#!/usr/bin/env python3
"""Extract two complete target layers already present in downloaded ranges."""
import hashlib
import json
from pathlib import Path
from assets import WORK,MODELS,atomic_json


def main():
    manifest=json.loads((WORK/'assets_manifest.json').read_text())
    headers=[(a,json.loads((WORK/'headers'/(Path(a['name']).name+'.json')).read_text())) for a in manifest['files']]
    md=next(h['metadata'] for a,h in headers if '00001-of' in a['name'])
    out=WORK/'layer_probe';out.mkdir(exist_ok=True)
    for layer in (12,15):
        dest=out/f'layer{layer}.bin';metadata=out/f'layer{layer}.json'
        if not metadata.exists():
            source_tensors=[(a,t) for a,h in headers for t in h['tensors'] if t['name'].startswith(f'blk.{layer}.')]
            digest=hashlib.sha256();offset=0;entries=[]
            with dest.open('wb') as f:
                for asset,t in source_tensors:
                    source=MODELS/asset['name'];source=source if source.exists() else source.with_suffix('.gguf.partial')
                    if source.stat().st_size<t['file_offset']+t['bytes']:raise ValueError('required range still downloading: '+t['name'])
                    with source.open('rb') as src:
                        src.seek(t['file_offset']);remaining=t['bytes']
                        while remaining:
                            block=src.read(min(remaining,8*1024**2))
                            if not block:raise EOFError(t['name'])
                            f.write(block);digest.update(block);remaining-=len(block)
                    entries.append(dict(t,source_asset=asset['name'],source_file_offset=t['file_offset'],file_offset=offset));offset+=t['bytes']
            atomic_json(metadata,dict(diagnostic_only=True,full_shard_verified=False,sha256=digest.hexdigest(),size=offset,tensors=entries))
        info=json.loads(metadata.read_text())
        if dest.stat().st_size!=info['size']:raise ValueError('incomplete diagnostic component')
        lines=['COB_QWEN_INDEX_V1','VERIFIED 1']
        for key in ('layer_multipliers','head_offsets','head_vocab_sizes'):lines.append(' '.join(map(str,md['qwen4exp.ple.'+key])))
        lines.append(str(md['qwen4exp.ple.eos_token_id']))
        lines.append(f'FILE\t0\t{info["size"]}\t{dest}')
        for t in info['tensors']:
            dims=t['dims']+[1]*(4-len(t['dims']))
            lines.append(f'TENSOR\t{t["name"]}\t0\t{t["file_offset"]}\t{t["bytes"]}\t{t["type"]}\t'+'\t'.join(map(str,dims)))
        (out/f'layer{layer}.index').write_text('\n'.join(lines)+'\n')
        print(f'Layer {layer}: diagnostic tensors ready ({info["size"]/1024**3:.3f} GiB)',flush=True)


if __name__=='__main__':main()
