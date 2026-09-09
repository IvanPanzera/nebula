#!/usr/bin/env python3
"""Generate the narrow native loader index from audited GGUF metadata."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from derived import mtp_iq4,core_q4,ple_cache

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work/qwen"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata-only',action='store_true',help='prepare index before downloads finish; native loader still checks files')
    ap.add_argument('--mtp-iq4',action='store_true',help='use verified IQ4_NL MTP down projection; target weights unchanged')
    ap.add_argument('--core-q4',action='store_true',help='diagnostic target core Q4 candidate; requires quality validation')
    ap.add_argument('--no-ple-cache',action='store_true',help='diagnostic original PLE location; weights are identical')
    args=ap.parse_args()
    # A started SSD copy is a dependency of the next index. Its exclusive lock
    # is released only after an atomic manifest, or on a reported copy failure.
    cache_lock=(WORK/'ple_cache.lock').open('a')
    fcntl.flock(cache_lock,fcntl.LOCK_SH)
    index_lock=(WORK/'model.index.lock').open('a')
    try:fcntl.flock(index_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:ap.error('close the native/reference model before changing its index')
    manifest=json.loads((WORK/'assets_manifest.json').read_text())
    headers=[];sources=[]
    for asset in manifest['files']:
        h=json.loads((WORK/'headers'/(Path(asset['name']).name+'.json')).read_text())
        if h['asset']!=asset:
            raise ValueError('header/manifest mismatch')
        path=ROOT/'models/qwen'/asset['name']
        if not args.metadata_only:
            verified=json.loads(path.with_suffix('.gguf.verified.json').read_text())
            stat=path.stat()
            if verified['sha256']!=asset['sha256'] or stat.st_size!=asset['size'] or stat.st_mtime_ns!=verified['mtime_ns']:
                raise ValueError(f'unverified or modified weights: {path}')
            sources.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=asset['sha256']))
        headers.append((asset,path,h))
    if args.core_q4:
        targets=[asset for asset,path,h in headers if asset['name'].startswith('UD-Q4_K_XL/')]
        path,info=core_q4(targets,verify_hash=True)
        expected={t['name']:t for asset,path,h in headers if asset['name'].startswith('UD-Q4_K_XL/')
                  for t in h['tensors'] if t['type']==8 and '_exps.' not in t['name'] and 'ffn_gate_inp' not in t['name']}
        if {t['name']:t['source_tensor'] for t in info['tensors']}!=expected:
            raise ValueError('core conversion does not cover exactly the audited Q8 target matrices')
        headers=[(a,p,dict(h,tensors=[t for t in h['tensors'] if t['name'] not in expected])) for a,p,h in headers]
        headers.append((dict(size=info['size']),path,dict(tensors=info['tensors'])))
        sources.append(dict(path=str(path),size=info['size'],mtime_ns=info['mtime_ns'],sha256=info['sha256']))
    if args.mtp_iq4:
        source=next(asset for asset,path,h in headers if asset.get('name','').startswith('MTP/'))
        path,info=mtp_iq4(source,verify_hash=True);name=info['tensor']['name']
        headers=[(a,p,dict(h,tensors=[t for t in h['tensors'] if t['name']!=name])) for a,p,h in headers]
        headers.append((dict(size=info['size']),path,dict(tensors=[info['tensor']])))
        sources.append(dict(path=str(path),size=info['size'],mtime_ns=info['mtime_ns'],sha256=info['sha256']))
    use_ple_cache=not args.no_ple_cache and (WORK/'ple_cache.json').exists()
    if use_ple_cache:
        targets=[asset for asset,path,h in headers if asset.get('name','').startswith('UD-Q4_K_XL/')]
        path,info=ple_cache(targets,verify_hash=True);name=info['tensor']['name']
        original=[t for a,p,h in headers for t in h['tensors'] if t['name']==name]
        if original!=[info['source_tensor']]:raise ValueError('PLE cache does not match the audited source tensor')
        headers=[(a,p,dict(h,tensors=[t for t in h['tensors'] if t['name']!=name])) for a,p,h in headers]
        headers.append((dict(size=info['size']),path,dict(tensors=[info['tensor']])))
        sources.append(dict(path=str(path),size=info['size'],mtime_ns=info['mtime_ns'],sha256=info['sha256']))
    if len(headers)>8:raise ValueError('native index supports at most eight weight files')
    trunk=next(h for a,p,h in headers if '00001-of' in a.get('name',''))
    md=trunk['metadata']
    expected={'block_count':48,'embedding_length':2560,'expert_count':512,'expert_used_count':10,
              'expert_feed_forward_length':640,'attention.head_count':24,'attention.head_count_kv':2,
              'hyper_connection.count':4,'hyper_connection.low_rank':320}
    for k,v in expected.items():
        if md['qwen4exp.'+k]!=v:
            raise ValueError('unsupported model shape: '+k)
    lines=['COB_QWEN_INDEX_V1',f'VERIFIED {int(not args.metadata_only)}']
    for key in ('layer_multipliers','head_offsets','head_vocab_sizes'):
        values=md['qwen4exp.ple.'+key]
        lines.append(' '.join(map(str,values)))
    lines.append(str(md['qwen4exp.ple.eos_token_id']))
    tensors=[]
    for file_id,(asset,path,h) in enumerate(headers):
        if '\t' in str(path) or '\n' in str(path):raise ValueError('unsupported path character')
        lines.append(f'FILE\t{file_id}\t{asset["size"]}\t{path}')
        for t in h['tensors']:
            dims=t['dims']+[1]*(4-len(t['dims']))
            tensors.append(f'TENSOR\t{t["name"]}\t{file_id}\t{t["file_offset"]}\t{t["bytes"]}\t{t["type"]}\t'+ '\t'.join(map(str,dims)))
    lines.extend(tensors)
    text='\n'.join(lines)+'\n'
    path=WORK/'model.index';path.write_text(text)
    (WORK/'model_index_manifest.json').write_text(json.dumps(dict(revision=manifest['revision'],index_sha256=hashlib.sha256(text.encode()).hexdigest(),tensors=len(tensors),weights_verified=not args.metadata_only,sources=sources,mtp_down_iq4=args.mtp_iq4,core_q4=args.core_q4,ple_ssd_cache=use_ple_cache),indent=2))
    print(f'{path}: {len(tensors)} tensors, weights_verified={not args.metadata_only}')


if __name__=='__main__':main()
