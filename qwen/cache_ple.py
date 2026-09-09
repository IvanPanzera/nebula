#!/usr/bin/env python3
"""Copy only the immutable PLE payload to a fast local filesystem, unchanged."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'work/qwen'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',default=str(Path.home()/'.cache/city-of-brass/qwen'))
    ap.add_argument('--host-free-bytes',type=int,help='actual backing-drive free space, required for a new WSL cache')
    args=ap.parse_args()
    with (WORK/'ple_cache.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assets=json.loads((WORK/'assets_manifest.json').read_text())['files']
        targets=[a for a in assets if a['name'].startswith('UD-Q4_K_XL/')]
        selected=[]
        for asset in targets:
            header=json.loads((WORK/'headers'/(Path(asset['name']).name+'.json')).read_text())
            selected.extend((asset,t) for t in header['tensors'] if t['name']=='per_layer_token_embd.weight')
        if len(selected)!=1:raise ValueError('expected one audited PLE tensor')
        asset,tensor=selected[0];source=ROOT/'models/qwen'/asset['name'];stamp=source.stat()
        verified=json.loads(source.with_suffix('.gguf.verified.json').read_text())
        if verified['sha256']!=asset['sha256'] or stamp.st_size!=asset['size'] or stamp.st_mtime_ns!=verified['mtime_ns']:
            raise ValueError('PLE parent shard is not fully verified')
        manifest=WORK/'ple_cache.json'
        if manifest.exists():
            from derived import ple_cache
            path,info=ple_cache(targets,verify_hash=True)
            print('Verified existing PLE SSD cache:',path,flush=True);return
        directory=Path(args.directory).expanduser().resolve()
        directory.mkdir(parents=True,exist_ok=True)
        required=tensor['bytes']+24*1024**3
        if shutil.disk_usage(directory).free<required:raise ValueError('PLE cache needs its payload plus 24 GiB free headroom')
        is_wsl='microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower()
        if is_wsl and args.host_free_bytes is None:
            raise ValueError('provide actual host backing-drive free bytes; WSL virtual free space is insufficient evidence')
        if args.host_free_bytes is not None and args.host_free_bytes<required:
            raise ValueError('backing drive needs the PLE payload plus 24 GiB headroom')
        dest=directory/f'ple-IQ4_NL-{asset["sha256"][:12]}.bin'
        temporary=dest.with_suffix('.partial')
        if dest.exists():raise ValueError('untracked cache destination already exists: '+str(dest))
        digest=hashlib.sha256();copied=0;begin=time.monotonic();last=begin
        with source.open('rb') as src,temporary.open('wb') as dst:
            src.seek(tensor['file_offset'])
            while copied<tensor['bytes']:
                data=src.read(min(16*1024**2,tensor['bytes']-copied))
                if not data:raise EOFError('truncated PLE payload')
                dst.write(data);digest.update(data);copied+=len(data)
                if time.monotonic()-last>=30:
                    print(f'PLE SSD cache {copied/tensor["bytes"]:.1%}',flush=True);last=time.monotonic()
        if source.stat().st_mtime_ns!=stamp.st_mtime_ns:raise ValueError('source changed during PLE copy')
        temporary.replace(dest)
        info=dict(path=str(dest),source_asset=asset,source_tensor=tensor,
            tensor=dict(tensor,file_offset=0,offset=0),weights_changed=False,
            size=copied,mtime_ns=dest.stat().st_mtime_ns,sha256=digest.hexdigest(),
            seconds=time.monotonic()-begin,host_free_bytes_before=args.host_free_bytes)
        pending=manifest.with_suffix('.tmp');pending.write_text(json.dumps(info,indent=2));pending.replace(manifest)
        print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
