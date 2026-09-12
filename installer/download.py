"""Resumable downloads with pinned SHA256; no private accounts or tokens."""
import hashlib
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def digest(path, progress=None):
    h=hashlib.sha256();done=0;last=0
    with Path(path).open('rb') as source:
        while data:=source.read(8*2**20):
            h.update(data);done+=len(data)
            if progress and time.monotonic()-last>5:
                progress(f'Checking {Path(path).name}: {done/2**30:.1f} GiB');last=time.monotonic()
    return h.hexdigest()


def download(url, path, sha256, size=None, progress=print):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if (size is None or path.stat().st_size==size) and digest(path,progress)==sha256:return path
        raise ValueError(f'Checksum mismatch in {path.name}; remove this damaged installer download and retry.')
    partial=path.with_suffix(path.suffix+'.download')
    for attempt in range(5):
        try:
            start=partial.stat().st_size if partial.exists() else 0
            if size is not None and start==size:break
            if size is not None and start>size:raise ValueError('Partial download is larger than the asset')
            req=Request(url,headers={'User-Agent':'Nebula-Setup/1.0','Accept-Encoding':'identity'})
            if start:req.add_header('Range',f'bytes={start}-')
            with urlopen(req,timeout=90) as response:
                if start and response.status==206:
                    if not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise ValueError('Download resume offset mismatch')
                elif response.status==200:start=0
                elif response.status!=206:raise ValueError(f'Unexpected HTTP status {response.status}')
                if not start and response.status==206 and not response.headers.get('Content-Range','').startswith('bytes 0-'):
                    raise ValueError('Download starts at the wrong offset')
                last=0
                with partial.open('ab' if start else 'wb') as dest:
                    while data:=response.read(8*2**20):
                        dest.write(data);start+=len(data)
                        if size is not None and start>size:raise ValueError('Download size exceeds manifest')
                        if time.monotonic()-last>3:
                            progress(f'Downloading {path.name}: {start/2**30:.2f}'+(f' / {size/2**30:.2f}' if size else '')+' GiB')
                            last=time.monotonic()
                    dest.flush();os.fsync(dest.fileno())
            if size is not None and start!=size:raise OSError('Download ended before the complete file arrived')
            break
        except (OSError, TimeoutError) as exc:
            if attempt==4:raise RuntimeError(f'Download interrupted. Run Setup again to resume: {exc}') from exc
            progress(f'Connection interrupted; retry {attempt+1}/4.');time.sleep(min(2**attempt,8))
    if digest(partial,progress)!=sha256:
        # This is our own temporary download, not an existing user/model file.
        partial.unlink()
        raise ValueError(f'Checksum failed for {path.name}; rerun Setup to download it again.')
    partial.replace(path);return path
