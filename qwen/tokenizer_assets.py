#!/usr/bin/env python3
"""Fetch only tokenizer/template/config data from the pinned official model."""
import hashlib
import json
from pathlib import Path
import requests
from assets import WORK,atomic_json

REPO='Qwen/Qwen3.8-Flash-Next'
REVISION='de4b8e4d43b917e7706784d8bb445c9af86a3540'
OUT=WORK.parent/'research/qwen38/Qwen3.8-Flash-Next'


def ensure_tokenizer():
    response=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}',params={'blobs':'true'},timeout=60)
    response.raise_for_status();metadata=response.json()
    if metadata['sha']!=REVISION:raise ValueError('official model revision mismatch')
    files={s['rfilename']:s for s in metadata['siblings']};records=[];OUT.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        entry=files[name];path=OUT/name
        def valid(data):
            if len(data)!=entry['size']:return False
            if entry.get('lfs'):return hashlib.sha256(data).hexdigest()==entry['lfs']['sha256']
            return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entry['blobId']
        data=path.read_bytes() if path.exists() else b''
        if not valid(data):
            response=requests.get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',timeout=120);response.raise_for_status();data=response.content
            if not valid(data):raise ValueError('official asset hash mismatch: '+name)
            temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)
        records.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    config=json.loads((OUT/'config.json').read_text())['text_config']
    if config['output_gate_type']!='sigmoid' or config['num_experts_per_tok']!=10:raise ValueError('unsupported target configuration')
    atomic_json(WORK/'tokenizer_manifest.json',dict(repo=REPO,revision=REVISION,files=records))
    print('Official tokenizer, template and configuration verified',flush=True)


if __name__=='__main__':ensure_tokenizer()
