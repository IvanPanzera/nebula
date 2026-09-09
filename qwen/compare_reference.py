#!/usr/bin/env python3
"""Run the pinned external engine serially, with identical rendered prompts."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from benchmark import quality
from chat import encode_chat
from monitor import Monitor

ROOT=Path(__file__).resolve().parent.parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',default='arithmetic,grounding,json,reasoning')
    ap.add_argument('--context',type=int,default=8192);ap.add_argument('--max-tokens',type=int)
    ap.add_argument('--out',default=str(ROOT/'work/qwen/reference_comparison'));args=ap.parse_args()
    assets=json.loads((ROOT/'work/qwen/assets_manifest.json').read_text())['files']
    targets=[a for a in assets if a['name'].startswith('UD-Q4_K_XL/')]
    for a in targets:
        path=ROOT/'models/qwen'/a['name'];stamp=path.stat()
        verified=json.loads(path.with_suffix('.gguf.verified.json').read_text())
        if stamp.st_size!=a['size'] or verified['sha256']!=a['sha256'] or verified['mtime_ns']!=stamp.st_mtime_ns:
            raise ValueError('target shard has not passed full verification: '+a['name'])
    model=ROOT/'models/qwen'/next(a['name'] for a in targets if '00001-of' in a['name'])
    binary=ROOT/'work/qwen/upstream/build/bin/qwen-reference-check'
    cases=json.loads((ROOT/'qwen/bench_prompts.json').read_text());wanted=set(args.cases.split(','))
    cases=[c for c in cases if c['id'] in wanted]
    if {c['id'] for c in cases}!=wanted:ap.error('unknown case ID')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Honor the native default index lock: never overlap two large model runs.
    with (ROOT/'work/qwen/model.index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[str(binary),str(model),str(args.context),str(out)]
        report=dict(external_reference=True,command=command,runs=[],complete=False,
            provenance=json.loads((ROOT/'work/qwen/upstream/provenance.json').read_text()),
            adapter_source_sha256=hashlib.sha256((ROOT/'qwen/reference_check.c').read_bytes()).hexdigest(),
            adapter_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            load_policy='ordinary tensors bulk-read into allocated RAM/GPU; PLE alone remains lazy and mmap-backed',
            numerical_contract='different quantized activation kernels; compare output quality, not bitwise logits')
        def save():
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        stderr=out/'engine.stderr.txt'
        with stderr.open('w') as ferr:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=ferr,text=True,bufsize=1)
            with Monitor(out/'resources.jsonl',pid=child.pid) as monitor:
                try:
                    for case in cases:
                        tokenizer,rendered,tokens=encode_chat([dict(role='user',content=case['prompt'])])
                        (out/(case['id']+'.prompt.txt')).write_text(rendered)
                        limit=args.max_tokens or case['max_tokens']
                        if len(tokens)+limit>args.context:raise ValueError('case exceeds reference context')
                        monitor.phase=case['id'];print('REFERENCE START',case['id'],flush=True);begin=time.monotonic()
                        child.stdin.write(f'{len(tokens)} {limit}\n'+' '.join(map(str,tokens))+'\n');child.stdin.flush()
                        line=child.stdout.readline()
                        if not line:raise RuntimeError('external reference ended: '+str(stderr))
                        row=json.loads(line);answer=tokenizer.decode(row['output_ids'],skip_special_tokens=False)
                        row.update(case=case['id'],prompt_tokens=len(tokens),elapsed_seconds=time.monotonic()-begin,
                            answer=answer,quality=quality(case,answer),at_output_limit=len(row['output_ids'])==limit,
                            initial_logits_file=str(out/f"case-{row['sequence']}.logits.f32"))
                        report['runs'].append(row);report['resources']=monitor.summary();save()
                        print(json.dumps(row,ensure_ascii=False),flush=True)
                    child.stdin.close();code=child.wait();report['returncode']=code
                    if code:raise RuntimeError('external reference failed: '+str(stderr))
                    report['complete']=True
                except BaseException as exc:
                    report['error']=f'{type(exc).__name__}: {exc}';child.terminate();child.wait();raise
                finally:
                    report['resources']=monitor.summary();save()
            report['resources']=monitor.summary();save()


if __name__=='__main__':main()
