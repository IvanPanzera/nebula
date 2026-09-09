#!/usr/bin/env python3
"""Serial 24K regression, long retrieval, MTP equivalence and exact KV boundary.

Uses the existing verified index. The model lock must be free. No second model
is loaded, and old 8K benchmark outputs remain an immutable reference.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from benchmark import long_case
from chat import encode_chat
from config import CONTEXT
from decode import DecodeStats, generate
from monitor import Monitor
from native import Native

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'work/qwen/context24k'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference=json.loads((ROOT.parent/'work/qwen/benchmark_daily_q4/report.json').read_text())
    report=dict(context=CONTEXT,prefill_batch=2048,resident_experts=24,runs=[],complete=False,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    ('qwen.c','gpu.cu','native.py','decode.py','config.py','check_context.py','build/libqwen.so')})
    def save():
        tmp=OUT/'report.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(OUT/'report.json')
    with Monitor(OUT/'resources.jsonl') as monitor:
        try:
            monitor.phase='load';started=time.monotonic()
            with Native(context=CONTEXT,resident=24,prefill_batch=2048) as backend:
                report['load_seconds']=time.monotonic()-started;report['native_loaded']=backend.stats();save()
                print(f"Loaded in {report['load_seconds']:.1f}s; context {CONTEXT}, prefill 2048, cache 24",flush=True)
                def run(name,prompt,depth,max_tokens,expected=None,reset=True):
                    if reset:backend.reset()
                    stats=DecodeStats();before=backend.stats();prefill=None;ids=[]
                    monitor.phase=name+'/prefill'
                    for token in generate(backend,prompt,max_new_tokens=max_tokens,draft_max=depth,adaptive=False,stats=stats):
                        if prefill is None:
                            prefill=backend.stats()['target_tokens']-before['target_tokens'];monitor.phase=name+'/decode'
                        ids.append(token)
                    tokenizer,_,_=encode_chat([dict(role='user',content='')])
                    row=dict(name=name,depth=depth,prompt_tokens=len(prompt),output_ids=ids,
                        answer=tokenizer.decode(ids),stats=asdict(stats),
                        reused_tokens=len(prompt)-prefill if prefill is not None else None,
                        tokens_per_second=max(0,len(ids)-1)/stats.decode_seconds if stats.decode_seconds else 0,
                        matches_reference=ids==expected if expected is not None else None)
                    report['runs'].append(row);save()
                    print(json.dumps({k:row[k] for k in ('name','prompt_tokens','answer','reused_tokens','tokens_per_second','matches_reference')},ensure_ascii=False),flush=True)
                    if expected is not None and ids!=expected:raise AssertionError(name+': token regression')
                    return row
                # The saved prompts do not change when current documentation or
                # implementation changes. They cover PLE, GDN, QSA and MTP.
                for old in reference['runs']:
                    if old['depth']!=1:continue
                    run('old8k/'+old['case'],reference['prompts'][old['case']]['token_ids'],1,
                        len(old['output_ids'])+1,old['output_ids'])
                case=long_case(CONTEXT)
                messages=[dict(role='user',content=case['prompt'])]
                tokenizer,rendered,prompt=encode_chat(messages)
                if not CONTEXT-128<=len(prompt)<CONTEXT:raise AssertionError('long prompt does not exercise full context')
                (OUT/'long_context.user.txt').write_text(case['prompt'])
                (OUT/'long_context.prompt.txt').write_text(rendered)
                report['long_prompt_ids']=prompt;report['expected_code']=case['exact'];save()
                causal=run('24k/causal',prompt,0,32)
                speculative=run('24k/mtp',prompt,1,32,causal['output_ids'])
                report['retrieval_passed']=causal['answer'].strip()==case['exact'];save()
                followup=messages+[dict(role='assistant',content=speculative['answer']),
                    dict(role='user',content='Ripeti soltanto lo stesso codice di verifica.')]
                _,_,next_prompt=encode_chat(followup)
                if len(next_prompt)>=CONTEXT:raise AssertionError('follow-up fixture exceeds context')
                turn=run('24k/prefix_reuse',next_prompt,1,24,reset=False)
                report['followup_retrieval_passed']=turn['answer'].strip()==case['exact']
                if turn['reused_tokens']<CONTEXT-128:raise AssertionError('long prefix was not reused')
                # Reach the last physical KV row with an ordinary vocabulary ID.
                # Then verify checkpoint/rollback and rejection beyond capacity.
                monitor.phase='24k/final_row'
                newline=tokenizer.encode('\n',add_special_tokens=False).ids[0]
                edge=next_prompt+turn['output_ids']
                edge=(edge+[newline]*CONTEXT)[:CONTEXT-1]
                backend.prefill(edge,mtp=True);backend.checkpoint();token=backend.top1()[0]
                backend.target([token]);backend.catchup([token],CONTEXT-1)
                logits=bytes(backend.logits());backend.restore()
                backend.target([token]);backend.catchup([token],CONTEXT-1)
                if bytes(backend.logits())!=logits:raise AssertionError('final row replay changed logits')
                try:backend.target([token])
                except RuntimeError as exc:report['capacity_error']=str(exc)
                else:raise AssertionError('target accepted a token beyond capacity')
                if backend.position!=CONTEXT:raise AssertionError('capacity rejection changed history')
                report['final_row_replay_exact']=True;report['native_final']=backend.stats();save()
                if not report['retrieval_passed'] or not report['followup_retrieval_passed']:
                    raise AssertionError('24K retrieval quality failed')
                report['complete']=True;save()
        except Exception as exc:
            report['error']=str(exc);save();raise
        finally:
            monitor.sample();report['resources']=monitor.summary();save()
    print('24K context validation passed.',flush=True)


if __name__=='__main__':main()
