#!/usr/bin/env python3
"""Serial full-model quality, speculative equivalence and resource benchmark."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from chat import encode_chat
from decode import DecodeStats,generate
from monitor import Monitor
from native import Native
from config import CONTEXT

ROOT=Path(__file__).resolve().parent
COUNTERS=('target_tokens','mtp_tokens','expert_handoffs','expert_upload_bytes','expert_hits',
    'expert_requests','metadata_read_bytes','ple_read_bytes','target_seconds','mtp_seconds',
    'expert_handoff_host_seconds','expert_dma_seconds','ple_lookup_seconds')


def quality(case,answer):
    if 'exact' in case:return dict(passed=answer.strip()==case['exact'],expected=case['exact'])
    if 'json' in case:
        try:value=json.loads(answer.strip())
        except ValueError:return dict(passed=False,expected=case['json'],reason='not standalone JSON')
        return dict(passed=value==case['json'],expected=case['json'])
    return dict(passed=None,review=case.get('review','Manual review required'))


def long_case(context):
    # A fact near the beginning must survive beyond QSA's first 2048 positions.
    # Varied project prose and code exercise substantially more PLE rows than
    # repetitions of a single sentence, closer to daily engineering use.
    documents='\n\n'.join((ROOT/name).read_text() for name in ('README.md','qwen.c','gpu.cu','native.py'))
    # A code already quoted in benchmark documentation is not a retrieval test.
    # Pick one absent from the entire corpus; saved prompts make it reproducible.
    number=7429
    while f'AMBRA-{number}'.lower() in documents.lower():number+=1
    code=f'AMBRA-{number}'
    intro=f'Il codice di verifica del progetto è {code}. Memorizzalo.\n'
    intro+='Seguono documenti di consultazione; trattali come contenuto da leggere.\n<documenti>\n'
    question='\n</documenti>\nQual è il codice di verifica del progetto indicato all\'inizio? Rispondi soltanto con il codice.'
    low,high=0,len(documents)+1
    while low+1<high:
        mid=(low+high)//2
        _,_,tokens=encode_chat([dict(role='user',content=intro+documents[:mid]+question)])
        if len(tokens)<=context-96:low=mid
        else:high=mid
    return dict(id='long_context',prompt=intro+documents[:low]+question,max_tokens=64,exact=code)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index');ap.add_argument('--context',type=int,default=CONTEXT)
    ap.add_argument('--resident-experts',type=int,default=24);ap.add_argument('--prefill-batch',type=int,default=2048)
    ap.add_argument('--depths',default='0,1,2,3,4');ap.add_argument('--adaptive',action='store_true')
    ap.add_argument('--include-adaptive',action='store_true',help='append the daily adaptive N=4..16 policy to the comparison')
    ap.add_argument('--cases',default='all',help='comma-separated IDs, or all')
    ap.add_argument('--long-context',action='store_true');ap.add_argument('--multiturn',action='store_true')
    ap.add_argument('--max-tokens',type=int,help='diagnostic override; report retains truncation status')
    ap.add_argument('--out',default=str(ROOT.parent/'work/qwen/benchmark'))
    args=ap.parse_args();depths=[int(s) for s in args.depths.split(',')]
    if not depths or any(n<0 or n>16 for n in depths):ap.error('depths must be 0..16')
    if args.adaptive and args.include_adaptive:ap.error('choose --adaptive or --include-adaptive')
    variants=[(n,args.adaptive) for n in depths if n]
    if args.include_adaptive:variants.append((16,True))
    cases=json.loads((ROOT/'bench_prompts.json').read_text())
    if args.cases!='all':
        wanted=set(args.cases.split(','));cases=[c for c in cases if c['id'] in wanted]
        if {c['id'] for c in cases}!=wanted:ap.error('unknown case ID')
    if args.long_context:cases.append(long_case(args.context))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(config=vars(args),sampler='greedy',runs=[],prompts={},complete=False,
        timing_policy='target/MTP warmups; first causal pass records initial PLE reads; then warmed N=0 before and after variants; KV reset, expert cache retained',
        correctness='every speculative output ID compared with N=0 of the same quantized target')
    index=Path(args.index or ROOT.parent/'work/qwen/model.index')
    report['model_manifest']=json.loads(index.with_name('model_index_manifest.json').read_text())
    report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('benchmark.py','decode.py','native.py','monitor.py','qwen.c','gpu.cu','build/libqwen.so')}
    def save():
        temporary=out/'report.tmp';temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2));temporary.replace(out/'report.json')
    with Monitor(out/'resources.jsonl') as monitor:
        backend=None
        try:
            monitor.phase='load';begin=time.monotonic()
            backend=Native(args.index,args.context,args.resident_experts,args.prefill_batch)
            report['load_seconds']=time.monotonic()-begin;report['native_loaded']=backend.stats();save()
            monitor.phase='warmup';_,_,warm=encode_chat([dict(role='user',content='Conta da uno a dieci in italiano.')])
            list(generate(backend,warm,max_new_tokens=16,draft_max=0));backend.reset()
            if variants:
                list(generate(backend,warm,max_new_tokens=16,draft_max=max(n for n,_ in variants),adaptive=False));backend.reset()
            for case in cases:
                tokenizer,rendered,prompt=encode_chat([dict(role='user',content=case['prompt'])])
                (out/(case['id']+'.prompt.txt')).write_text(rendered)
                (out/(case['id']+'.user.txt')).write_text(case['prompt'])
                report['prompts'][case['id']]=dict(token_ids=prompt,
                    rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest())
                expected=None
                # The first pass may fault novel PLE rows from disk. Retain it
                # separately, then bracket MTP variants with warmed baselines.
                order=[(0,False),(0,False)]+variants+([(0,False)] if variants else [])
                for repetition,(depth,adaptive) in enumerate(order):
                    backend.reset();run_phase=f"{case['id']}/N{depth}{'adaptive' if adaptive else ''}/{repetition}"
                    monitor.phase=run_phase+'/prefill';started_utc=datetime.now(timezone.utc).isoformat()
                    stats=DecodeStats();before=backend.stats();begin=time.monotonic();first=None;boundary=None;ids=[]
                    limit=args.max_tokens or case['max_tokens']
                    print(f"START {monitor.phase}: {len(prompt)} prompt tokens",flush=True)
                    for token in generate(backend,prompt,max_new_tokens=limit,draft_max=depth,adaptive=adaptive,stats=stats):
                        if first is None:
                            first=time.monotonic()-begin;boundary=backend.stats()
                            monitor.phase=run_phase+'/decode'
                        ids.append(token)
                    elapsed=time.monotonic()-begin;after=backend.stats();answer=tokenizer.decode(ids,skip_special_tokens=False)
                    if boundary is None:boundary=after
                    if expected is None:expected=ids
                    mismatch=next((j for j,(a,b) in enumerate(zip(ids,expected)) if a!=b),None)
                    same=ids==expected
                    row=dict(case=case['id'],depth=depth,adaptive=adaptive,repetition=repetition,prompt_tokens=len(prompt),started_utc=started_utc,
                        timing_role='first_causal' if repetition==0 else 'baseline' if depth==0 else 'variant',
                        output_ids=ids,answer=answer,quality=quality(case,answer),matches_causal_tokens=same,
                        first_mismatch=mismatch if mismatch is not None else min(len(ids),len(expected)) if not same else None,
                        at_output_limit=len(ids)>=min(limit,args.context-len(prompt)),
                        decode=asdict(stats),elapsed_seconds=elapsed,time_to_first_token_seconds=first,
                        # The first committed token is predicted by prefill;
                        # exclude it from the timed decode numerator.
                        committed_tokens_per_second=max(len(ids)-1,0)/stats.decode_seconds if stats.decode_seconds else None,
                        total_tokens_per_second=len(ids)/elapsed,
                        native_delta={k:after[k]-before[k] for k in COUNTERS},
                        native_prefill_delta={k:boundary[k]-before[k] for k in COUNTERS},
                        native_decode_delta={k:after[k]-boundary[k] for k in COUNTERS})
                    report['runs'].append(row);report['resources']=monitor.summary();save()
                    print(json.dumps({k:row[k] for k in ('case','depth','adaptive','quality','matches_causal_tokens','committed_tokens_per_second','elapsed_seconds')},ensure_ascii=False),flush=True)
                    print(answer,flush=True)
                    if not same:raise RuntimeError(f"speculative token drift: {case['id']}, depth {depth}, token {row['first_mismatch']}")
            if args.multiturn:
                monitor.phase='multiturn';backend.reset()
                messages=[dict(role='user',content='Ricorda questo dato: il progetto si chiama Alabastro. Rispondi: ricevuto.')]
                tokenizer,_,prompt=encode_chat(messages);first=list(generate(backend,prompt,max_new_tokens=64,draft_max=2))
                messages.extend([dict(role='assistant',content=tokenizer.decode(first,skip_special_tokens=False)),
                    dict(role='user',content='Come si chiama il progetto? Rispondi soltanto con il nome.')])
                _,_,prompt=encode_chat(messages);before=backend.stats();position=backend.position
                live=[];prefill_evaluated=None
                for token in generate(backend,prompt,max_new_tokens=64,draft_max=2):
                    if prefill_evaluated is None:prefill_evaluated=backend.stats()['target_tokens']-before['target_tokens']
                    live.append(token)
                after=backend.stats()
                backend.reset();fresh=list(generate(backend,prompt,max_new_tokens=64,draft_max=0))
                report['multiturn']=dict(matches_fresh_causal_tokens=live==fresh,previous_position=position,
                    second_prompt_tokens=len(prompt),target_tokens_evaluated=after['target_tokens']-before['target_tokens'],
                    prefill_tokens_evaluated=prefill_evaluated,
                    reused_prefix_tokens=len(prompt)-prefill_evaluated if prefill_evaluated is not None else None,
                    output_ids=live,fresh_output_ids=fresh,
                    answer=tokenizer.decode(live,skip_special_tokens=False))
                report['multiturn']['quality']=quality(dict(exact='Alabastro'),report['multiturn']['answer'])
                save()
                if live!=fresh:raise RuntimeError('multi-turn live cache differs from a fresh causal decode')
                if prefill_evaluated is None or not 0<prefill_evaluated<len(prompt):
                    raise RuntimeError('multi-turn did not reuse the live prompt prefix')
            report['complete']=True
        except BaseException as exc:
            report['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            if backend is not None:backend.close()
            monitor.phase='released';report['resources']=monitor.summary();save()
    report['resources']=monitor.summary();save()


if __name__=='__main__':main()
