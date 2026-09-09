#!/usr/bin/env python3
"""Summarize committed-token timings against each case's bracketing baselines."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(report):
    if not report.get('complete') or report.get('error'):
        raise ValueError('benchmark has not completed successfully')
    cases=defaultdict(list)
    for row in report['runs']:
        if not row['matches_causal_tokens']:
            raise ValueError('cannot compare speed with divergent causal tokens')
        cases[row['case']].append(row)
    groups=defaultdict(list)
    for name,rows in cases.items():
        baselines=[r for r in rows if r['depth']==0 and r.get('timing_role')!='first_causal']
        if not baselines:raise ValueError('missing causal baseline: '+name)
        # Legacy first-full reports bracketed with an initially cold PLE pass.
        # Use only the final N=0 there instead of inflating the MTP speedup.
        if not any('timing_role' in r for r in rows):baselines=baselines[-1:]
        ids=baselines[0]['output_ids']
        if any(r['output_ids']!=ids for r in rows):
            raise ValueError('output IDs differ within case: '+name)
        baseline_seconds=sum(r['decode']['decode_seconds'] for r in baselines)/len(baselines)
        for row in rows:
            # Do not count the repeated N=0 as a second prompt in an aggregate.
            if row['depth']==0 and row is not baselines[0]:continue
            measured=row['decode'];seconds=baseline_seconds if row['depth']==0 else measured['decode_seconds']
            sources=baselines if row['depth']==0 else [row]
            def mean(section,key):return sum(r[section][key] for r in sources)/len(sources)
            def decode_mean(key):
                return mean('native_decode_delta',key) if all('native_decode_delta' in r for r in sources) else None
            tokens=max(len(ids)-1,0)
            adaptive=bool(row.get('adaptive',report['config'].get('adaptive',False))) and row['depth']>0
            groups[(row['depth'],adaptive)].append(dict(case=name,committed_decode_tokens=tokens,
                decode_seconds=seconds,baseline_decode_seconds=baseline_seconds,
                tokens_per_second=tokens/seconds if seconds else None,
                speedup_vs_causal=baseline_seconds/seconds if tokens and seconds else None,
                prefill_seconds=mean('decode','prefill_seconds'),
                proposed=measured['proposed'],accepted=measured['accepted'],
                semantic_misses=measured['semantic_misses'],replay_tokens=measured['replay_tokens'],
                expert_upload_bytes=mean('native_delta','expert_upload_bytes'),
                expert_dma_seconds=mean('native_delta','expert_dma_seconds'),
                ple_lookup_seconds=mean('native_delta','ple_lookup_seconds'),
                decode_expert_upload_bytes=decode_mean('expert_upload_bytes'),
                decode_expert_dma_seconds=decode_mean('expert_dma_seconds'),
                decode_expert_handoff_host_seconds=decode_mean('expert_handoff_host_seconds'),
                decode_expert_hits=decode_mean('expert_hits'),decode_expert_requests=decode_mean('expert_requests'),
                decode_ple_lookup_seconds=decode_mean('ple_lookup_seconds'),
                proposed_by_position=measured['proposed_by_position'],
                accepted_by_position=measured['accepted_by_position'],
                quality=row['quality'],at_output_limit=row['at_output_limit']))
    summaries=[]
    for (depth,adaptive),rows in sorted(groups.items()):
        def total(key):return sum(r[key] for r in rows)
        tokens=total('committed_decode_tokens');seconds=total('decode_seconds');proposed=total('proposed')
        positions=[]
        for j in range(4):
            n=sum(r['proposed_by_position'][j] for r in rows);a=sum(r['accepted_by_position'][j] for r in rows)
            positions.append(a/n if n else None)
        decode_native=None
        if all(r['decode_expert_upload_bytes'] is not None for r in rows):
            requests=total('decode_expert_requests')
            decode_native=dict(expert_upload_gib=total('decode_expert_upload_bytes')/1024**3,
                expert_upload_mib_per_committed_token=total('decode_expert_upload_bytes')/1024**2/tokens if tokens else None,
                expert_dma_seconds=total('decode_expert_dma_seconds'),
                expert_handoff_host_seconds=total('decode_expert_handoff_host_seconds'),
                expert_hit_ratio=total('decode_expert_hits')/requests if requests else None,
                ple_lookup_seconds=total('decode_ple_lookup_seconds'))
        summaries.append(dict(depth=depth,adaptive=adaptive,cases=len(rows),committed_decode_tokens=tokens,
            decode_seconds=seconds,committed_tokens_per_second=tokens/seconds if seconds else None,
            speedup_vs_causal=total('baseline_decode_seconds')/seconds if tokens and seconds else None,
            acceptance=total('accepted')/proposed if proposed else None,acceptance_by_position=positions,
            semantic_misses=total('semantic_misses'),replay_tokens=total('replay_tokens'),
            # Native counters include prefill. Label them explicitly; dividing
            # these by decode tokens would misrepresent the handoff cost.
            full_run_expert_upload_gib=total('expert_upload_bytes')/1024**3,
            full_run_expert_dma_seconds=total('expert_dma_seconds'),
            full_run_ple_lookup_seconds=total('ple_lookup_seconds'),
            decode_native=decode_native,
            quality_failures=[r['case'] for r in rows if r['quality']['passed'] is False],
            manual_review=[r['case'] for r in rows if r['quality']['passed'] is None],
            output_limit_cases=[r['case'] for r in rows if r['at_output_limit']],per_case=rows))
    return dict(config=report['config'],load_seconds=report['load_seconds'],resources=report['resources'],
        timing='sum of decode times and committed decode tokens; mean warmed bracketing N=0 times; legacy reports use only their final N=0',
        limitations='small prompt suite; manual quality review and long-context checks remain separate',depths=summaries)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('reports',nargs='+');ap.add_argument('--out')
    args=ap.parse_args();results=[]
    for name in args.reports:
        result=summarize(json.loads(Path(name).read_text()));result['source_report']=str(Path(name).resolve());results.append(result)
    output=json.dumps(results,ensure_ascii=False,indent=2)
    if args.out:Path(args.out).write_text(output+'\n')
    else:print(output)


if __name__=='__main__':main()
