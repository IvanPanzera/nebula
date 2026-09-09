#!/usr/bin/env python3
"""Validate 24K through an already-running WebUI, reusing its one model.

Waits for the current user response to finish, then sends serial test requests.
Does not write browser history. No Native/model object is created here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen

from benchmark import long_case
from chat import encode_chat
from config import CONTEXT
from monitor import Monitor

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'work/qwen/context24k'
BASE='http://127.0.0.1:8090'


def status():
    with urlopen(BASE+'/api/status',timeout=10) as response:return json.load(response)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--worker-pid',type=int,required=True)
    ap.add_argument('--long-only',action='store_true',help='retry the saved long fixture; preserve the original report and skip passed short tests')
    args=ap.parse_args()
    command=Path(f'/proc/{args.worker_pid}/cmdline').read_bytes()
    if b'web_worker.py' not in command:raise ValueError('PID is not the WebUI model worker')
    OUT.mkdir(parents=True,exist_ok=True)
    reference=json.loads((ROOT.parent/'work/qwen/benchmark_daily_q4/report.json').read_text())
    prefix='web_retry' if args.long_only else 'web'
    previous_long=json.loads((OUT/'web_report.json').read_text()) if args.long_only else None
    report=dict(context=CONTEXT,prefill_batch=2048,resident_experts=24,worker_pid=args.worker_pid,
        transport='existing WebUI HTTP API; one resident model; no browser history changes',
        complete=False,long_only=args.long_only,runs=[],source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('qwen.c','gpu.cu','native.py','decode.py','config.py','web_worker.py','check_web_context.py','build/libqwen.so')})
    def save():
        tmp=OUT/(prefix+'_report.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(OUT/(prefix+'_report.json'))
    with Monitor(OUT/(prefix+'_resources.jsonl'),pid=args.worker_pid) as monitor:
        try:
            monitor.phase='wait_for_user_response';start=time.monotonic();previous=None
            while True:
                state=status();progress=(state['phase'],state.get('load_progress',0)//10)
                if progress!=previous:print('Waiting for current request:',state,flush=True);previous=progress
                if state['phase']=='error':raise RuntimeError(state.get('error') or 'model startup failed')
                if not state['busy'] and state['model_loaded']:break
                if time.monotonic()-start>3600:raise TimeoutError('current user request did not finish within one hour')
                time.sleep(2)
            report['initial_status']=state;save()
            def request(name,messages,max_tokens=2048,expect_error=False):
                monitor.phase=name+'/preparing';started=time.monotonic()
                data=json.dumps(dict(messages=messages,max_tokens=max_tokens)).encode()
                req=Request(BASE+'/api/chat',data=data,headers={'Content-Type':'application/json','X-Qwen-Client':'webui'})
                final=None;parts=[]
                with urlopen(req,timeout=1800) as response:
                    for line in response:
                        event=json.loads(line)
                        if event['type']=='phase':monitor.phase=name+'/'+event['phase'];print(name,event['phase'],flush=True)
                        elif event['type']=='token':parts.append(event['text'])
                        elif event['type'] in ('done','error'):final=event
                if final is None:raise RuntimeError('stream ended without terminal event')
                if expect_error:return final
                if final['type']=='error':raise RuntimeError(final['message'])
                if final.get('cancelled'):
                    report['interrupted_request']=dict(name=name,event=final,elapsed_seconds=time.monotonic()-started)
                    save();raise RuntimeError('test response was interrupted')
                if ''.join(parts)!=final['text']:raise AssertionError('streamed and final text differ')
                metrics=final['metrics']
                if metrics['context']!=CONTEXT:raise AssertionError('worker has old context')
                if metrics['draft_output_tokens']+metrics['target_output_tokens']!=metrics['output_tokens']:
                    raise AssertionError('output-origin counters do not add up')
                if sum(x['events'] for x in metrics['handoffs_by_layer'])!=metrics['expert_handoff_events']:
                    raise AssertionError('per-layer handoffs do not add up')
                row=dict(name=name,answer=final['text'],metrics=metrics,elapsed_seconds=time.monotonic()-started)
                report['runs'].append(row);save()
                print(json.dumps(dict(name=name,answer=row['answer'],tokens_per_second=metrics['tokens_per_second'],
                    prompt_tokens=metrics['prompt_tokens'],reused=metrics['reused_prefix_tokens']),ensure_ascii=False),flush=True)
                return row
            for old in reference['runs']:
                if args.long_only:break
                if old['depth']!=1:continue
                content=(ROOT.parent/'work/qwen/benchmark_daily_q4'/(old['case']+'.user.txt')).read_text()
                messages=[dict(role='user',content=content)]
                _,_,tokens=encode_chat(messages)
                if tokens!=reference['prompts'][old['case']]['token_ids']:raise AssertionError('reference prompt changed')
                row=request('old8k/'+old['case'],messages,len(old['output_ids'])+1)
                row['matches_previous_answer']=row['answer']==old['answer'];save()
                if not row['matches_previous_answer']:raise AssertionError('8K output text changed: '+old['case'])
            case=(dict(prompt=(OUT/'web_long_context.user.txt').read_text(),exact=previous_long['expected_code'])
                  if args.long_only else long_case(CONTEXT))
            messages=[dict(role='user',content=case['prompt'])]
            _,_,tokens=encode_chat(messages)
            if not CONTEXT-128<=len(tokens)<CONTEXT:raise AssertionError('fixture does not fill context')
            if case['prompt'].count(case['exact'])!=1:raise AssertionError('retrieval code must occur exactly once')
            (OUT/'web_long_context.user.txt').write_text(case['prompt']);report['long_prompt_ids']=tokens
            report['expected_code']=case['exact'];save()
            row=request('24k/retrieval',messages,32)
            report['retrieval_passed']=row['answer'].strip()==case['exact'];save()
            over=request('24k/oversized',[dict(role='user',content=case['prompt']+' testo'*600)],8,True)
            report['oversized_rejected']=over.get('code')=='context_full';save()
            followup=messages+[dict(role='assistant',content=row['answer']),dict(role='user',content='Ripeti soltanto lo stesso codice di verifica.')]
            row=request('24k/prefix_reuse',followup,24)
            report['followup_retrieval_passed']=row['answer'].strip()==case['exact']
            report['long_prefix_reused']=row['metrics']['reused_prefix_tokens']>=CONTEXT-128
            report['final_status']=status()
            report['same_worker_alive']=Path(f'/proc/{args.worker_pid}/cmdline').read_bytes()==command
            report['complete']=all(report[k] for k in ('retrieval_passed','oversized_rejected','followup_retrieval_passed','long_prefix_reused','same_worker_alive'))
            save()
            if not report['complete']:raise AssertionError('24K acceptance checks failed; see report')
        except Exception as exc:
            report['error']=str(exc);save();raise
        finally:
            monitor.sample();report['resources']=monitor.summary();save()
    print('WebUI 24K validation passed.',flush=True)


if __name__=='__main__':main()
