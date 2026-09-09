#!/usr/bin/env python3
"""Local Qwen chat/benchmark entry point. Requires verified model assets."""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from decode import DecodeStats,generate
from config import CONTEXT, DRAFT_MAX
from native import Native

ROOT=Path(__file__).resolve().parent.parent
TOKENIZER=ROOT/'work/research/qwen38/Qwen3.8-Flash-Next'


@lru_cache(maxsize=1)
def chat_assets():
    manifest=json.loads((ROOT/'work/qwen/tokenizer_manifest.json').read_text())
    if manifest['repo']!='Qwen/Qwen3.8-Flash-Next' or manifest['revision']!='de4b8e4d43b917e7706784d8bb445c9af86a3540':
        raise ValueError('unsupported official tokenizer revision')
    files={entry['name']:entry for entry in manifest['files']}
    for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        data=(TOKENIZER/name).read_bytes();entry=files[name]
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('official tokenizer asset changed: '+name)
    def reject(message):raise ValueError(message)
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    env.globals['raise_exception']=reject
    template=env.from_string((TOKENIZER/'chat_template.jinja').read_text())
    return Tokenizer.from_file(str(TOKENIZER/'tokenizer.json')),template


def encode_chat(messages,thinking='off'):
    tok,template=chat_assets()
    rendered=template.render(messages=messages,add_generation_prompt=True,
        enable_thinking=thinking!='off',reasoning_effort='low' if thinking=='off' else thinking,
        preserve_thinking=True,tools=None)
    return tok,rendered,tok.encode(rendered,add_special_tokens=False).ids


def read_turn(messages,context,thinking,backend=None):
    """Reject an oversized turn before changing history or touching model state."""
    while True:
        try:prompt=input('Tu: ')
        except (EOFError,KeyboardInterrupt):return None
        if prompt.strip() in ('/exit','/quit'):return None
        if prompt.strip()=='/reset':
            messages.clear()
            if backend is not None:backend.reset()
            print('Conversazione azzerata.'+(' Il modello rimane caricato.' if backend is not None else ''),file=sys.stderr)
            continue
        if not prompt.strip():continue
        proposed=messages+[dict(role='user',content=prompt)]
        tokenizer,rendered,tokens=encode_chat(proposed,thinking)
        if len(tokens)>=context:
            print(f'Il prompt richiede {len(tokens)} token; il contesto totale è {context}. '
                  'Accorcia il messaggio oppure usa /reset per iniziare una nuova conversazione.',file=sys.stderr)
            continue
        messages[:]=proposed
        return tokenizer,rendered,tokens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt');p.add_argument('--prompt-file');p.add_argument('--context',type=int,default=CONTEXT)
    p.add_argument('--interactive',action='store_true',help='continue the conversation after an initial prompt or file')
    p.add_argument('--resident-experts',type=int,default=24);p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--draft-max',type=int,default=DRAFT_MAX,help='adaptive draft ceiling (default: 16, starts at 4); 0 disables MTP')
    p.add_argument('--fixed-draft',action='store_true',help='diagnostic: keep N fixed at --draft-max')
    p.add_argument('--prefill-batch',type=int,default=2048)
    p.add_argument('--thinking',choices=['off','low','medium','xhigh'],default='off')
    p.add_argument('--report');p.add_argument('--tokenize-only',action='store_true')
    args=p.parse_args()
    if not 256<=args.context<=CONTEXT:p.error(f'context must be 256..{CONTEXT}')
    if not 10<=args.resident_experts<=128:p.error('resident experts must be 10..128')
    if not 1<=args.prefill_batch<=2048:p.error('prefill batch must be 1..2048')
    if not 0<=args.draft_max<=DRAFT_MAX:p.error(f'draft max must be 0..{DRAFT_MAX}')
    if args.max_tokens<0:p.error('max tokens must be nonnegative')
    prompt=Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    messages=[];interactive=prompt is None or args.interactive
    if prompt is None:
        turn=read_turn(messages,args.context,args.thinking)
        if turn is None:return
        tokenizer,rendered,tokens=turn
    else:
        messages.append(dict(role='user',content=prompt))
        tokenizer,rendered,tokens=encode_chat(messages,args.thinking)
    if args.tokenize_only:
        print(json.dumps(dict(rendered=rendered,token_ids=tokens,length=len(tokens)),ensure_ascii=False,indent=2));return
    if len(tokens)>=args.context:p.error(f'prompt has {len(tokens)} tokens and leaves no room in context {args.context}')
    with Native(context=args.context,resident=args.resident_experts,prefill_batch=args.prefill_batch) as backend:
        while True:
            generated=[];shown='';stats=DecodeStats();prefill_tokens_evaluated=None
            before=backend.stats()
            for token in generate(backend,tokens,max_new_tokens=args.max_tokens,draft_max=args.draft_max,
                                  adaptive=not args.fixed_draft,stats=stats):
                if prefill_tokens_evaluated is None:
                    prefill_tokens_evaluated=backend.stats()['target_tokens']-before['target_tokens']
                generated.append(token)
                text=tokenizer.decode(generated,skip_special_tokens=False).rstrip('\ufffd')
                if not text.startswith(shown):raise RuntimeError('non-monotonic tokenizer streaming output')
                print(text[len(shown):],end='',flush=True);shown=text
            answer=tokenizer.decode(generated,skip_special_tokens=False)
            print(answer[len(shown):])
            output_limit=min(args.max_tokens,args.context-len(tokens))
            at_limit=output_limit>0 and stats.output_tokens>=output_limit
            report=dict(config=vars(args),prompt_tokens=len(tokens),output_ids=generated,answer=answer,
                        decode=asdict(stats),native_before=before,native_after=backend.stats(),sampler='greedy',
                        at_output_limit=at_limit,
                        prefill_tokens_evaluated=prefill_tokens_evaluated,
                        reused_prefix_tokens=len(tokens)-prefill_tokens_evaluated if prefill_tokens_evaluated is not None else None,
                        target_experts='all top-10, dynamic cache handoff; no expert omission')
            if args.report:
                report_path=Path(args.report);report_path.parent.mkdir(parents=True,exist_ok=True)
                report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            rate=max(stats.output_tokens-1,0)/stats.decode_seconds if stats.decode_seconds else 0.0
            print(f"[{stats.output_tokens} token, {rate:.2f} token/s, {stats.prefill_seconds:.2f}s prefill, "
                  f"draft {stats.accepted}/{stats.proposed}, miss {stats.semantic_misses}, "
                  f"N max proposto {stats.max_draft_used}]",file=sys.stderr)
            if at_limit:
                kind='della risposta' if args.max_tokens<=args.context-len(tokens) else 'del contesto'
                print(f'Raggiunto il limite {kind}.',file=sys.stderr)
            if not interactive:return
            if args.thinking!='off' and '</think>' in answer:
                reasoning,content=answer.split('</think>',1)
                messages.append({'role':'assistant','content':content.strip(),'reasoning_content':reasoning.strip()})
            else:
                messages.append({'role':'assistant','content':answer})
            turn=read_turn(messages,args.context,args.thinking,backend)
            if turn is None:return
            tokenizer,rendered,tokens=turn


if __name__=='__main__':main()
