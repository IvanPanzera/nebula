"""Persistent model process. CUDA calls stay on one thread, as in chat.py."""
from dataclasses import asdict
import json
import signal
import sys

from chat import encode_chat
from decode import DecodeStats, generate
from native import Native
from config import CONTEXT, DRAFT_MAX, HANDOFF_MODE, RESIDENT_EXPERTS, MISSING_MASS_LIMIT
from verification import DEFAULT_LEVEL, settings


def emit(kind, **data):
    print(json.dumps(dict(type=kind, **data), ensure_ascii=False), flush=True)


def handoff_metrics(before, prefill, after, native_before, native_after):
    layers=[]
    for i in range(48):
        events=after['layer_handoffs'][i]-before['layer_handoffs'][i]
        prompt_events=prefill['layer_handoffs'][i]-before['layer_handoffs'][i]
        layers.append(dict(layer=i, events=events, prefill=prompt_events, decode=events-prompt_events,
                           calls=after['layer_calls'][i]-before['layer_calls'][i]))
    return dict(expert_handoff_events=sum(x['events'] for x in layers),
                prefill_handoff_events=sum(x['prefill'] for x in layers),
                decode_handoff_events=sum(x['decode'] for x in layers),
                layer_executions=sum(x['calls'] for x in layers), handoffs_by_layer=layers,
                expert_uploads=native_after['expert_handoffs']-native_before['expert_handoffs'],
                expert_upload_gib=(native_after['expert_upload_bytes']-native_before['expert_upload_bytes'])/2**30)


def serve():
    backend = None
    stopped = False

    def cancel(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGUSR1, cancel)
    try:
        for line in sys.stdin:
            stopped = False
            iterator = None
            try:
                request = json.loads(line)
                verification_level=settings(request.get('verification_level',DEFAULT_LEVEL))['level']
                thinking = request.get('thinking', False)
                tokenizer, _, prompt = encode_chat(request['messages'], 'low' if thinking else 'off')
                end_think = tokenizer.token_to_id('</think>')
                if len(prompt) >= CONTEXT:
                    emit('error', message=f'This conversation uses {len(prompt):,} tokens and leaves no room for a response in the {CONTEXT:,}-token context. Shorten the message or start a new chat.', code='context_full')
                    continue
                if backend is None:
                    emit('phase', phase='loading')
                    backend = Native()
                if stopped:
                    emit('done', text='', cancelled=True, metrics=None)
                    continue
                emit('phase', phase='prefill', prompt_tokens=len(prompt), model_loaded=True)
                stats = DecodeStats()
                native_before = backend.stats()
                cpu_before, prune_before = backend.cpu_stats(), backend.prune_stats()
                optimization_before = backend.optimization_stats()
                storage_before = backend.storage_stats()
                before = native_before['target_tokens']
                profile_before = backend.handoff_profile()
                profile_prefill = None
                prefill = None
                ids, answer_ids, reasoning_ids, shown = [], [], [], ''
                in_reasoning = thinking
                reasoning_tokens = 0
                iterator = generate(backend, prompt, max_new_tokens=request['max_tokens'],
                                    draft_max=DRAFT_MAX, adaptive=True, stats=stats, verification_level=verification_level)
                while not stopped:
                    try:
                        token = next(iterator)
                    except StopIteration:
                        break
                    if prefill is None:
                        prefill = backend.stats()['target_tokens'] - before
                        profile_prefill = backend.handoff_profile()
                        emit('phase', phase='thinking' if in_reasoning else 'generating')
                    ids.append(token)
                    if in_reasoning:
                        # The opening <think> is already in the official prompt.
                        # Keep reasoning for the next turn, never stream it as answer text.
                        reasoning_tokens += 1
                        if token == end_think:
                            in_reasoning = False
                            emit('phase', phase='generating')
                        else:
                            reasoning_ids.append(token)
                    else:
                        answer_ids.append(token)
                    text = tokenizer.decode(answer_ids, skip_special_tokens=False).lstrip('\n').rstrip('\ufffd')
                    if not text.startswith(shown):
                        raise RuntimeError('non-monotonic tokenizer streaming output')
                    emit('token', text=text[len(shown):], tokens=len(ids))
                    shown = text
                iterator.close()
                iterator = None
                if prefill is None:
                    prefill = backend.stats()['target_tokens'] - before
                    profile_prefill = backend.handoff_profile()
                profile_after = backend.handoff_profile()
                native_after = backend.stats()
                cpu_after, prune_after = backend.cpu_stats(), backend.prune_stats()
                optimization_after = backend.optimization_stats()
                storage_after = backend.storage_stats()
                text = tokenizer.decode(answer_ids, skip_special_tokens=False).lstrip('\n')
                if stopped:
                    # A generator can stop between accepted tokens of a block.
                    # Drop KV, keeping weights; the next turn rebuilds exact state.
                    backend.reset()
                limit = min(request['max_tokens'], CONTEXT-len(prompt))
                metrics = dict(asdict(stats), output_tokens=len(ids), prompt_tokens=len(prompt),
                    answer_tokens=len(answer_ids), reasoning_tokens=reasoning_tokens,
                    thinking=thinking, reasoning_complete=not in_reasoning,
                    context_tokens=len(prompt)+len(ids), context=CONTEXT,
                    reused_prefix_tokens=len(prompt)-prefill if prefill is not None else 0,
                    prefill_tokens_evaluated=prefill,
                    prefill_tokens_per_second=prefill/stats.prefill_seconds if stats.prefill_seconds else 0,
                    tokens_per_second=max(len(ids)-1, 0)/stats.decode_seconds if stats.decode_seconds else 0,
                    target_cpu_tokens=0,
                    target_execution=HANDOFF_MODE, resident_experts_per_layer=RESIDENT_EXPERTS,
                    target_quantization='all-experts-Q4_K-IQ4_NL',cpu_threads=backend.cpu_threads,
                    missing_mass_limit=MISSING_MASS_LIMIT,
                    cpu_moe_layer_calls=cpu_after['layer_handoffs']-cpu_before['layer_handoffs'],
                    cpu_moe_token_layers=cpu_after['token_layers']-cpu_before['token_layers'],
                    cpu_moe_seconds=cpu_after['compute_seconds']-cpu_before['compute_seconds'],
                    cpu_activation_mib=(cpu_after['activation_bytes']-cpu_before['activation_bytes'])/2**20,
                    ple_host_gib=native_after.get('ple_host_bytes',0)/2**30,
                    ple_lookup_seconds=native_after.get('ple_lookup_seconds',0)-native_before.get('ple_lookup_seconds',0),
                    ple_transfer_bytes=native_after.get('ple_read_bytes',0)-native_before.get('ple_read_bytes',0),
                    marginal_experts_skipped=prune_after['skipped_selections']-prune_before['skipped_selections'],
                    marginal_handoffs_avoided=prune_after['avoided_token_handoffs']-prune_before['avoided_token_handoffs'],
                    at_limit=not stopped and len(ids) >= limit)
                metrics.update({k:optimization_after[k]-optimization_before[k] for k in optimization_after})
                metrics['storage'] = dict(ram_experts_per_layer=storage_after['ram_experts_per_layer'],
                    ssd_experts_per_layer=512-storage_after['ram_experts_per_layer'],
                    ngram='ssd' if storage_after['ple_on_ssd'] else 'ram',
                    transient_peak_mib=storage_after['transient_peak_bytes']/2**20,
                    **{k:storage_after[k]-storage_before[k] for k in ('expert_read_bytes','expert_reads',
                       'expert_io_seconds','expert_wait_seconds','ple_disk_bytes','ple_cache_hits','ple_cache_misses')})
                metrics.update(handoff_metrics(profile_before,profile_prefill,profile_after,native_before,native_after))
                metrics['stop_reason'] = ('cancelled' if stopped else
                    'context_limit' if metrics['at_limit'] and limit<request['max_tokens'] else
                    'output_limit' if metrics['at_limit'] else 'eos')
                emit('done', text=text, cancelled=stopped, metrics=metrics,
                     reasoning_content=tokenizer.decode(reasoning_ids, skip_special_tokens=False))
            except Exception as exc:
                if iterator is not None:
                    iterator.close()
                if backend is not None:
                    try:
                        backend.reset()
                    except Exception:
                        backend.close()
                        backend = None
                message = str(exc)
                if 'holds the model lock' in message:
                    message = 'The engine is already running in another terminal. Close that chat with /exit and try again.'
                emit('error', message=message, model_loaded=backend is not None)
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    serve()
