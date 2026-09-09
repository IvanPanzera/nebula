"""Standalone CUDA model, using the pinned llama.cpp server's native API."""
import fcntl
import http.client
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

from model_registry import PROJECT, GPU_LOCK, get_model


def emit(kind, **data):
    print(json.dumps(dict(type=kind, **data), ensure_ascii=False), flush=True)


class LightBackend:
    def __init__(self, model, stopped):
        self.process = None
        self.lock = None
        self.model = model
        self.port = None
        self.lock = GPU_LOCK.open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            binary = PROJECT/'runtime/llama/llama-server'
            if not binary.is_file():
                binary = PROJECT/'work/qwen/upstream/build/bin/llama-server'
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                self.port = probe.getsockname()[1]
            command = [str(binary), '--model', model['file'], '--alias', 'qwen-light',
                       '--host', '127.0.0.1', '--port', str(self.port), '--no-webui',
                       '--ctx-size', str(model['context']), '--parallel', '1',
                       '--batch-size', '2048', '--ubatch-size', '256',
                       '--gpu-layers', 'all', '--override-tensor', 'token_embd.weight=CUDA0',
                       '--fit', 'off', '--load-mode', 'none', '--flash-attn', 'on',
                       '--log-verbosity', '4',
                       '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
                       '--no-context-shift', '--jinja']
            # The parent WebUI owns this process group and kills it as a unit
            # on model changes, including if this Python bridge crashes.
            environment = os.environ.copy()
            # The upstream build embeds its original build directory in RUNPATH.
            # Prefer the shipped libraries so a moved installation is independent.
            environment['LD_LIBRARY_PATH'] = str(binary.parent) + (
                ':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
            self.process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr,
                                            env=environment)
            deadline = time.monotonic() + 300
            while True:
                if stopped():
                    raise InterruptedError('Caricamento interrotto.')
                if self.process.poll() is not None:
                    raise RuntimeError('Il modello light non si è caricato. Consulta work/qwen/webui_engine.log.')
                try:
                    if self.call('/health').get('status') == 'ok':
                        break
                except (OSError, ValueError, http.client.HTTPException):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('Tempo di caricamento del modello light esaurito.')
                time.sleep(.2)
            self.end_think = self.special('</think>')
            self.start_think = self.special('<think>')
            self.eos = {self.special(x) for x in ('<|im_end|>', '<|endoftext|>')}
        except Exception:
            self.close()
            raise

    def call(self, endpoint, data=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            connection.request('GET' if data is None else 'POST', endpoint,
                               body=None if data is None else json.dumps(data).encode(),
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            result = json.load(response)
            if response.status != 200:
                raise ValueError(str(result.get('error', result)))
            return result
        finally:
            connection.close()

    def special(self, text):
        ids = self.call('/tokenize', dict(content=text, parse_special=True))['tokens']
        if len(ids) != 1:
            raise ValueError('Il modello light richiede un tokenizer Qwen con token speciali dedicati.')
        return ids[0]

    def events(self, data, stopped):
        """Read SSE on a helper thread so Stop also interrupts a long prefill."""
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=300)
        events = queue.Queue()
        def read():
            try:
                connection.request('POST', '/completion', body=json.dumps(data).encode(),
                                   headers={'Content-Type': 'application/json'})
                response = connection.getresponse()
                if response.status != 200:
                    raise ValueError(response.read().decode())
                for line in response:
                    if line.startswith(b'data: '):
                        payload = line[6:].strip()
                        if payload == b'[DONE]':
                            break
                        events.put(json.loads(payload))
            except Exception as exc:
                events.put(exc)
            finally:
                events.put(None)
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        try:
            while not stopped():
                try:
                    item = events.get(timeout=.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
                if item.get('stop'):
                    break
        finally:
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()
            thread.join(timeout=2)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.lock is not None:
            self.lock.close()
            self.lock = None


def generate_reply(backend, request, stopped):
    thinking = request.get('thinking', False)
    prompt = backend.call('/apply-template', dict(messages=request['messages'],
        chat_template_kwargs=dict(enable_thinking=thinking, preserve_thinking=True)))['prompt']
    ids = backend.call('/tokenize', dict(content=prompt, add_special=True))['tokens']
    context = backend.model['context']
    if len(ids) >= context - 1:
        raise ValueError(f'La conversazione richiede {len(ids):,} token nel contesto di {context:,}. Accorcia il testo o inizia una nuova chat.')
    budget = min(request['max_tokens'], context-len(ids)-1)
    in_reasoning = prompt.rfind('<think>') > prompt.rfind('</think>')
    if in_reasoning and not thinking:
        raise ValueError('Il template del modello selezionato non ha disattivato Thinking. Occorre verificarne il supporto prima di usare questa modalità.')
    emit('phase', phase='prefill', prompt_tokens=len(ids), model_loaded=True)
    text = reasoning = ''
    answer_tokens = reasoning_tokens = previous = 0
    timings = {}
    final = None
    first = True
    payload = dict(prompt=ids, stream=True, n_predict=budget, cache_prompt=True,
                   temperature=.6 if thinking else .7, top_p=.95 if thinking else .8,
                   top_k=20, min_p=0, special=True, return_tokens=True,
                   timings_per_token=True, stream_options=dict(include_usage=True))
    for event in backend.events(payload, stopped):
        if 'error' in event:
            raise ValueError(str(event['error']))
        timings = event.get('timings', timings)
        if event.get('stop'):
            final = event
            break
        if event.get('prompt_progress') or not event.get('tokens'):
            continue
        count = event['tokens_predicted'] - previous
        previous = event['tokens_predicted']
        token = event['tokens'][-1]
        piece = event.get('content', '')
        # A server chunk can finish a multibyte character and stand for several
        # sampled tokens. Count its cumulative token delta, not SSE chunks or
        # re-tokenized text. Dedicated reasoning markers cannot split UTF-8.
        if token in backend.eos:
            continue
        if first:
            emit('phase', phase='thinking' if in_reasoning else 'generating')
            first = False
        if token == backend.start_think:
            in_reasoning = True
            reasoning_tokens += count
            emit('phase', phase='thinking')
        elif token == backend.end_think:
            reasoning_tokens += count
            in_reasoning = False
            emit('phase', phase='generating')
        elif in_reasoning:
            reasoning += piece
            reasoning_tokens += count
        else:
            text += piece
            answer_tokens += count
            emit('token', text=piece, tokens=answer_tokens+reasoning_tokens)
    if final is None and not stopped():
        raise RuntimeError('Il motore light ha chiuso lo stream prima del risultato finale.')
    total = answer_tokens + reasoning_tokens
    at_limit = bool(final and final.get('stop_type') == 'limit')
    prompt_new = timings.get('prompt_n', 0)
    prompt_seconds = timings.get('prompt_ms', 0)/1000
    decode_seconds = timings.get('predicted_ms', 0)/1000
    metrics = dict(model_id='light', model=backend.model['label'], speculative=False,
        answer_tokens=answer_tokens, output_tokens=total, reasoning_tokens=reasoning_tokens,
        thinking=thinking or reasoning_tokens > 0, reasoning_complete=not in_reasoning,
        prompt_tokens=len(ids), context=context, context_tokens=len(ids)+total,
        prefill_tokens_evaluated=prompt_new, reused_prefix_tokens=timings.get('cache_n', 0),
        prefill_seconds=prompt_seconds, decode_seconds=decode_seconds,
        prefill_tokens_per_second=prompt_new/prompt_seconds if prompt_seconds else 0,
        tokens_per_second=timings.get('predicted_per_second', 0),
        target_output_tokens=total, target_cpu_tokens=0, at_limit=at_limit,
        stop_reason='cancelled' if stopped() else 'context_limit' if at_limit and budget < request['max_tokens']
                    else 'output_limit' if at_limit else 'eos')
    emit('done', text=text, reasoning_content=reasoning, cancelled=stopped(), metrics=metrics)


def serve():
    stopped = False
    backend = None
    def cancel(*_):
        nonlocal stopped
        stopped = True
    def shutdown(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGUSR1, cancel)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        for line in sys.stdin:
            stopped = False
            try:
                request = json.loads(line)
                if backend is None:
                    emit('phase', phase='loading')
                    backend = LightBackend(get_model('light'), lambda: stopped)
                if stopped:
                    emit('done', text='', cancelled=True, metrics=None)
                else:
                    generate_reply(backend, request, lambda: stopped)
            except InterruptedError:
                emit('done', text='', cancelled=True, metrics=None, model_loaded=False)
            except BlockingIOError:
                emit('error', message='Un altro processo Qwen sta usando il modello. Chiudilo e riprova.', model_loaded=False)
            except Exception as exc:
                emit('error', message=str(exc), model_loaded=backend is not None)
    except KeyboardInterrupt:
        pass
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    serve()
