"""Standalone bridge accounting and model-switch lifecycle, without model weights."""
import json
import os
from pathlib import Path
import sys
import time
import unittest
import tempfile
from unittest.mock import patch

import light_worker
import model_registry
import native
from web_server import Engine, validate_request


class Backend:
    model = dict(label='Qwen light test', context=64)
    start_think, end_think = 101, 102
    eos = {103, 104}
    def __init__(self, events, thinking=False):
        self.stream=events;self.thinking=thinking;self.payload=None;self.template_request=None
    def call(self, endpoint, data):
        if endpoint=='/apply-template':
            self.template_request=data
            return dict(prompt='<think>\n' if self.thinking else '<think>\n\n</think>\n\n')
        return dict(tokens=list(range(10)))
    def events(self,data,stopped):
        self.payload=data
        for event in self.stream:
            if stopped():break
            yield event


def chunk(token, total, text):
    return dict(tokens=[token],tokens_predicted=total,content=text,
                timings=dict(prompt_n=6,cache_n=4,prompt_ms=100,predicted_ms=200,
                             predicted_per_second=20))


class LightTests(unittest.TestCase):
    def reply(self, backend, cancel_after=None, budget=20):
        events=[];cancelled=False
        def emit(kind,**data):
            nonlocal cancelled
            events.append(dict(type=kind,**data))
            if kind=='token' and cancel_after is not None:
                cancelled=True
        request=dict(messages=[dict(role='user',content='Ciao')],thinking=backend.thinking,max_tokens=budget)
        with patch.object(light_worker,'emit',side_effect=emit):
            light_worker.generate_reply(backend,request,lambda:cancelled)
        return events

    def test_hidden_reasoning_and_utf8_chunk_token_counts(self):
        # A single UTF-8 chunk represents two sampled tokens. Retokenizing its
        # text or counting chunks would produce the wrong answer-token count.
        backend=Backend([chunk(10,1,'PRIVATE'),chunk(102,2,'</think>'),
                         chunk(11,4,'è'),chunk(103,5,'<|im_end|>'),dict(stop=True,stop_type='eos')],True)
        events=self.reply(backend);done=events[-1];m=done['metrics']
        self.assertEqual(done['text'],'è');self.assertEqual(done['reasoning_content'],'PRIVATE')
        self.assertEqual(''.join(e['text'] for e in events if e['type']=='token'),'è')
        self.assertEqual((m['answer_tokens'],m['reasoning_tokens'],m['output_tokens']),(2,2,4))
        self.assertEqual(m['target_output_tokens'],4);self.assertFalse(m['speculative'])
        self.assertEqual(m['prefill_tokens_per_second'],60);self.assertEqual(m['reused_prefix_tokens'],4)
        self.assertTrue(backend.template_request['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(m['reasoning_complete'])

    def test_non_thinking_counts_exclude_eos(self):
        backend=Backend([chunk(10,1,'42'),chunk(103,2,'<|im_end|>'),dict(stop=True,stop_type='eos')])
        done=self.reply(backend)[-1]
        self.assertEqual(done['metrics']['answer_tokens'],1)
        self.assertEqual(done['metrics']['reasoning_tokens'],0)
        self.assertEqual(done['text'],'42')
        self.assertFalse(backend.template_request['chat_template_kwargs']['enable_thinking'])

    def test_cancel_preserves_delivered_counts_and_next_request(self):
        backend=Backend([chunk(10,1,'prima'),chunk(11,2,' seconda'),dict(stop=True,stop_type='eos')])
        done=self.reply(backend,cancel_after=1)[-1]
        self.assertEqual(done['text'],'prima');self.assertTrue(done['cancelled'])
        self.assertEqual(done['metrics']['output_tokens'],1)
        self.assertEqual(self.reply(backend)[-1]['text'],'prima seconda')

    def test_context_budget_and_unfinished_reasoning(self):
        backend=Backend([chunk(10,1,'PRIVATE'),dict(stop=True,stop_type='limit')],True)
        done=self.reply(backend,budget=60)[-1]
        self.assertEqual(backend.payload['n_predict'],53)
        self.assertEqual(done['text'],'');self.assertFalse(done['metrics']['reasoning_complete'])
        self.assertEqual(done['metrics']['stop_reason'],'context_limit')

    def test_incomplete_stream_is_error(self):
        with self.assertRaisesRegex(RuntimeError,'risultato finale'):
            self.reply(Backend([chunk(10,1,'ciao')]))


class SwitchTests(unittest.TestCase):
    def test_native_and_standalone_share_lock_across_installations(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(native.Native,'_open'), \
             patch.object(native,'GPU_LOCK',Path(tmp)/'gpu.lock'), \
             patch.object(light_worker,'GPU_LOCK',Path(tmp)/'gpu.lock'):
            backend=native.Native()
            try:
                with self.assertRaises(BlockingIOError):
                    light_worker.LightBackend(dict(file='unused.gguf'),lambda:False)
            finally:backend.close()
            native.Native().close()

    def test_failed_native_load_releases_shared_lock(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(native,'GPU_LOCK',Path(tmp)/'gpu.lock'):
            with patch.object(native.Native,'_open',side_effect=ValueError('failed load')):
                with self.assertRaises(ValueError):native.Native()
            with patch.object(native.Native,'_open'):native.Native().close()

    def test_unknown_model_rejected_before_process_start(self):
        with self.assertRaises(ValueError):
            validate_request(dict(model='../../other',messages=[dict(role='user',content='test')]))

    def test_switch_closes_previous_worker_before_starting_next(self):
        fixture=[sys.executable,'-u',str(Path(__file__).with_name('test_webui.py')),'--fixture']
        entries=[dict(id='flash-next',label='Flash test',speculative=True,context=24576,available=True),
                 dict(id='light',label='Light test',speculative=False,context=16384,available=True)]
        with patch.object(model_registry,'models',return_value=entries):
            engine=Engine(fixture)
            try:
                request=dict(model='flash-next',messages=[dict(role='user',content='test')],max_tokens=16)
                self.assertTrue(engine.reserve());list(engine.stream(request));old=engine.process
                request['model']='light';self.assertTrue(engine.reserve());events=list(engine.stream(request))
                self.assertIsNotNone(old.poll());self.assertNotEqual(old.pid,engine.process.pid)
                self.assertFalse(events[-1]['metrics']['speculative'])
                self.assertEqual(engine.status()['context'],16384)
            finally:engine.close()

    def test_close_reaps_backend_child_process_group(self):
        code="import subprocess,json,time; p=subprocess.Popen(['sleep','120']); print(json.dumps({'type':'phase','phase':'loading','child':p.pid}),flush=True); time.sleep(120)"
        engine=Engine([sys.executable,'-u','-c',code])
        engine.start();event=engine.events.get(timeout=10);pid=event['child']
        engine.close()
        state=Path(f'/proc/{pid}/stat')
        if state.exists():self.assertEqual(state.read_text().split()[2],'Z')


if __name__=='__main__':unittest.main()
