"""Web protocol and model bridge tests; no large model is opened."""
import io
import json
from pathlib import Path
import signal
import sys
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from test_decode import Backend
from config import CONTEXT,CPU_THREADS,RESIDENT_EXPERTS
import web_worker
from web_server import Engine, make_server, validate_request


class ReferenceBackend(Backend):
    cpu_threads=CPU_THREADS
    def __init__(self, miss_positions=range(200)):
        super().__init__(CONTEXT, miss_positions=miss_positions)
        self.evaluated=0;self.resets=0;self.closed=False;self.visits=0
    def prefill(self,prompt,mtp=True):
        self.evaluated+=len(prompt);self.visits+=1;super().prefill(prompt,mtp)
    def target(self,tokens):
        self.evaluated+=len(tokens);self.visits+=1;super().target(tokens)
    def stats(self):return dict(target_tokens=self.evaluated,expert_handoffs=1152,expert_upload_bytes=1152*1024)
    def handoff_profile(self):return dict(layer_calls=[self.visits]*48,layer_handoffs=[self.visits]*48)
    def optimization_stats(self):return dict(retained_tokens=0,prefix_restores=0,graph_launches=0,graph_captures=0,prefix_restore_seconds=0.)
    def storage_stats(self):return dict(ram_experts_per_layer=512,ple_on_ssd=0,transient_peak_bytes=0,expert_read_bytes=0,expert_reads=0,expert_io_seconds=0.,expert_wait_seconds=0.,ple_disk_bytes=0,ple_cache_hits=0,ple_cache_misses=0)
    def cpu_stats(self):return dict(layer_handoffs=self.visits*48,token_layers=self.evaluated*48,missing_selections=self.visits*200,activation_bytes=self.visits*1000,compute_seconds=self.visits*.01,handoff_seconds=self.visits*.011)
    def prune_stats(self):return dict(token_layers=self.evaluated*48,missing_selections=self.visits*200,skipped_selections=self.visits*7,avoided_token_handoffs=self.visits*3,skipped_mass=self.visits*.05)
    def reset(self):self.history=[];self.resets+=1
    def close(self):self.closed=True


class SequenceBackend(ReferenceBackend):
    """Real decoder/cache protocol with a known token sequence, no model load."""
    def __init__(self, tokens):
        super().__init__(miss_positions=())
        self.sequence=tokens;self.base=0
    def prefill(self,prompt,mtp=True):
        self.base=len(prompt)
        super().prefill(prompt,mtp)
    def predict(self,prefix):
        index=len(prefix)-self.base
        return self.sequence[index] if 0<=index<len(self.sequence) else 248046


class WorkerTests(unittest.TestCase):
    def run_worker(self, requests, cancel=False, backend=None):
        result=[];callbacks={};backend=backend or ReferenceBackend()
        def emit(kind,**data):
            result.append(dict(type=kind,**data))
            if cancel and kind=='token' and sum(x['type']=='token' for x in result)==1:
                callbacks[signal.SIGUSR1]()
        with patch.object(web_worker,'emit',side_effect=emit), \
             patch.object(web_worker.signal,'signal',side_effect=lambda sig,fn:callbacks.update({sig:fn})), \
             patch.object(web_worker.sys,'stdin',io.StringIO(''.join(json.dumps(r)+'\n' for r in requests))), \
             patch.object(web_worker,'Native',return_value=backend) as factory:
            web_worker.serve()
        return result,backend,factory.call_count

    def test_persistent_worker_uses_real_decoder_and_official_tokenizer(self):
        req=dict(messages=[dict(role='user',content='What is 2 + 2?')],max_tokens=8)
        events,backend,loads=self.run_worker([req,req])
        self.assertEqual(loads,1)
        done=[e for e in events if e['type']=='done']
        self.assertEqual(len(done),2);self.assertEqual(done[0]['text'],done[1]['text'])
        self.assertTrue(all(e['metrics']['output_tokens']==8 for e in done))
        for event in done:
            metrics=event['metrics']
            self.assertEqual(metrics['draft_output_tokens']+metrics['target_output_tokens'],8)
            self.assertEqual(metrics['target_cpu_tokens'],0)
            self.assertEqual(metrics['target_execution'],'hybrid-layer-cpu')
            self.assertEqual(metrics['target_quantization'],'all-experts-Q4_K-IQ4_NL')
            self.assertEqual(metrics['resident_experts_per_layer'],RESIDENT_EXPERTS)
            self.assertEqual(metrics['cpu_threads'],CPU_THREADS)
            self.assertEqual(metrics['storage']['ram_experts_per_layer'],512)
            self.assertEqual(metrics['storage']['ssd_experts_per_layer'],0)
            self.assertEqual(metrics['storage']['ngram'],'ram')
            self.assertEqual(metrics['storage']['expert_read_bytes'],0)
            self.assertEqual(metrics['cpu_moe_layer_calls'],metrics['expert_handoff_events'])
            self.assertGreater(metrics['cpu_moe_token_layers'],0)
            self.assertGreater(metrics['cpu_activation_mib'],0)
            self.assertGreater(metrics['marginal_handoffs_avoided'],0)
            self.assertEqual(metrics['answer_tokens'],8)
            self.assertEqual(metrics['reasoning_tokens'],0)
            self.assertFalse(metrics['thinking'])
            self.assertGreater(metrics['prefill_tokens_per_second'],0)
            self.assertEqual(metrics['prefill_tokens_evaluated'],metrics['prompt_tokens'])
            self.assertEqual(metrics['expert_handoff_events'],metrics['prefill_handoff_events']+metrics['decode_handoff_events'])
            self.assertEqual(metrics['prefill_handoff_events'],48)
            self.assertEqual(sum(x['events'] for x in metrics['handoffs_by_layer']),metrics['expert_handoff_events'])
            self.assertEqual(metrics['expert_uploads'],0)
            self.assertEqual(metrics['expert_upload_gib'],0)
            self.assertEqual([x['layer'] for x in metrics['handoffs_by_layer']],list(range(48)))
        self.assertTrue(backend.closed)
        self.assertTrue(any(c[0]=='commit_prefix' for c in backend.calls))

    def test_oversized_context_rejected_before_native_load(self):
        events,_,loads=self.run_worker([dict(messages=[dict(role='user',content='testo '*CONTEXT)],max_tokens=8)])
        self.assertEqual(loads,0);self.assertEqual(events[0]['code'],'context_full')

    def test_cancel_resets_cache_keeps_weights_and_next_request_works(self):
        req=dict(messages=[dict(role='user',content='Hello')],max_tokens=8)
        events,backend,loads=self.run_worker([req,req],cancel=True)
        done=[e for e in events if e['type']=='done']
        self.assertTrue(done[0]['cancelled']);self.assertEqual(done[0]['metrics']['output_tokens'],1)
        self.assertFalse(done[1]['cancelled']);self.assertEqual(done[1]['metrics']['output_tokens'],8)
        self.assertEqual(loads,1);self.assertEqual(backend.resets,1)

    def test_context_limit_is_distinct_from_output_limit(self):
        tokenizer,_,_=web_worker.encode_chat([dict(role='user',content='Hello')])
        request=dict(messages=[dict(role='user',content='Hello')],max_tokens=2)
        with patch.object(web_worker,'encode_chat',return_value=(tokenizer,'',[1]*(CONTEXT-1))):
            events,_,_=self.run_worker([request])
        self.assertEqual(events[-1]['metrics']['stop_reason'],'context_limit')
        self.assertEqual(events[-1]['metrics']['output_tokens'],1)
        events,_,_=self.run_worker([request])
        self.assertEqual(events[-1]['metrics']['stop_reason'],'output_limit')
        self.assertEqual(events[-1]['metrics']['output_tokens'],2)

    def test_web_default_adapts_to_sixteen_despite_resolved_handoffs(self):
        req=dict(messages=[dict(role='user',content='Count')],max_tokens=100)
        events,_,_=self.run_worker([req],backend=ReferenceBackend(miss_positions=()))
        metrics=events[-1]['metrics']
        self.assertEqual(metrics['draft_policy'],'adaptive-time')
        self.assertGreater(metrics['max_draft_used'],4)
        self.assertLessEqual(metrics['max_draft_used'],16)
        self.assertEqual(metrics['draft_trace'][0][0],4)
        self.assertGreater(metrics['expert_handoff_events'],0)

    def test_thinking_is_hidden_streamed_answer_and_counts_are_exact(self):
        messages=[dict(role='user',content='What is six times seven?')]
        tok,off,_=web_worker.encode_chat(messages)
        _,on,_=web_worker.encode_chat(messages,'low')
        self.assertTrue(off.endswith('<think>\n\n</think>\n\n'))
        self.assertTrue(on.endswith('<think>\n'))
        self.assertIn('Reasoning effort is set to low',on)
        hidden=tok.encode('PRIVATE_REASONING_TEST',add_special_tokens=False).ids
        answer=tok.encode('42.',add_special_tokens=False).ids
        sequence=hidden+[tok.token_to_id('</think>')]+answer
        req=dict(messages=messages,max_tokens=64,thinking=True)
        events,_,_=self.run_worker([req],backend=SequenceBackend(sequence))
        done=events[-1];metrics=done['metrics']
        self.assertEqual(done['text'],'42.')
        self.assertEqual(''.join(e['text'] for e in events if e['type']=='token'),'42.')
        self.assertEqual(done['reasoning_content'],'PRIVATE_REASONING_TEST')
        self.assertEqual(metrics['output_tokens'],len(sequence))
        self.assertEqual(metrics['answer_tokens'],len(answer))
        self.assertEqual(metrics['reasoning_tokens'],len(hidden)+1)
        self.assertEqual(metrics['answer_tokens']+metrics['reasoning_tokens'],metrics['output_tokens'])
        self.assertEqual(metrics['draft_output_tokens']+metrics['target_output_tokens'],len(sequence))
        self.assertTrue(metrics['reasoning_complete'])
        history=messages+[dict(role='assistant',content=done['text'],reasoning_content=done['reasoning_content']),dict(role='user',content='Can you confirm?')]
        clean=validate_request(dict(messages=history,thinking=True))
        _,rendered,_=web_worker.encode_chat(clean['messages'],'low')
        self.assertIn('<think>\nPRIVATE_REASONING_TEST\n</think>\n\n42.',rendered)

    def test_reasoning_exhaustion_and_cancellation_never_become_visible_answer(self):
        tok,_,_=web_worker.encode_chat([dict(role='user',content='Hello')])
        sequence=tok.encode('PRIVATE reasoning that cannot fit in the budget',add_special_tokens=False).ids
        req=dict(messages=[dict(role='user',content='Hello')],max_tokens=2,thinking=True)
        for cancel in (False,True):
            events,_,_=self.run_worker([req],cancel=cancel,backend=SequenceBackend(sequence))
            done=events[-1];metrics=done['metrics']
            self.assertEqual(done['text'],'')
            self.assertTrue(all(e['text']=='' for e in events if e['type']=='token'))
            self.assertFalse(metrics['reasoning_complete'])
            self.assertEqual(metrics['answer_tokens'],0)
            self.assertEqual(metrics['reasoning_tokens'],1 if cancel else 2)
            self.assertEqual(metrics['stop_reason'],'cancelled' if cancel else 'output_limit')


def fixture():
    """Only the test harness starts this synthetic child, never the real UI."""
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGUSR1,stop)
    def emit(kind,**data):print(json.dumps(dict(type=kind,**data)),flush=True)
    for line in sys.stdin:
        req=json.loads(line);stopped=False
        emit('phase',phase='prefill',model_loaded=True);time.sleep(.12)
        emit('phase',phase='thinking' if req.get('thinking') else 'generating')
        text='Simulated response for interface testing.\n\n**A Python example:**\n\n```python\ndef unique(items):\n    return list(dict.fromkeys(items))\n```\n\n- Preserves the original order.\n- Removes duplicates.\n\n<script>alert("test")</script>'
        shown=''
        slow='slow' in req['messages'][-1]['content']
        for part in text.splitlines(keepends=True):
            if stopped:break
            shown+=part;emit('token',text=part,tokens=len(shown));time.sleep(.4 if slow else .015)
        metrics=dict(output_tokens=53,prefill_seconds=.12,
             tokens_per_second=4.29,reused_prefix_tokens=35,context_tokens=189,at_limit=False,
             prefill_tokens_per_second=841.7,draft_output_tokens=22,target_output_tokens=31,
             target_cpu_tokens=0,max_accepted_draft=16,max_draft_used=16,
             answer_tokens=53,thinking=req.get('thinking',False),reasoning_tokens=0,reasoning_complete=True,
             draft_policy='adaptive',draft_min=4,draft_max=16,
             draft_trace=[[4,4,4,4],[4,4,4,5],[5,5,5,6],[6,6,6,8],[8,8,8,11],[11,11,11,16],[16,16,8,14]],
             expert_handoff_events=192,prefill_handoff_events=48,decode_handoff_events=144,layer_executions=240,
             expert_uploads=580,expert_upload_gib=1.72,
             handoffs_by_layer=[dict(layer=i+1,events=4,prefill=1,decode=3,calls=5) for i in range(48)])
        if req.get('model')=='light':
            metrics={k:v for k,v in metrics.items() if not any(x in k for x in ('draft','expert','handoff','layer_'))}
            metrics.update(target_output_tokens=53,speculative=False)
        emit('done',text=shown,cancelled=stopped,metrics=metrics)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=Engine([sys.executable,'-u',str(Path(__file__).resolve()),'--fixture'])
        cls.engine.external_busy=lambda:False
        cls.server=make_server(0,cls.engine)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.engine.close();cls.server.server_close();cls.thread.join()
    def post(self,path,data=None,headers=None):
        return urlopen(Request(self.url+path,data=json.dumps(data or {}).encode(),
            headers=headers or {'X-Qwen-Client':'webui','Content-Type':'application/json'}),timeout=8)
    def test_assets_and_origin_restrictions(self):
        for path in ('/','/style.css','/app.js','/nebula-logo.svg','/api/status'):
            with urlopen(self.url+path) as response:
                self.assertEqual(response.status,200);self.assertIn("frame-ancestors 'none'",response.headers['Content-Security-Policy'])
        for headers in ({'X-Qwen-Client':'webui','Origin':'https://unrelated.example'}, {'Content-Type':'application/json'}):
            with self.assertRaises(HTTPError) as error:self.post('/api/stop',headers=headers)
            self.assertEqual(error.exception.code,403)
            error.exception.close()
        with self.assertRaises(HTTPError) as error:urlopen(self.url+'/../web_worker.py')
        self.assertEqual(error.exception.code,404)
        error.exception.close()
    def test_webui_only_exposes_and_accepts_flagship(self):
        with urlopen(self.url+'/api/status') as response:
            status=json.load(response)
        self.assertEqual([m['id'] for m in status['models']],['flash-next'])
        for model in ('light','unknown',None,[],{}):
            with self.assertRaises(HTTPError) as error:
                self.post('/api/chat',dict(model=model,messages=[dict(role='user',content='Test')]))
            self.assertEqual(error.exception.code,400)
            self.assertIn('only supports Flash-Next',error.exception.read().decode())
            error.exception.close()
        with urlopen(self.url+'/nebula-logo.svg') as response:
            self.assertEqual(response.headers.get_content_type(),'image/svg+xml')
            self.assertIn(b'NEBULA',response.read())
    def test_invalid_history_and_limits(self):
        for body in ({'messages':[]},{'messages':[{'role':'assistant','content':'test'}]},
            {'messages':[{'role':'user','content':'test'}],'max_tokens':9000}):
            with self.assertRaises(HTTPError) as error:self.post('/api/chat',body)
            self.assertEqual(error.exception.code,400)
            error.exception.close()
        for thinking in ('on',1,None):
            with self.assertRaises(ValueError):validate_request(dict(messages=[dict(role='user',content='Hello')],thinking=thinking))
        self.assertFalse(validate_request(dict(messages=[dict(role='user',content='Hello')]))['thinking'])

    def test_stop_while_thinking_and_followup(self):
        request=dict(messages=[dict(role='user',content='slow response')],thinking=True)
        with self.post('/api/chat',request) as response:
            while json.loads(response.readline()).get('phase')!='thinking':pass
            with self.post('/api/stop'):pass
            events=[json.loads(line) for line in response]
        self.assertTrue(events[-1]['cancelled'])
        with self.post('/api/chat',dict(messages=[dict(role='user',content='Hello')],thinking=False)) as response:
            events=[json.loads(line) for line in response]
        self.assertFalse(events[-1]['cancelled'])
    def test_streaming_serialization_stop_and_reuse(self):
        request=dict(messages=[dict(role='user',content='slow response')],max_tokens=64)
        with self.post('/api/chat',request) as response:
            first=json.loads(response.readline());self.assertEqual(first['type'],'phase')
            while json.loads(response.readline()).get('phase')!='generating':pass
            with self.assertRaises(HTTPError) as error:self.post('/api/chat',request)
            self.assertEqual(error.exception.code,409)
            error.exception.close()
            with self.post('/api/stop') as stopped:self.assertEqual(stopped.status,200)
            events=[json.loads(line) for line in response]
            self.assertTrue(events[-1]['cancelled'])
        pid=self.engine.process.pid
        with self.post('/api/chat',dict(messages=[dict(role='user',content='Second question')])) as response:
            events=[json.loads(line) for line in response]
        self.assertEqual(events[-1]['type'],'done');self.assertFalse(events[-1]['cancelled'])
        self.assertIn('```python',events[-1]['text']);self.assertEqual(self.engine.process.pid,pid)

    def test_disconnect_drains_worker_and_releases_gate(self):
        request=dict(messages=[dict(role='user',content='slow response')])
        response=self.post('/api/chat',request)
        while json.loads(response.readline()).get('phase')!='generating':pass
        response.close()
        deadline=time.monotonic()+7
        while self.engine.gate.locked() and time.monotonic()<deadline:time.sleep(.05)
        self.assertFalse(self.engine.gate.locked())
        self.assertIsNone(self.engine.process.poll())

    def test_stop_before_first_worker_phase_is_deferred(self):
        engine=Engine([sys.executable,'-u',str(Path(__file__).resolve()),'--fixture'])
        try:
            self.assertTrue(engine.reserve())
            stream=engine.stream(dict(messages=[dict(role='user',content='slow response')]))
            self.assertEqual(next(stream)['phase'],'preparing')
            engine.stop()
            events=list(stream)
            self.assertTrue(events[-1]['cancelled']);self.assertIsNone(engine.process.poll())
        finally:engine.close()


if __name__=='__main__':
    if '--fixture' in sys.argv:fixture()
    else:unittest.main(verbosity=2)
