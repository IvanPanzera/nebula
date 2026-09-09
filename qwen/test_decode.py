import unittest
from decode import DecodeStats,DraftPolicy,generate


def oracle(prefix):
    return (sum((i+3)*t for i,t in enumerate(prefix))+7*len(prefix))%97+1


class Backend:
    """A causal reference which detects invalid state restoration and positions."""
    def __init__(self,context,miss_positions=(),eos_position=None):
        self.context=context;self.history=[];self.miss=set(miss_positions);self.eos=eos_position
        self.saved=None;self.calls=[];self.guessed=[];self.rows=[]
    @property
    def position(self):return len(self.history)
    def predict(self,prefix):return 0 if len(prefix)==self.eos else oracle(prefix)
    def prefill(self,prompt,mtp=True):self.history=list(prompt);self.rows=[self.predict(self.history)]
    def target(self,tokens):
        if self.position+len(tokens)>self.context:raise AssertionError('context overflow')
        self.calls.append(('target',self.position,list(tokens)));self.rows=[]
        for t in tokens:self.history.append(t);self.rows.append(self.predict(self.history))
    def top1(self):return self.rows
    def checkpoint(self):self.saved=self.history.copy();self.guessed=[]
    def restore(self):self.history=self.saved.copy();self.calls.append(('restore',))
    def draft(self,token,position,target_hidden):
        if target_hidden:self.guessed=self.history.copy()
        if len(self.guessed)!=position:raise AssertionError('MTP position mismatch')
        self.guessed.append(token);prediction=self.predict(self.guessed)
        return prediction%97+1 if position in self.miss else prediction
    def catchup(self,tokens,position):
        if self.history[position:]!=list(tokens):raise AssertionError('MTP catchup includes rejected suffix')
        self.calls.append(('catchup',position,list(tokens)))


class DecodeTests(unittest.TestCase):
    def test_arbitrary_rejections_match_causal_reference(self):
        prompt=[3,7,11]
        for size in (0,1,2,4,5,6,8,11,16):
            for failures in ((),range(200),range(4,200,3),range(5,200,7)):
                b=Backend(128,failures);stats=DecodeStats()
                result=list(generate(b,prompt,max_new_tokens=61,draft_max=size,eos_ids=(0,),stats=stats))
                reference=[]
                for _ in range(61):reference.append(oracle(prompt+reference))
                self.assertEqual(result,reference)
                self.assertEqual(b.history,(prompt+reference)[:b.position])
                self.assertEqual(stats.output_tokens,61)
                self.assertEqual(stats.target_output_tokens+stats.draft_output_tokens,61)
                self.assertLessEqual(stats.max_accepted_draft,size)
                if size==0:self.assertEqual(stats.draft_output_tokens,0)
                if size and failures:self.assertGreater(stats.semantic_misses,0)
    def test_eos_inside_accepted_speculation(self):
        for end in (3,4,5,8,13,48,61):
            b=Backend(64,eos_position=end)
            stats=DecodeStats()
            output=list(generate(b,[1,2,3],max_new_tokens=64,draft_max=16,eos_ids=(0,),stats=stats))
            self.assertEqual(len(output),max(0,end-3));self.assertNotIn(0,output)
            self.assertEqual(stats.target_output_tokens+stats.draft_output_tokens,len(output))

    def test_output_origin_and_chain_count_only_emitted_tokens(self):
        for failures in ((),range(100)):
            stats=DecodeStats()
            output=list(generate(Backend(128,failures),[1,2],max_new_tokens=11,draft_max=4,
                                 adaptive=False,eos_ids=(0,),stats=stats))
            self.assertEqual(stats.target_output_tokens+stats.draft_output_tokens,len(output))
            if failures:
                self.assertEqual(stats.draft_output_tokens,0)
                self.assertEqual(stats.max_accepted_draft,0)
            else:
                self.assertEqual(stats.target_output_tokens,3)
                self.assertEqual(stats.draft_output_tokens,8)
                self.assertEqual(stats.max_accepted_draft,4)
    def test_context_and_output_boundaries(self):
        for context in range(4,38):
            for limit in (0,1,2,3,20):
                b=Backend(context,range(20))
                output=list(generate(b,[1,2,3],max_new_tokens=limit,draft_max=16,eos_ids=(0,)))
                self.assertEqual(len(output),min(limit,context-3));self.assertLessEqual(b.position,context)
        with self.assertRaises(ValueError):list(generate(Backend(3),[1,2,3]))
    def test_complete_blocks_need_no_restore(self):
        b=Backend(128);stats=DecodeStats()
        list(generate(b,[1,2],max_new_tokens=50,draft_max=4,eos_ids=(0,),stats=stats))
        self.assertEqual(stats.semantic_misses,0);self.assertEqual(stats.replay_tokens,0)
        self.assertGreater(stats.full_blocks,0);self.assertEqual(stats.max_draft_used,4)
        self.assertFalse(any(c[0]=='restore' for c in b.calls))

    def test_fixed_depth_and_position_counters(self):
        for depth in (1,2,3,4,8,16):
            stats=DecodeStats()
            list(generate(Backend(128),[1,2],max_new_tokens=50,draft_max=depth,
                          adaptive=False,eos_ids=(0,),stats=stats))
            self.assertEqual(stats.max_draft_used,depth)
            self.assertEqual(sum(stats.proposed_by_position),stats.proposed)
            self.assertEqual(stats.proposed_by_position,stats.accepted_by_position)
            self.assertTrue(all(v==0 for v in stats.proposed_by_position[depth:]))

    def test_default_growth_proposes_and_catches_up_sixteen_real_tokens(self):
        b=Backend(256);stats=DecodeStats()
        output=list(generate(b,[1,2,3],max_new_tokens=100,eos_ids=(0,),stats=stats))
        self.assertEqual([row[0] for row in stats.draft_trace[:7]],[4,4,5,6,8,11,16])
        self.assertEqual([row[3] for row in stats.draft_trace[:7]],[4,5,6,8,11,16,16])
        self.assertEqual(stats.max_draft_used,16)
        self.assertEqual(stats.max_accepted_draft,16)
        self.assertTrue(any(c[0]=='target' and len(c[2])==17 for c in b.calls))
        self.assertTrue(any(c[0]=='catchup' and len(c[2])==17 for c in b.calls))
        self.assertEqual(sum(row[1] for row in stats.draft_trace),stats.proposed)
        self.assertEqual(sum(row[2] for row in stats.draft_trace),stats.accepted)
        self.assertEqual(len(output),stats.draft_output_tokens+stats.target_output_tokens)

    def test_partial_misses_regress_and_reset_streaks(self):
        p=DraftPolicy()
        for _ in range(6):p.update(p.n,p.n)
        self.assertEqual(p.n,16)
        self.assertEqual(p.update(16,8),14)
        self.assertEqual(p.update(14,2),10)
        self.assertEqual(p.update(10,1),4)
        self.assertEqual(p.update(4,4),4)
        self.assertEqual(p.negative,0)
        self.assertEqual(p.update(4,4),5)
        self.assertEqual(p.update(5,1),4)
        self.assertEqual(p.negative,1)
        p=DraftPolicy()
        for _ in range(6):p.update(p.n,p.n)
        self.assertEqual(p.update(16,0),4)
        self.assertEqual(p.positive,0)

    def test_miss_at_each_position_of_sixteen_matches_causal(self):
        prompt=[1,2,3];reference=[]
        for _ in range(90):reference.append(oracle(prompt+reference))
        for offset in range(16):
            b=Backend(128,miss_positions=(len(prompt)+offset,));stats=DecodeStats()
            result=list(generate(b,prompt,max_new_tokens=90,draft_max=16,adaptive=False,eos_ids=(0,),stats=stats))
            self.assertEqual(result,reference)
            self.assertEqual(stats.draft_trace[0][2],offset)
            self.assertEqual(b.history,(prompt+reference)[:b.position])

    def test_policy_restarts_for_each_response_and_counts_clipped_proposals(self):
        b=Backend(256)
        for limit in (100,3,100):
            stats=DecodeStats()
            list(generate(b,[1,2],max_new_tokens=limit,eos_ids=(0,),stats=stats))
            self.assertEqual(stats.draft_trace[0][0],4)
            self.assertEqual(stats.draft_trace[0][1],min(4,limit-1))
        for depth in (-1,17):
            with self.assertRaises(ValueError):list(generate(b,[1,2],draft_max=depth))


if __name__=='__main__':unittest.main(verbosity=2)
