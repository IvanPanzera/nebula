"""Speculative decoding with compact recurrent prefix restoration.

Level 3 is exact greedy verification. Levels 1--2 intentionally relax it;
none claims equivalence to stochastic speculative sampling. Expert handoff
is resolved by the CPU/GPU target backend, independently of verification.
"""
from dataclasses import dataclass, field
import time
from config import DRAFT_MIN, DRAFT_MAX
from verification import SCALE, DEFAULT_LEVEL, settings, accepted_prefix


class DraftPolicy:
    """User policy: successes +0,+1,+1,+2,+3,+5; partial misses -2,-4,-6.

    A first-token miss returns to the minimum. Handoffs are completed inside
    the backend: only verified token agreement changes these streaks.
    Smaller explicit maxima remain available for diagnostic comparisons.
    """
    def __init__(self, maximum=DRAFT_MAX, adaptive=True):
        self.minimum = min(DRAFT_MIN, maximum)
        self.maximum = maximum
        self.adaptive = adaptive
        self.n = self.minimum if adaptive else maximum
        self.positive = self.negative = 0
        self.samples = 0
        self.costs = {name: {} for name in ('target', 'catchup')}
        self.draft_cost = self.checkpoint_cost = self.restore_cost = 0.0
        self.reached = [2.0]*maximum
        self.matched = [1.0]*maximum
        self.timing_active = False
        self.predicted_tps = 0.0

    @staticmethod
    def _cost(points, size):
        """Interpolate measured shapes; use a nonnegative fit outside them."""
        keys=sorted(points)
        if not keys:return 0.0
        if size in points:return points[size]
        for a,b in zip(keys,keys[1:]):
            if a<size<b:return points[a]+(points[b]-points[a])*(size-a)/(b-a)
        if len(keys)==1:return points[keys[0]]*size/keys[0]
        # An outlier must not predict negative compute or free larger batches.
        sx=sum(keys)/len(keys);sy=sum(points.values())/len(keys)
        slope=max(0.0,sum((k-sx)*(points[k]-sy) for k in keys)/sum((k-sx)**2 for k in keys))
        return max(min(points.values()),max(0.0,sy-slope*sx)+slope*size)

    def observe(self, proposed, accepted, timing):
        """Estimate useful tokens per second, including drafting and recovery.

        Proposals after the first rejection are censored, not failures: their
        conditioning prefix was wrong. A small prior regularizes deep positions.
        Shape costs and acceptance observations decay as the response changes.
        """
        self.samples+=1
        for j in range(self.maximum):
            self.reached[j]=2+(self.reached[j]-2)*.96
            self.matched[j]=1+(self.matched[j]-1)*.96
        for j in range(min(proposed,accepted+1)):
            self.reached[j]+=1;self.matched[j]+=j<accepted
        for name,size in (('target',proposed+1),('catchup',accepted+1)):
            old=self.costs[name].get(size,timing[name])
            self.costs[name][size]=.75*old+.25*timing[name]
        for attr,value in (('draft_cost',timing['draft']/proposed),
                           ('checkpoint_cost',timing['checkpoint'])):
            old=getattr(self,attr);setattr(self,attr,.8*old+.2*value if old else value)
        if accepted<proposed:
            self.restore_cost=.8*self.restore_cost+.2*timing['restore'] if self.restore_cost else timing['restore']

    def timed_choice(self, proposed, accepted, timing):
        suggested=self.update(proposed,accepted)
        if not self.adaptive or proposed<1:return suggested
        self.observe(proposed,accepted,timing)
        # Two bounded probes obtain actual batch costs even when misses keep
        # resetting the acceptance-only policy to four. Clipped EOS/limit blocks
        # are still measured at their real shape, never at the requested size.
        if self.samples in (3,6) and self.maximum>self.minimum:
            self.n=min(self.maximum,self.minimum+(2 if self.samples==3 else 4))
        elif self.samples>=8 and len(self.costs['target'])>=2:
            rates={};survival=1.;useful=1.
            for n in range(1,self.maximum+1):
                survival*=self.matched[n-1]/self.reached[n-1];useful+=survival
                cost=(self.checkpoint_cost+n*self.draft_cost+self._cost(self.costs['target'],n+1)
                      +self._cost(self.costs['catchup'],useful)+(1-survival)*self.restore_cost)
                if n>=self.minimum:rates[n]=useful/max(cost,1e-9)
            best=max(rates,key=rates.get)
            # Hysteresis and bounded movement avoid chasing noisy timings.
            current=max(self.minimum,min(self.maximum,suggested))
            if rates[best]>rates[current]*1.05:current=max(current-2,min(current+2,best))
            self.n=current;self.predicted_tps=rates[current];self.timing_active=True
            # Occasionally measure a nearby unexplored width; never a long
            # unbounded probe. This prevents a self-confirming fixed-N policy.
            if self.samples%16==0:
                probes=[n for n in (current+1,current-1) if self.minimum<=n<=self.maximum and n+1 not in self.costs['target']]
                if probes:self.n=probes[0]
        return self.n

    def update(self, proposed, accepted):
        if not self.adaptive or not proposed:
            return self.n
        if accepted == proposed:
            self.n += (0, 1, 1, 2, 3, 5)[min(self.positive, 5)]
            self.positive = min(self.positive+1, 6)
            self.negative = 0
        else:
            self.positive = 0
            if accepted == 0:
                self.n = self.minimum
                self.negative = 1
            else:
                self.n -= (2, 4, 6)[min(self.negative, 2)]
                self.negative = min(self.negative+1, 3)
        self.n = max(self.minimum, min(self.maximum, self.n))
        return self.n


@dataclass
class DecodeStats:
    verification_scale: int = SCALE
    verification_level: int = DEFAULT_LEVEL
    verification_top_k: int | None = 1
    verification_min_ratio: float | None = 1.0
    verification_max_relaxed_per_block: int | None = 0
    relaxed_accepted: int = 0
    relaxed_output_tokens: int = 0
    # [zero-based output position, emitted draft ID, target greedy ID].
    relaxed_trace: list = field(default_factory=list)
    proposed: int = 0
    accepted: int = 0
    semantic_misses: int = 0
    replay_tokens: int = 0
    output_tokens: int = 0
    target_output_tokens: int = 0
    draft_output_tokens: int = 0
    max_accepted_draft: int = 0
    full_blocks: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    max_draft_used: int = 0
    draft_policy: str = 'disabled'
    draft_min: int = 0
    draft_max: int = 0
    # Compact per-block records: [requested N, actual proposals, accepted, next N].
    # These are observations, including EOS/context clipping, not scheduled N.
    draft_trace: list = field(default_factory=list)
    timing_trace: list = field(default_factory=list)
    retained_prefix_tokens: int = 0
    prefix_restore_seconds: float = 0.0
    proposed_by_position: list = field(default_factory=lambda: [0]*DRAFT_MAX)
    accepted_by_position: list = field(default_factory=lambda: [0]*DRAFT_MAX)


def generate(backend, prompt, *, max_new_tokens=512, draft_max=DRAFT_MAX,
             adaptive=True, eos_ids=(248044, 248046), stats=None, verification_level=DEFAULT_LEVEL):
    """Yield only committed token IDs. backend.history is the evaluated prefix."""
    if stats is None:
        stats = DecodeStats()
    verification=settings(verification_level)
    for key in ('level','top_k','min_ratio','max_relaxed_per_block'):
        setattr(stats,'verification_'+key,verification[key])
    if not prompt or max_new_tokens < 0 or not 0 <= draft_max <= DRAFT_MAX:
        raise ValueError(f'nonempty prompt, nonnegative output limit, draft_max 0..{DRAFT_MAX} required')
    if len(prompt) >= backend.context:
        raise ValueError('prompt leaves no space for an answer in the configured context')
    limit = min(max_new_tokens, backend.context-len(prompt))
    if not limit:
        return
    policy = DraftPolicy(draft_max, adaptive)
    stats.draft_policy = ('adaptive-time' if adaptive else 'fixed') if draft_max else 'disabled'
    stats.draft_min = policy.minimum if adaptive else draft_max
    stats.draft_max = draft_max
    start = time.monotonic()
    backend.prefill(prompt, mtp=draft_max > 0)
    stats.prefill_seconds = time.monotonic()-start
    pending = backend.top1()[-1]
    start = time.monotonic()
    try:
        while stats.output_tokens < limit:
            if pending in eos_ids:
                return
            stats.output_tokens += 1
            stats.target_output_tokens += 1
            yield pending
            if stats.output_tokens >= limit:
                return
            pos = backend.position
            proposed_count = min(policy.n, limit-stats.output_tokens, backend.context-pos-1)
            if not proposed_count:
                backend.target([pending])
                if draft_max:
                    backend.catchup([pending], pos)
                pending = backend.top1()[-1]
                continue
            tick=time.monotonic();backend.checkpoint()
            timing={'checkpoint':time.monotonic()-tick,'restore':0.0}
            tick=time.monotonic()
            candidates = [pending]
            for step in range(proposed_count):
                candidate = backend.draft(candidates[-1], pos+step, step == 0)
                candidates.append(candidate)
                if candidate in eos_ids:
                    break
            count = len(candidates)-1
            stats.proposed += count
            for j in range(count):
                stats.proposed_by_position[j] += 1
            stats.max_draft_used = max(stats.max_draft_used, count)
            timing['draft']=time.monotonic()-tick
            tick=time.monotonic();backend.target(candidates)
            if verification_level != DEFAULT_LEVEL:
                choices,eligible=backend.verify(candidates[1:],verification['top_k'],verification['min_ratio'])
            else:
                choices=backend.top1();eligible=[False]*count
            timing['target']=time.monotonic()-tick
            # The first row predicts candidates[1], never candidates[0]. The
            # anchor came from the preceding valid target distribution.
            accepted,relaxed=accepted_prefix(candidates[1:],choices,eligible,verification_level)
            stats.relaxed_accepted+=relaxed
            keep = 1+accepted
            stats.accepted += accepted
            for j in range(accepted):
                stats.accepted_by_position[j] += 1
            if accepted != count:
                stats.semantic_misses += 1
                tick=time.monotonic();backend.commit_prefix(keep)
                timing['restore']=time.monotonic()-tick
                stats.retained_prefix_tokens+=keep
                stats.prefix_restore_seconds+=timing['restore']
                # This correction is already the TARGET's distribution after
                # the valid prefix. It is never selected by the draft. Its own
                # features are evaluated as the anchor of the following block.
                pending = choices[keep-1]
            else:
                stats.full_blocks += 1
                pending = choices[-1]
            requested = policy.n
            # Teacher-force the MTP cache with accepted target features. Its KV
            # projections differ from the target; copying target KV is invalid.
            tick=time.monotonic();backend.catchup(candidates[:keep], pos)
            timing['catchup']=time.monotonic()-tick
            next_n = policy.timed_choice(count, accepted, timing)
            stats.draft_trace.append([requested, count, accepted, next_n])
            stats.timing_trace.append(dict(timing,proposed=count,accepted=accepted,next_n=next_n,
                                           timed=policy.timing_active,predicted_tokens_per_second=policy.predicted_tps))
            accepted_output_chain = 0
            for j,token in enumerate(candidates[1:keep]):
                if token in eos_ids:
                    return
                if token!=choices[j]:
                    stats.relaxed_output_tokens+=1
                    stats.relaxed_trace.append([stats.output_tokens,token,choices[j]])
                stats.output_tokens += 1
                stats.draft_output_tokens += 1
                accepted_output_chain += 1
                stats.max_accepted_draft = max(stats.max_accepted_draft, accepted_output_chain)
                yield token
                if stats.output_tokens >= limit:
                    return
    finally:
        stats.decode_seconds += time.monotonic()-start
