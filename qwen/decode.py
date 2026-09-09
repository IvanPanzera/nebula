"""Greedy speculative decoding with one checkpoint and causal prefix replay.

The draft is deterministic. Accepted tokens exactly match the greedy verifier;
there is no probability tolerance or claim of equivalence to stochastic sampling.
Expert handoff is resolved inside the GPU target backend and does not penalize N.
"""
from dataclasses import dataclass, field
import time
from config import DRAFT_MIN, DRAFT_MAX


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
    proposed_by_position: list = field(default_factory=lambda: [0]*DRAFT_MAX)
    accepted_by_position: list = field(default_factory=lambda: [0]*DRAFT_MAX)


def generate(backend, prompt, *, max_new_tokens=512, draft_max=DRAFT_MAX,
             adaptive=True, eos_ids=(248044, 248046), stats=None):
    """Yield only committed token IDs. backend.history is the evaluated prefix."""
    if stats is None:
        stats = DecodeStats()
    if not prompt or max_new_tokens < 0 or not 0 <= draft_max <= DRAFT_MAX:
        raise ValueError(f'nonempty prompt, nonnegative output limit, draft_max 0..{DRAFT_MAX} required')
    if len(prompt) >= backend.context:
        raise ValueError('prompt leaves no space for an answer in the configured context')
    limit = min(max_new_tokens, backend.context-len(prompt))
    if not limit:
        return
    policy = DraftPolicy(draft_max, adaptive)
    stats.draft_policy = ('adaptive' if adaptive else 'fixed') if draft_max else 'disabled'
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
            backend.checkpoint()
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
            backend.target(candidates)
            choices = backend.top1()
            # The first row predicts candidates[1], never candidates[0]. The
            # anchor came from the preceding valid target distribution.
            accepted = 0
            for j in range(count):
                if candidates[j+1] != choices[j]:
                    break
                accepted += 1
            keep = 1+accepted
            stats.accepted += accepted
            for j in range(accepted):
                stats.accepted_by_position[j] += 1
            if accepted != count:
                stats.semantic_misses += 1
                backend.restore()
                backend.target(candidates[:keep])
                stats.replay_tokens += keep
                replay_choice = backend.top1()[-1]
                if replay_choice != choices[keep-1]:
                    raise RuntimeError('target argmax drift after speculative rollback')
                pending = replay_choice
            else:
                stats.full_blocks += 1
                pending = choices[-1]
            requested = policy.n
            next_n = policy.update(count, accepted)
            stats.draft_trace.append([requested, count, accepted, next_n])
            # Teacher-force the MTP cache with accepted target features. Its KV
            # projections differ from the target; copying target KV is invalid.
            backend.catchup(candidates[:keep], pos)
            accepted_output_chain = 0
            for token in candidates[1:keep]:
                if token in eos_ids:
                    return
                stats.output_tokens += 1
                stats.draft_output_tokens += 1
                accepted_output_chain += 1
                stats.max_accepted_draft = max(stats.max_accepted_draft, accepted_output_chain)
                yield token
                if stats.output_tokens >= limit:
                    return
    finally:
        stats.decode_seconds += time.monotonic()-start
