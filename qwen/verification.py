"""User-selected approximate verification, independent of expert placement.

Ratios use the target distribution at temperature one. Level three preserves
the existing greedy path; lower levels intentionally change its output.
"""
SCALE = 3
DEFAULT_LEVEL = 3
LEVELS = {
    3: ('Strict', 1, 1.0, 0),
    2: ('Limited tolerance', 3, .80, 1),
    1: ('Wide tolerance', 10, .20, 4),
}


def settings(level):
    if type(level) is not int or level not in LEVELS:
        raise ValueError('The verification level must be an integer from 1 to 3.')
    label, top_k, ratio, cap = LEVELS[level]
    return dict(level=level, label=label, top_k=top_k, min_ratio=ratio,
                max_relaxed_per_block=cap)


def accepted_prefix(proposals, choices, eligible, level):
    """Stop at the first rejection; all later rows have a discarded prefix."""
    cap = settings(level)['max_relaxed_per_block']
    accepted = relaxed = 0
    for token, target, allowed in zip(proposals, choices, eligible):
        if token != target:
            if not allowed or relaxed >= cap:
                break
            relaxed += 1
        accepted += 1
    return accepted, relaxed
