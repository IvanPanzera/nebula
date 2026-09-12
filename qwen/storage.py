"""Explicit storage policy, shared by CLI/WebUI; hardware profiles are user-set."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANKING_SHA256 = '3f698494cb9c0e1d4882b60bbde60cdbe10ca371be86a2a1470a3e87b7571855'
DEFAULT = dict(ram_experts_per_layer=512, ngram='ram', buffer_mib=1024, io_threads=2)


def load_storage(path=None, resident=27):
    config = dict(DEFAULT)
    file = Path(path) if path is not None else ROOT/'storage.json'
    if file.is_file():
        value = json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or set(value)-set(DEFAULT):
            raise ValueError('Unknown storage configuration fields')
        config.update(value)
    elif path is not None:
        raise ValueError(f'Storage configuration does not exist: {file}')
    for key, lo, hi in [('ram_experts_per_layer', resident, 512), ('buffer_mib', 32, 4096), ('io_threads', 1, 4)]:
        if type(config[key]) is not int or not lo <= config[key] <= hi:
            raise ValueError(f'{key} must be an integer from {lo} to {hi}')
    if config['ngram'] not in ('ram', 'ssd'):
        raise ValueError('ngram must be ram or ssd')
    return config


def load_ranking(hotlist, path=None):
    raw=(Path(path) if path is not None else ROOT/'expert_ranking.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=RANKING_SHA256:
        raise ValueError('The complete expert ranking has changed')
    ranking=json.loads(raw)['generation']
    if len(ranking)!=48 or any(sorted(row)!=list(range(512)) for row in ranking):
        raise ValueError('Expected 512 distinct experts in each of 48 layers')
    resident=len(hotlist)//48
    for layer, row in enumerate(ranking):
        if row[:resident]!=hotlist[layer*resident:(layer+1)*resident]:
            raise ValueError('Storage ranking must preserve every fixed GPU expert')
    return [e for row in ranking for e in row]
