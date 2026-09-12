"""Approved installation profiles. Counts include GPU experts' CPU copies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GIB = 2**30
TABLE_SHA256 = '9a78419cd551938469ea10169fb7eec2801176efc0c89892eaa305e4120ecf52'
GPU_TIERS = (8, 12, 16, 24)
RAM_TIERS = (32, 64, 96, 128)


def table():
    raw = (ROOT/'hardware_profiles.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != TABLE_SHA256:
        raise ValueError('The approved hardware table has changed; rebuild the installer manifest.')
    return json.loads(raw)['profiles']


def lower_tier(byte_count, tiers, tolerance=0):
    eligible = [tier for tier in tiers if byte_count + tolerance >= tier*GIB]
    if not eligible:
        missing = tiers[0]*GIB-byte_count
        raise ValueError(f'At least {tiers[0]} GiB required; missing {missing/GIB:.2f} GiB.')
    return max(eligible)


def select(vram_bytes, ram_bytes):
    # NVIDIA reserves a few MiB before reporting memory.total. This allowance
    # recognizes nominal 8/12/16/24 GiB cards, never rounds 20 up to 24.
    gpu = lower_tier(vram_bytes, GPU_TIERS, 64*2**20)
    ram = lower_tier(ram_bytes, RAM_TIERS)
    return dict(next(p for p in table() if p['vram_gib'] == gpu and p['ram_gib'] == ram))


def validate_runtime(value):
    if value.get('version') != 1:
        raise ValueError('Unsupported installed hardware profile version')
    approved = select(value['profile']['vram_gib']*GIB, value['profile']['ram_gib']*GIB)
    if value['profile'] != approved:
        raise ValueError('Installed settings differ from the approved hardware table')
    if type(value['cpu_threads']) is not int or not 1 <= value['cpu_threads'] <= 28:
        raise ValueError('CPU threads must be in 1..28')
    if not isinstance(value['gpu_uuid'], str) or not value['gpu_uuid'].startswith('GPU-'):
        raise ValueError('A physical NVIDIA GPU UUID is required')
    if len(value['hotlist_sha256']) != 64:
        raise ValueError('Missing installed hotlist checksum')
    return value


def read_runtime(root=ROOT):
    path = Path(root)/'runtime_profile.json'
    return validate_runtime(json.loads(path.read_text())) if path.is_file() else None


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def configure(root, index, profile, gpu_uuid, cpu_threads):
    """Commit runtime marker last; installer invokes this before first launch."""
    from storage import RANKING_SHA256
    root, index = Path(root), Path(index)
    raw = (root/'expert_ranking.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != RANKING_SHA256:
        raise ValueError('The complete expert ranking has changed')
    rows = json.loads(raw)['generation']
    if len(rows) != 48 or any(sorted(row) != list(range(512)) for row in rows):
        raise ValueError('Invalid complete expert ranking')
    x = profile['gpu_experts']
    hot = dict(version=1, population='generation', resident_per_layer=x, layers=48,
               experts_per_layer=512, experts=[row[:x] for row in rows],
               model_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
               source_ranking_sha256=RANKING_SHA256)
    atomic_json(root/'hotlist.json', hot)
    atomic_json(root/'storage.json', dict(ram_experts_per_layer=profile['ram_experts'],
                ngram=profile['ngram'], buffer_mib=1024, io_threads=2))
    value = dict(version=1, profile=profile, gpu_uuid=gpu_uuid, cpu_threads=cpu_threads,
                 hotlist_sha256=hashlib.sha256((root/'hotlist.json').read_bytes()).hexdigest())
    atomic_json(root/'runtime_profile.json', validate_runtime(value))
    return value
