"""The official Flash-Next speculative model exposed by Nebula."""
from pathlib import Path
import sys

from config import CONTEXT, DRAFT_MIN, DRAFT_MAX, HANDOFF_MODE, RESIDENT_EXPERTS, MISSING_MASS_LIMIT

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
GPU_LOCK = Path('/tmp/tenseng-qwen-model.lock')


def models():
    return [dict(id='flash-next', label='Flash-Next · speculative',
                    context=CONTEXT, speculative=True, draft_min=DRAFT_MIN,
                    draft_max=DRAFT_MAX, available=True, target_execution=HANDOFF_MODE,
                    resident_experts_per_layer=RESIDENT_EXPERTS, missing_mass_limit=MISSING_MASS_LIMIT)]


def get_model(model_id):
    for entry in models():
        if entry['id'] == model_id:
            if not entry['available']:
                raise ValueError('The selected model weights are unavailable.')
            return entry
    raise ValueError('Model is not configured.')


def worker_command(model_id):
    get_model(model_id)
    return [sys.executable, '-u', str(ROOT/'web_worker.py')]


def public_models():
    return [{k: v for k, v in entry.items() if k != 'file'} for entry in models()]
