"""Two chat modes: Flash-Next and one locally configured standalone model."""
import json
from pathlib import Path
import sys

from config import CONTEXT, DRAFT_MIN, DRAFT_MAX

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LIGHT_CONFIG = PROJECT/'model/light.json'
GPU_LOCK = Path('/tmp/tenseng-qwen-model.lock')


def models():
    entries = [dict(id='flash-next', label='Flash-Next · speculativo',
                    context=CONTEXT, speculative=True, draft_min=DRAFT_MIN,
                    draft_max=DRAFT_MAX, available=True)]
    if LIGHT_CONFIG.is_file():
        light = json.loads(LIGHT_CONFIG.read_text())
        path = (PROJECT/'model'/light['file']).resolve()
        if not path.is_relative_to((PROJECT/'model').resolve()) or path.suffix != '.gguf':
            raise ValueError('Percorso del modello light non valido.')
        context = light.get('context', 24576)
        if type(context) is not int or not 1024 <= context <= 32768:
            raise ValueError('Contesto del modello light non valido.')
        entries.append(dict(id='light', label=light['label'], context=context,
                            speculative=False, available=path.is_file(), file=str(path)))
    return entries


def get_model(model_id):
    for entry in models():
        if entry['id'] == model_id:
            if not entry['available']:
                raise ValueError('I pesi del modello selezionato non sono disponibili.')
            return entry
    raise ValueError('Modello non configurato.')


def worker_command(model_id):
    get_model(model_id)
    worker = 'web_worker.py' if model_id == 'flash-next' else 'light_worker.py'
    return [sys.executable, '-u', str(ROOT/worker)]


def public_models():
    return [{k: v for k, v in entry.items() if k != 'file'} for entry in models()]
