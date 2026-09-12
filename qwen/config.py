"""Daily limits shared by the CLI, WebUI and native binding."""
CONTEXT = 24576
DRAFT_MIN = 4
DRAFT_MAX = 16
RESIDENT_EXPERTS = 27
CPU_THREADS = 14
PREFILL_BATCH = 2048
HANDOFF_MODE = 'hybrid-layer-cpu'
MISSING_MASS_LIMIT = 0.10
HOTLIST_SHA256 = '08b2f1e814a195a9ab8979529d6f0729d833f54f29662b483ef8672482778ff4'

# Development installations without a profile retain the official defaults.
# Installed profiles are checked against the approved document on every start.
from profiles import read_runtime
_installed = read_runtime()
if _installed:
    import os
    _profile = _installed['profile']
    CONTEXT = _profile['context']
    RESIDENT_EXPERTS = _profile['gpu_experts']
    PREFILL_BATCH = _profile['prefill']
    CPU_THREADS = _installed['cpu_threads']
    HOTLIST_SHA256 = _installed['hotlist_sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = _installed['gpu_uuid']
