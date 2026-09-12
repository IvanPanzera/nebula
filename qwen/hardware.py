#!/usr/bin/env python3
"""Read-only WSL2 hardware probe for the Windows installer.

Uses only the Python standard library, so it can run before bootstrap.py.
It never loads weights, changes the running service, or selects an unapproved profile.
"""
import argparse
import csv
import ctypes
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
        return result.stdout.strip(), None
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)

def parse_cpuinfo(text, affinity=None):
    entries = []
    for block in text.strip().split('\n\n'):
        entry = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        entry = {k.strip(): v.strip() for k, v in entry.items()}
        if entry and (affinity is None or int(entry.get('processor', -1)) in affinity):
            entries.append(entry)
    flags = set.intersection(*(set(e.get('flags', '').split()) for e in entries)) if entries else set()
    base = {'avx2', 'fma', 'f16c'}
    backend = 'avx512' if base | {'avx512f', 'avx512dq', 'avx512bw', 'avx512vl'} <= flags else 'avx2' if base <= flags else 'scalar'
    pairs = {(e['physical id'], e['core id']) for e in entries if 'physical id' in e and 'core id' in e}
    physical = len(pairs) if len(pairs) and all('core id' in e and 'physical id' in e for e in entries) else None
    return {'name': entries[0].get('model name') if entries else None, 'logical_available': len(entries),
            'physical_available': physical, 'isa_from_flags': backend,
            'suggested_threads': max(1, min(28, physical or len(entries) or 1)),
            'thread_limit_current_engine': 28, 'flags': sorted(flags)}

def parse_memory(text):
    fields = {m[0]: int(m[1])*1024 for m in re.findall(r'^(\w+):\s+(\d+) kB', text, re.M)}
    return {'wsl_total_bytes': fields.get('MemTotal'), 'wsl_available_bytes': fields.get('MemAvailable'),
            'swap_total_bytes': fields.get('SwapTotal'), 'swap_free_bytes': fields.get('SwapFree')}

def parse_gpus(text):
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if len(r) != 7:
            raise ValueError('Unexpected nvidia-smi output')
        idx, uuid, name, total, free, cc, driver = (v.strip() for v in r)
        rows.append({'index': int(idx), 'uuid': uuid, 'name': name, 'total_bytes': int(total)*1024**2,
                     'free_bytes': int(free)*1024**2, 'compute_capability': cc,
                     'cuda_arch': 'sm_'+cc.replace('.', ''), 'driver': driver})
    return rows

def detect(storage_path=ROOT):
    if platform.system() != 'Linux' or 'microsoft' not in platform.release().lower():
        raise RuntimeError('Run from the configured WSL2 distribution on Windows.')
    affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    cpu = parse_cpuinfo(Path('/proc/cpuinfo').read_text(), affinity)
    library = ROOT/'build/libqwen_cpu.so'
    if library.exists():
        lib = ctypes.CDLL(str(library))
        if hasattr(lib, 'qc_backend_name'):
            lib.qc_backend_name.restype = ctypes.c_char_p
            cpu['selected_native_backend'] = lib.qc_backend_name().decode()
    raw, gpu_error = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version', '--format=csv,noheader,nounits'])
    gpus = parse_gpus(raw) if raw else []
    codes, nvcc_error = command(['nvcc', '--list-gpu-code'])
    supported = set(re.findall(r'\bsm_\d+[a-z]?\b', codes or ''))
    for gpu in gpus:
        gpu['nvcc_supports_detected_arch'] = gpu['cuda_arch'] in supported if codes else None
    mount, mount_error = command(['findmnt', '--json', '--target', str(Path(storage_path).resolve()), '--output', 'SOURCE,TARGET,FSTYPE'])
    disk = shutil.disk_usage(storage_path)
    return {'platform': {'distribution_kernel': platform.release(), 'machine': platform.machine(),
                         'supported_scope': 'Windows + WSL2 + NVIDIA, x86-64'},
            'cpu': cpu, 'memory': parse_memory(Path('/proc/meminfo').read_text()), 'gpus': gpus,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'storage': {'path': str(Path(storage_path).resolve()), 'free_bytes': disk.free, 'total_bytes': disk.total,
                        'mount': json.loads(mount) if mount else None,
                        'physical_media': 'See Windows volume data; a WSL mount does not establish SSD speed.'},
            'diagnostics': {'nvidia_smi_error': gpu_error, 'nvcc_error': nvcc_error, 'mount_error': mount_error},
            'configuration_applied': False, 'profiles_status': 'Approved 4 VRAM x 4 RAM profiles in hardware_profiles.json',
            'streaming_status': 'Explicit expert and PLE SSD streaming; installer selects approved RAM residency and N-gram location'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-path', type=Path, default=ROOT)
    parser.add_argument('--cuda-arch', action='store_true', help='Print build architecture for a single visible NVIDIA GPU.')
    args = parser.parse_args()
    result = detect(args.storage_path)
    if args.cuda_arch:
        gpus = result['gpus']
        visible = result['cuda_visible_devices']
        if visible is not None:
            first = visible.split(',')[0].strip()
            gpus = [g for g in gpus if first in (str(g['index']), g['uuid'])]
        if len(gpus) != 1:
            raise SystemExit('Select one GPU with CUDA_VISIBLE_DEVICES or pass CUDA_ARCH=sm_XX to make.')
        if not gpus[0]['nvcc_supports_detected_arch']:
            raise SystemExit('Installed nvcc does not support the detected GPU architecture; check the CUDA toolkit.')
        print(gpus[0]['cuda_arch'])
    else:
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
