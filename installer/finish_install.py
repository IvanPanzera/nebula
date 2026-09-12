"""Apply the hardware selection only after weights and native checks succeeded."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'qwen'))
from profiles import configure,select
from hardware import detect

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8-sig'));gpu=plan['Hardware']['Gpu']
    profile=select(gpu['TotalBytes'],plan['Hardware']['RamBytes'])
    if profile!=plan['Profile']:raise ValueError('Hardware plan does not match the approved profiles')
    actual=detect(ROOT)
    selected=[g for g in actual['gpus'] if g['uuid']==gpu['Uuid']]
    if not selected:raise ValueError('Selected GPU is unavailable in WSL')
    for label,available,required in [('RAM',actual['memory']['wsl_available_bytes'],plan['Budget']['ram_required_free']),
                                   ('VRAM',selected[0]['free_bytes'],plan['Budget']['gpu_required_free'])]:
        if available is None:raise ValueError(f'Unable to verify available {label}')
        if available<required:raise ValueError(f'Release another {(required-available)/2**30:.2f} GiB of {label}, then run Setup again. The downloaded model is preserved.')
    threads=min(plan['CpuThreads'],actual['cpu']['suggested_threads'])
    library=C.CDLL(str(ROOT/'qwen/build/libqwen.so'))
    library.qwen_validate_index.argtypes=[C.c_char_p];library.qwen_validate_index.restype=C.c_int
    library.qwen_last_error.restype=C.c_char_p
    if library.qwen_validate_index(str(ROOT/'work/qwen/model.index').encode()):
        raise ValueError(library.qwen_last_error().decode())
    installed=configure(ROOT/'qwen',ROOT/'work/qwen/model.index',profile,gpu['Uuid'],threads)
    print('Hardware profile applied: '+json.dumps(installed['profile']),flush=True)
    # Verify tokenizer/template/index/hotlist/storage through the actual binding,
    # without allocating another full model just to mark installation complete.
    from chat import encode_chat
    from native import load_hotlist
    from storage import load_storage,load_ranking
    encode_chat([dict(role='user',content='Hello')])
    hot=load_hotlist(ROOT/'work/qwen/model.index',profile['gpu_experts'])
    load_ranking(hot);load_storage(resident=profile['gpu_experts'])
    print('Nebula installation verified. First launch will load the model.',flush=True)

if __name__=='__main__':main()
