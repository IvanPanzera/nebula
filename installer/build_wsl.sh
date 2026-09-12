#!/usr/bin/env bash
set -euo pipefail
installation=$1
plan=$2
cd /home/nebula/app
python3 -m venv qwen/build/venv
python=qwen/build/venv/bin/python
"$python" -m pip install --disable-pip-version-check -r qwen/requirements.txt
export CUDA_VISIBLE_DEVICES=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["Hardware"]["Gpu"]["Uuid"])' "$plan")
arch=$(python3 qwen/hardware.py --cuda-arch)
jobs=$(python3 -c 'import json,sys;print(max(1,min(8,json.load(open(sys.argv[1]))["CpuThreads"])))' "$plan")
make -C qwen -j"$jobs" CUDA_ARCH="$arch" all build/libqwen_cpu.so build/libqwen_probe.so
echo 'Checking CPU instruction dispatch and CUDA numerical results...'
(cd qwen && build/venv/bin/python test_cpu_dispatch.py && build/venv/bin/python -m unittest test_profiles test_large_profiles test_gpu test_storage)
"$python" installer/prepare_model.py --model-dir "$installation/model" --cache "$installation/cache" --jobs "$jobs"
