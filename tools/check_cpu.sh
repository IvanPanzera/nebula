#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python=${1:-python3}
python=$("$python" -c 'import sys;print(sys.executable)')
make -C qwen -j2 build/libqwen_cpu.so build/test_handoffs build/test_host_weights build/test_ssd build/test_storage_math
qwen/build/test_handoffs
qwen/build/test_host_weights
qwen/build/test_ssd
qwen/build/test_storage_math
"$python" qwen/test_cpu_dispatch.py
"$python" installer/test_prepare.py
(
    cd qwen
  NEBULA_TEST_CPU_ONLY=1 "$python" -m unittest test_profiles test_storage test_hardware test_decode test_verification test_webui test_documents
)
