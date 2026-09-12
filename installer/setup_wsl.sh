#!/usr/bin/env bash
set -euo pipefail
payload=$1
installation=$2
plan=$3
export DEBIAN_FRONTEND=noninteractive
export PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.8/bin:$PATH
export NEEDRESTART_MODE=a
echo 'Installing Linux build tools and Python...'
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl build-essential cmake ninja-build git python3 python3-venv python3-pip
python3 - "$payload" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'installer'))
from download import download
info=json.loads((Path(sys.argv[1])/'installer/dependencies.json').read_text())['cuda_keyring']
download(info['url'],Path('/tmp/nebula-cuda-keyring.deb'),info['sha256'],info['size'])
PY
dpkg -i /tmp/nebula-cuda-keyring.deb
apt-get update
# WSL already receives its GPU driver from Windows. Install toolkit components only.
apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8
if ! id nebula >/dev/null 2>&1; then useradd --create-home --shell /bin/bash nebula; fi
usermod --lock nebula
mkdir -p /home/nebula/app/qwen /home/nebula/app/installer /home/nebula/app/work/qwen /home/nebula/app/assets
# Payload contains release code and verified tokenizer assets, never user chats or weights.
cp -a "$payload/qwen/." /home/nebula/app/qwen/
cp -a "$payload/installer/." /home/nebula/app/installer/
cp -a "$payload/assets/." /home/nebula/app/assets/
cp "$payload/QWEN-LICENSE.txt" "$payload/THIRD_PARTY.md" "$payload/LICENSE" /home/nebula/app/
printf '[user]\ndefault=nebula\n' > /etc/wsl.conf
chown -R nebula:nebula /home/nebula/app
mkdir -p "$installation/model" "$installation/cache"
chown nebula:nebula "$installation/model" "$installation/cache" || true
runuser -u nebula -- env PATH="$PATH" bash /home/nebula/app/installer/build_wsl.sh "$installation" "$plan"
apt-get clean
