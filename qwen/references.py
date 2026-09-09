#!/usr/bin/env python3
"""Capture upstream implementation sources as reference data, never execute them."""
import hashlib
import json
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent.parent / "work/qwen/reference"
OUT.mkdir(parents=True, exist_ok=True)
pr = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/pulls/28243", timeout=60)
pr.raise_for_status()
pr = pr.json()
(OUT / "mtp_pr.json").write_text(json.dumps(pr, indent=2))
sha = pr["head"]["sha"]
repo = pr["head"]["repo"]["full_name"]
paths = ["src/models/qwen4exp.cpp", "src/models/qwen4exp-mtp.cpp", "src/models/qwen4exp.h",
         "common/speculative.cpp", "src/llama-graph.cpp", "src/llama-model-loader.cpp",
         "ggml/src/ggml-cuda/gated_delta_net.cu", "ggml/src/ggml-cpu/ops.cpp",
         "ggml/src/ggml-quants.c", "ggml/src/ggml-common.h", "LICENSE"]
manifest = dict(repo=repo, revision=sha, state=pr["state"], draft=pr["draft"], files=[])
for path in paths:
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        print(f"absent: {path}", flush=True)
        continue
    r.raise_for_status()
    target = OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(r.content)
    manifest["files"].append(dict(path=path, url=url, sha256=hashlib.sha256(r.content).hexdigest()))
    print(f"reference {path}", flush=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
