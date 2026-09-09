#!/usr/bin/env python3
"""Install test/frontend dependencies into a project-local venv only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import requests

root = Path(__file__).resolve().parent
build = root / "build"
build.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(build / "venv")], check=True)
r = requests.get("https://pypi.org/pypi/pip/json", timeout=60)
r.raise_for_status()
info = r.json()
wheel = next(x for x in info["urls"] if x["filename"].endswith("py3-none-any.whl"))
r = requests.get(wheel["url"], timeout=120)
r.raise_for_status()
if hashlib.sha256(r.content).hexdigest() != wheel["digests"]["sha256"]:
    raise ValueError("pip wheel hash mismatch")
path = build / wheel["filename"]
path.write_bytes(r.content)
env = dict(os.environ, PYTHONPATH=str(path))
py = str(build / "venv/bin/python")
subprocess.run([py, "-m", "pip", "install", "pip", "-r", str(root / "requirements.txt")], env=env, check=True)
freeze = subprocess.check_output([py, "-m", "pip", "freeze"], text=True)
(build / "requirements.lock").write_text(freeze)
print(freeze, flush=True)
