#!/usr/bin/env python3
"""Prepare a pinned external comparison binary; not the production C engine."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'work/qwen/upstream'
COMMIT='d1a92352cbd417fd840b4e765c0b82f5fe3d1d89'
URL=f'https://codeload.github.com/danielhanchen/llama.cpp/tar.gz/{COMMIT}'


def build_adapter(source,build):
    subprocess.run(['cc','-O2','-std=c11','-Wall','-Wextra',str(ROOT/'reference_check.c'),
        '-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-L'+str(build/'bin'),
        '-Wl,-rpath,$ORIGIN','-lllama','-lggml','-lggml-base','-lggml-cpu',
        '-o',str(build/'bin/qwen-reference-check')],check=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    archive=OUT/(COMMIT+'.tar.gz');source=OUT/('llama.cpp-'+COMMIT)
    if not archive.exists():
        temp=archive.with_suffix('.partial')
        with requests.get(URL,stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            with temp.open('wb') as f:
                for data in response.iter_content(1024**2):f.write(data)
        temp.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if not source.exists():
        with tarfile.open(archive,'r:gz') as package:
            for entry in package.getmembers():
                if Path(entry.name).parts[0]!=source.name:raise ValueError('unexpected archive root')
            package.extractall(OUT,filter='data')
    py=ROOT/'build/venv/bin/python';bin_dir=py.parent
    subprocess.run([str(py),'-m','pip','install','cmake==4.4.3','ninja==1.13.2'],check=True)
    build=OUT/'build';cmake=str(bin_dir/'cmake')
    subprocess.run([cmake,'-S',str(source),'-B',str(build),'-G','Ninja',
        '-DCMAKE_MAKE_PROGRAM='+str(bin_dir/'ninja'),'-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=89','-DLLAMA_CURL=OFF',
        '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_SERVER=OFF'],check=True)
    subprocess.run([cmake,'--build',str(build),'--target','llama-completion','-j','4'],check=True)
    build_adapter(source,build)
    (OUT/'provenance.json').write_text(json.dumps(dict(commit=COMMIT,url=URL,archive_sha256=digest,
        purpose='external numerical/quality comparison only; production is qwen/qwen.c and gpu.cu'),indent=2))
    print('External reference ready:',build/'bin/llama-completion',flush=True)


if __name__=='__main__':main()
