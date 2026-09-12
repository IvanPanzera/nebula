"""Build the self-contained Windows setup executable from the audited sources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent.parent
BUILD=ROOT/'build/installer'
OUTPUT=ROOT/'dist/NebulaSetup'


def main():
    BUILD.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    payload=BUILD/'payload.zip';files={}
    # Explicitly exclude current runtime config, chat history, diagnostic outputs,
    # personal paths, GGUF files and compiled binaries from other hardware.
    runtime={'config.py','profiles.py','native.py','chat.py','decode.py','verification.py',
        'model_registry.py','storage.py','hardware.py','quant.py','monitor.py','documents.py','document_worker.py',
        'web_server.py','web_worker.py','test_cpu_dispatch.py','test_profiles.py','test_large_profiles.py',
        'test_gpu.py','test_storage.py','test_hardware.py','test_decode.py','test_optimizations.py',
        'test_verification.py','test_webui.py'}
    for path in (ROOT/'qwen').iterdir():
        if path.name in runtime or (path.suffix in ('.c','.h','.cu') and path.name!='reference_check.c') or path.name in ('Makefile','requirements.txt','expert_ranking.json','hardware_profiles.json','storage.json','hotlist.json'):
            files[path.relative_to(ROOT).as_posix()]=path
    for path in (ROOT/'qwen/web').iterdir():
        if path.is_file():files[path.relative_to(ROOT).as_posix()]=path
    for path in (ROOT/'installer').iterdir():
        if path.suffix in ('.py','.ps1','.sh','.json') and not path.name.startswith(('make_','build_release','test_','check_')):
            files[path.relative_to(ROOT).as_posix()]=path
    tokenizer=json.loads((ROOT/'assets/tokenizer/manifest.json').read_text())
    for item in tokenizer['files']:
        path=ROOT/'assets/tokenizer'/item['name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Tokenizer checksum mismatch')
        files[path.relative_to(ROOT).as_posix()]=path
    files['assets/tokenizer/manifest.json']=ROOT/'assets/tokenizer/manifest.json'
    files['THIRD_PARTY.md']=ROOT/'installer/THIRD_PARTY.md'
    files['QWEN-LICENSE.txt']=ROOT/'installer/QWEN-LICENSE.txt'
    files['LICENSE']=ROOT/'LICENSE'
    files['qwen/README.md']=ROOT/'installer/README.md'
    manifest={}
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):
            raw=path.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            info=zipfile.ZipInfo(name,date_time=(2026,9,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw)
        z.writestr('release-manifest.json',json.dumps(manifest,indent=2))
    sha=hashlib.sha256(payload.read_bytes()).hexdigest()
    source=(ROOT/'installer/NebulaSetup.cs').read_text().replace('__PAYLOAD_SHA256__',sha)
    cs=BUILD/'NebulaSetup.cs';cs.write_text(source)
    csc=Path(os.environ.get('SystemRoot','C:/Windows'))/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    output=OUTPUT/'NebulaSetup.exe'
    subprocess.run([str(csc),'/nologo','/target:winexe','/platform:x64','/optimize+',
        '/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.IO.Compression.dll',
        '/r:System.IO.Compression.FileSystem.dll','/win32manifest:'+str(ROOT/'installer/app.manifest'),
        '/resource:'+str(payload)+',payload.zip','/out:'+str(output),str(cs)],check=True)
    result=dict(executable=str(output.name),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,
        payload_sha256=sha,files=manifest)
    (OUTPUT/'release-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUTPUT/'SHA256SUMS.txt').write_text(result['sha256']+'  NebulaSetup.exe\n')
    shutil.copy2(ROOT/'installer/README.md',OUTPUT/'README.md')
    print(f'Built {output}: {result["bytes"]/2**20:.1f} MiB, {len(manifest)} files')

if __name__=='__main__':main()
