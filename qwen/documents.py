"""Local document conversion protocol; shares the chat's exclusive job gate."""
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import uuid

from model_registry import PROJECT

DOCUMENTS = PROJECT/'work/documents'
OCR_PYTHON = PROJECT/'runtime/ocr/bin/python'
MAX_UPLOAD = 25 * 1024 * 1024
MAX_ATTACH = 250000


def available():
    model = PROJECT/'model/PaddleOCR-VL-1.5'
    return OCR_PYTHON.is_file() and all((model/name).is_file() for name in (
        'PaddleOCR-VL-1.5.gguf', 'PaddleOCR-VL-1.5-mmproj.gguf'))


def page_selection(value, count):
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError('Invalid page range.')
    selected = set()
    for part in value.split(',') if value.strip() else [f'1-{count}']:
        match = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', part)
        if not match:
            raise ValueError('Use a page range such as 1-5 or 1,3,7-9.')
        start, end = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= end <= count:
            raise ValueError(f'Pages must be between 1 and {count}.')
        if end-start >= 50:
            raise ValueError('Select at most 50 pages per conversion.')
        selected.update(range(start-1, end))
    if not 1 <= len(selected) <= 50:
        raise ValueError('Select at most 50 pages per conversion.')
    return sorted(selected)


def create_job(name, data, pages='', force_ocr=False):
    name = name.replace('\\', '/').rsplit('/', 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in ('.pdf', '.png', '.jpg', '.jpeg', '.webp'):
        raise ValueError('Supported formats: PDF, PNG, JPEG and WebP.')
    if not 0 < len(data) <= MAX_UPLOAD or not name or len(name) > 180 or any(ord(c)<32 for c in name):
        raise ValueError('Invalid file or size above 25 MiB.')
    if suffix == '.pdf' and not data[:1024].lstrip().startswith(b'%PDF-'):
        raise ValueError('The file is not a valid PDF.')
    if not isinstance(pages, str) or len(pages) > 200 or type(force_ocr) is not bool:
        raise ValueError('Invalid document options.')
    ident = uuid.uuid4().hex
    folder = DOCUMENTS/ident
    folder.mkdir(parents=True)
    (folder/('source'+suffix)).write_bytes(data)
    request = dict(id=ident, name=name, source='source'+suffix, pages=pages, force_ocr=force_ocr)
    (folder/'request.json').write_text(json.dumps(request), encoding='utf-8')
    return folder


def download(path):
    match = re.fullmatch(r'/api/documents/([a-f0-9]{32})/(document\.md|document\.zip)', path)
    if not match:
        raise ValueError('Invalid document.')
    return DOCUMENTS/match[1]/match[2]


def terminate(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # The OCR worker can own a llama-server child, including after a crash.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stream_document(engine, folder):
    events = queue.Queue()
    process = None
    reader = None
    try:
        with (folder/'engine.log').open('ab') as log:
            process = subprocess.Popen([str(OCR_PYTHON), '-u', str(PROJECT/'qwen/document_worker.py'), str(folder)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding='utf-8', start_new_session=True)
        engine.document_process = process
        reader = threading.Thread(target=engine._reader, args=(process, events), daemon=True)
        reader.start()
        while True:
            if engine.cancel_requested:
                terminate(process)
                yield dict(type='document_cancelled')
                return
            try:
                event = events.get(timeout=.5)
            except queue.Empty:
                yield dict(type='heartbeat')
                continue
            if engine.cancel_requested:
                continue
            kind = event.get('type')
            if kind == 'needs_ocr':
                # Native PDF text never evicts the chat model. Scans acquire
                # the GPU only after the currently loaded model has exited.
                engine.close()
                process.stdin.write('go\n')
                process.stdin.flush()
            elif kind == 'phase':
                engine.phase = event['phase']
                yield event
            elif kind == 'document_done':
                process.wait(timeout=10)
                if process.returncode:
                    raise RuntimeError('The document worker exited with an error.')
                yield event
                return
            elif kind in ('error', 'process_exit'):
                raise RuntimeError(event.get('message', 'Conversion interrupted. See the document log.'))
            else:
                yield event
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        yield dict(type='error', message=str(exc))
    finally:
        terminate(process)
        if reader:
            reader.join(timeout=2)
        if process:
            process.stdin.close()
            process.stdout.close()
        engine.document_process = None
        engine.phase = 'ready' if engine.loaded else 'idle'
        engine.gate.release()
