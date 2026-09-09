"""PDF text extraction and PaddleOCR-VL, isolated from the inference runtime."""
from contextlib import redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

from documents import MAX_ATTACH, page_selection
from model_registry import PROJECT, GPU_LOCK

PROTOCOL = sys.stdout


def emit(kind, **values):
    print(json.dumps(dict(type=kind, **values), ensure_ascii=False), file=PROTOCOL, flush=True)


def native_text(page):
    text = page.get_text().strip()
    area = max(1, page.rect.get_area())
    scanned = any((page.rect & type(page.rect)(item['bbox'])).get_area()/area > .6
                  for item in page.get_image_info())
    return sum(c.isalnum() for c in text) >= 80 and '\ufffd' not in text and not scanned


class OCR:
    def __init__(self):
        self.process = None
        self.lock = GPU_LOCK.open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            binary = PROJECT/'runtime/llama/llama-server'
            model = PROJECT/'model/PaddleOCR-VL-1.5'
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = str(binary.parent)+':'+env.get('LD_LIBRARY_PATH', '')
            command = [str(binary), '-m', str(model/'PaddleOCR-VL-1.5.gguf'),
                '--mmproj', str(model/'PaddleOCR-VL-1.5-mmproj.gguf'),
                '--alias', 'paddle-ocr', '--host', '127.0.0.1', '--port', str(port),
                '--no-webui', '--ctx-size', '8192', '--parallel', '1', '--batch-size', '1024',
                '--ubatch-size', '256', '--gpu-layers', 'all', '--fit', 'off',
                '--flash-attn', 'on', '--temp', '0', '--no-context-shift', '--jinja']
            self.process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr, env=env)
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Il motore OCR non si è avviato. Consulta engine.log del documento.')
                try:
                    with urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError('Tempo di avvio OCR superato.')
            # Layout on CPU; only the compact vision/language recognizer uses
            # CUDA. Cache and official assets stay in the moved application.
            os.environ['PADDLE_PDX_CACHE_HOME'] = str(model/'pipeline')
            os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            from paddleocr import PaddleOCRVL
            self.pipeline = PaddleOCRVL(pipeline_version='v1.5', device='cpu', cpu_threads=8,
                vl_rec_backend='llama-cpp-server', vl_rec_server_url=f'http://127.0.0.1:{port}/v1',
                vl_rec_api_model_name='paddle-ocr', vl_rec_max_concurrency=1, use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_chart_recognition=True, use_queues=False)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.lock.close()

    def convert(self, image, out):
        out.mkdir(parents=True)
        results = list(self.pipeline.predict(str(image), max_new_tokens=4096,
            temperature=0, markdown_ignore_labels=[]))
        if len(results) != 1:
            raise ValueError('Numero di pagine OCR inatteso.')
        result = results[0]
        result.save_to_markdown(save_path=str(out/'page.md'), pretty=False, show_formula_number=True)
        result.save_to_json(save_path=str(out/'page.json'))
        return (out/'page.md').read_text(encoding='utf-8')


def convert(folder):
    import pymupdf
    import pymupdf4llm
    from PIL import Image, ImageOps
    request = json.loads((folder/'request.json').read_text())
    source = folder/request['source']
    bundle = folder/'bundle'
    bundle.mkdir()
    started = time.monotonic()
    emit('phase', phase='document', message='Legge il documento…')
    pdf = pymupdf.open(source) if source.suffix == '.pdf' else None
    ocr = None
    try:
        if pdf and pdf.needs_pass:
            raise ValueError('Il PDF è protetto da password. Carica una copia sbloccata.')
        if pdf is None:
            with Image.open(source) as image:
                if image.width*image.height > 25_000_000:
                    raise ValueError('Immagine troppo grande: riducila a meno di 25 megapixel.')
                image.verify()
        count = len(pdf) if pdf is not None else 1
        selected = page_selection(request['pages'], count)
        parts, details = [], []
        for number, index in enumerate(selected, 1):
            page_started = time.monotonic()
            use_native = pdf is not None and not request['force_ocr'] and native_text(pdf[index])
            emit('phase', phase='document' if use_native else 'ocr',
                 message=f'Pagina {index+1} · {number}/{len(selected)}', page=index+1, completed=number-1, total=len(selected))
            if use_native:
                text = pymupdf4llm.to_markdown(pdf, pages=[index], use_ocr=False,
                                             show_progress=False, write_images=False)
                method = 'testo PDF'
            else:
                if ocr is None:
                    emit('needs_ocr')
                    if sys.stdin.readline().strip() != 'go':
                        raise InterruptedError('Conversione annullata.')
                    ocr = OCR()
                image_path = folder/f'page-{index+1}.png'
                if pdf is not None:
                    page = pdf[index]
                    # Bound page raster memory; recognition crops are handled
                    # by the official layout pipeline at their native detail.
                    zoom = min(200/72, 3500/max(page.rect.width, page.rect.height))
                    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(image_path)
                else:
                    with Image.open(source) as image:
                        if image.width*image.height > 25_000_000:
                            raise ValueError('Immagine troppo grande: riducila a meno di 25 megapixel.')
                        image = ImageOps.exif_transpose(image).convert('RGB')
                        image.thumbnail((3500, 3500))
                        image.save(image_path)
                subdir = bundle/'pages'/f'page-{index+1}'
                text = ocr.convert(image_path, subdir)
                # Preserve cropped figures in the downloadable Markdown bundle.
                text = text.replace('](imgs/', f'](pages/page-{index+1}/imgs/')
                method = 'PaddleOCR-VL-1.5'
            parts.append(f'<!-- Pagina {index+1} -->\n\n{text.strip()}')
            details.append(dict(page=index+1, method=method, seconds=round(time.monotonic()-page_started, 3)))
        text = '\n\n---\n\n'.join(parts)+'\n'
        if not any(char.isalnum() for char in '\n'.join(p.split('-->', 1)[-1] for p in parts)):
            raise ValueError('Non è stato riconosciuto testo nel documento.')
        (folder/'document.md').write_text(text, encoding='utf-8')
        (bundle/'document.md').write_text(text, encoding='utf-8')
        with zipfile.ZipFile(folder/'document.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in bundle.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle))
        result = dict(id=request['id'], name=Path(request['name']).stem+'.md',
                      pages=len(selected), source_pages=count, page_details=details,
                      seconds=round(time.monotonic()-started, 3), characters=len(text),
                      attachment_allowed=len(text)<=MAX_ATTACH,
                      text=text if len(text)<=MAX_ATTACH else '')
        (folder/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        if pdf is not None:
            pdf.close()
        if ocr is not None:
            ocr.close()
    # The recognizer must release CUDA before the browser can submit to light.
    emit('document_done', **result)


if __name__ == '__main__':
    try:
        with redirect_stdout(sys.stderr):
            convert(Path(sys.argv[1]))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit('error', message='La GPU è occupata da un altro processo. Attendi e riprova.'
             if isinstance(exc, BlockingIOError) else str(exc))
        raise SystemExit(1)
