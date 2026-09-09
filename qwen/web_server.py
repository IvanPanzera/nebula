"""Loopback-only Qwen WebUI; Python standard library, no external web service."""
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
from urllib.parse import urlsplit, unquote
from urllib.request import urlopen
from config import CONTEXT, DRAFT_MIN, DRAFT_MAX
from model_registry import GPU_LOCK, get_model, public_models, worker_command
import documents

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'web'
LOG = ROOT.parent/'work/qwen/webui_engine.log'


class Engine:
    def __init__(self, command=None):
        self.custom_command = command
        self.model_id = 'flash-next'
        self.command = command or worker_command(self.model_id)
        self.process = None
        self.gate = threading.Lock()
        self.state_lock = threading.Lock()
        self.events = queue.Queue()
        self.phase = 'idle'
        self.loaded = False
        self.log_offset = 0
        self.last_error = None
        self.cancel_requested = False
        self.reader_thread = None
        self.document_process = None

    def reserve(self):
        with self.state_lock:
            if not self.gate.acquire(blocking=False):
                return False
            self.phase = 'preparing'
            self.last_error = None
            self.cancel_requested = False
            return True

    def external_busy(self):
        if self.loaded or self.gate.locked():
            return False
        path = GPU_LOCK
        try:
            with path.open('rb') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(stream, fcntl.LOCK_UN)
                except BlockingIOError:
                    return True
        except FileNotFoundError:
            pass
        return False

    def status(self):
        with self.state_lock:
            state = dict(phase=self.phase, busy=self.gate.locked(), model_loaded=self.loaded,
                         error=self.last_error, cancel_requested=self.cancel_requested)
        model = get_model(self.model_id)
        state.update(app='qwen-webui', context=model['context'], model=model['label'],
                     model_id=self.model_id, models=public_models(), speculative=model['speculative'],
                     external_busy=self.external_busy(), draft_policy='adaptive' if model['speculative'] else None,
                     draft_min=model.get('draft_min'), draft_max=model.get('draft_max'))
        state['documents_available'] = documents.available()
        if state['phase'] == 'loading' and LOG.exists():
            with LOG.open('rb') as stream:
                stream.seek(max(self.log_offset, LOG.stat().st_size-8192))
                tail = stream.read().decode('utf-8', errors='replace')
            matches = re.findall(r'routed RAM ([\d.]+) / ([\d.]+) GiB', tail)
            if matches:
                current, total = map(float, matches[-1])
                state['load_progress'] = min(99, round(100*current/total))
        return state

    def _reader(self, process, events):
        try:
            for line in process.stdout:
                try:
                    events.put(json.loads(line))
                except ValueError:
                    events.put(dict(type='error', message='Risposta non valida dal motore. Consulta work/qwen/webui_engine.log.'))
        finally:
            events.put(dict(type='process_exit'))

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            self.close()
        self.events = queue.Queue()
        self.loaded = False
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self.log_offset = LOG.stat().st_size if LOG.exists() else 0
        with LOG.open('ab') as log:
            self.process = subprocess.Popen(self.command, cwd=ROOT.parent, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8', bufsize=1,
                start_new_session=True)
        self.reader_thread = threading.Thread(target=self._reader, args=(self.process, self.events), daemon=True)
        self.reader_thread.start()

    def stop(self):
        with self.state_lock:
            self.cancel_requested = self.gate.locked()
            process = self.process
            can_signal = self.phase in ('loading', 'prefill', 'thinking', 'generating')
        if self.document_process is not None:
            documents.terminate(self.document_process)
            return
        # A new Python child installs its handler before its first phase event.
        # Sending SIGUSR1 earlier would kill it with the default signal action.
        if process is not None and process.poll() is None and self.cancel_requested and can_signal:
            process.send_signal(signal.SIGUSR1)

    def stream(self, request):
        # The handler reserves gate before responding. One stream owns stdin and
        # drains all worker events, even when a browser disconnects mid-response.
        try:
            model_id = request.get('model', 'flash-next')
            get_model(model_id)
            if model_id != self.model_id:
                self.close()
                self.model_id = model_id
                self.command = self.custom_command or worker_command(model_id)
            self.start()
            self.process.stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
            self.process.stdin.flush()
            yield dict(type='phase', phase='preparing')
            while True:
                try:
                    event = self.events.get(timeout=1)
                except queue.Empty:
                    yield dict(type='heartbeat')
                    continue
                kind = event.get('type')
                if kind == 'done' and event.get('metrics'):
                    event['metrics'].update(model_id=self.model_id, model=get_model(self.model_id)['label'],
                                            speculative=get_model(self.model_id)['speculative'])
                with self.state_lock:
                    if kind == 'phase':
                        self.phase = event['phase']
                        self.loaded = event.get('model_loaded', self.loaded)
                    elif kind == 'error':
                        self.phase = 'error'
                        self.last_error = event['message']
                        self.loaded = event.get('model_loaded', self.loaded)
                    elif kind == 'done':
                        self.loaded = event.get('model_loaded', True)
                        self.phase = 'ready' if self.loaded else 'idle'
                    elif kind == 'process_exit':
                        self.phase = 'error'
                        self.loaded = False
                        event = dict(type='error', message='Il processo Qwen si è chiuso. Consulta work/qwen/webui_engine.log e riprova.')
                if kind == 'phase' and self.cancel_requested:
                    self.stop()
                yield event
                if kind in ('done', 'error', 'process_exit'):
                    return
        except (OSError, ValueError) as exc:
            with self.state_lock:
                self.phase = 'error'
                self.last_error = str(exc)
            yield dict(type='error', message=str(exc))
        finally:
            self.gate.release()

    def close(self):
        process = self.process
        if process is not None:
            # Standalone workers spawn llama-server. Terminate the entire
            # session even if the Python bridge has already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self.process = None
        self.loaded = False
        self.reader_thread = None


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError('Richiesta non valida.')
    messages = data.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 500:
        raise ValueError('La conversazione non è valida.')
    for i, item in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        if not isinstance(item, dict) or item.get('role') != role or not isinstance(item.get('content'), str):
            raise ValueError('Ordine o contenuto dei messaggi non valido.')
        if len(item['content']) > 300000:
            raise ValueError('Il messaggio è troppo lungo.')
        if 'reasoning_content' in item and (role != 'assistant' or
                not isinstance(item['reasoning_content'], str) or len(item['reasoning_content']) > 300000):
            raise ValueError('Contenuto del ragionamento non valido.')
    if len(messages) % 2 != 1 or not messages[-1]['content'].strip():
        raise ValueError('Scrivi un messaggio prima di inviarlo.')
    limit = data.get('max_tokens', 2048)
    if type(limit) is not int or not 1 <= limit <= 2048:
        raise ValueError('Il limite di risposta deve essere tra 1 e 2048 token.')
    thinking = data.get('thinking', False)
    if type(thinking) is not bool:
        raise ValueError('Thinking deve essere acceso o spento.')
    model = data.get('model', 'flash-next')
    if not isinstance(model, str):
        raise ValueError('Modello non valido.')
    get_model(model)
    clean = []
    for message in messages:
        item = dict(role=message['role'], content=message['content'])
        if 'reasoning_content' in message:
            item['reasoning_content'] = message['reasoning_content']
        clean.append(item)
    return dict(messages=clean, max_tokens=limit, thinking=thinking, model=model)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def allowed(self):
        port = self.server.server_port
        hosts = {f'localhost:{port}', f'127.0.0.1:{port}'}
        if self.headers.get('Host') not in hosts:
            return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {'http://'+host for host in hosts}

    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.reply(403, dict(error='Host non consentito.'))
        path = urlsplit(self.path).path
        if path == '/api/status':
            return self.reply(200, self.server.engine.status())
        if path.startswith('/api/documents/'):
            try:
                file = documents.download(path)
                return self.reply(200, file.read_bytes(),
                    'application/zip' if file.suffix == '.zip' else 'text/markdown; charset=utf-8')
            except (ValueError, FileNotFoundError):
                return self.reply(404, dict(error='Documento non trovato.'))
        files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                 '/style.css': ('style.css', 'text/css'), '/qwen-logo.png': ('qwen-logo.png', 'image/png')}
        if path not in files:
            return self.reply(404, dict(error='Risorsa non trovata.'))
        name, mime = files[path]
        self.reply(200, (ASSETS/name).read_bytes(), mime+('; charset=utf-8' if mime.startswith('text/') else ''))

    def do_POST(self):
        if not self.allowed() or self.headers.get('X-Qwen-Client') != 'webui':
            return self.reply(403, dict(error='Richiesta non consentita.'))
        path = urlsplit(self.path).path
        if path == '/api/stop':
            self.server.engine.stop()
            return self.reply(200, dict(stopping=True))
        is_document = path == '/api/documents'
        if path != '/api/chat' and not is_document:
            return self.reply(404, dict(error='Risorsa non trovata.'))
        if is_document and not documents.available():
            return self.reply(503, dict(error='Il modulo OCR non è ancora installato.'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= (documents.MAX_UPLOAD if is_document else 2_000_000):
                raise ValueError('Dimensione della richiesta non valida.')
            self.connection.settimeout(60 if is_document else 15)
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError('Caricamento incompleto.')
            if not is_document:
                request = validate_request(json.loads(data))
        except (ValueError, OSError) as exc:
            return self.reply(400, dict(error=str(exc)))
        if not self.server.engine.reserve():
            return self.reply(409, dict(error='Qwen sta già elaborando una risposta. Attendi oppure interrompila.'))
        if is_document:
            try:
                folder = documents.create_job(unquote(self.headers.get('X-Document-Name', '')),
                    data, self.headers.get('X-Document-Pages', ''), self.headers.get('X-Document-OCR') == 'true')
            except (OSError, ValueError) as exc:
                self.server.engine.gate.release()
                return self.reply(400, dict(error=str(exc)))
        self.connection.settimeout(15)
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
        except OSError:
            self.server.engine.gate.release()
            return
        disconnected = False
        try:
            stream = documents.stream_document(self.server.engine, folder) if is_document else self.server.engine.stream(request)
            for event in stream:
                if disconnected:
                    continue
                try:
                    self.wfile.write((json.dumps(event, ensure_ascii=False)+'\n').encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    disconnected = True
                    self.server.engine.stop()
        finally:
            self.close_connection = True


def make_server(port=8090, engine=None):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.engine = engine or Engine()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    try:
        server = make_server(args.port)
    except OSError as exc:
        if args.open_browser:
            try:
                with urlopen(f'http://127.0.0.1:{args.port}/api/status', timeout=2) as response:
                    existing = json.load(response)
                if existing.get('app') == 'qwen-webui':
                    print(f'WebUI già attiva: http://localhost:{args.port}', flush=True)
                    open_browser(args.port)
                    return
            except (OSError, ValueError):
                pass
        raise SystemExit(f'Impossibile aprire la WebUI sulla porta {args.port}: {exc}')
    print(f'Qwen WebUI: http://localhost:{server.server_port}', flush=True)
    print('Il modello si carica al primo messaggio. Ctrl+C chiude il server e libera la memoria.', flush=True)
    if args.open_browser:
        open_browser(server.server_port)
    def shutdown(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGHUP, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        documents.terminate(server.engine.document_process)
        server.engine.close()
        server.server_close()


def open_browser(port):
    try:
        subprocess.Popen(['cmd.exe', '/d', '/c', 'start', '', f'http://localhost:{port}'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        print(f'Apri http://localhost:{port} nel browser.', flush=True)


if __name__ == '__main__':
    main()
