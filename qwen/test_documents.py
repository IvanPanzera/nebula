"""Document boundaries and GPU handover, without loading model weights."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import documents
from web_server import Engine


class DocumentTests(unittest.TestCase):
    def test_page_ranges_do_not_truncate_or_reorder_silently(self):
        self.assertEqual(documents.page_selection('3,1-2,2', 5), [0, 1, 2])
        self.assertEqual(documents.page_selection('', 2), [0, 1])
        for value, count in [('0', 2), ('2-1', 3), ('1-4', 3), ('1,', 3), ('', 51), ('1-999999999', 999999999)]:
            with self.assertRaises(ValueError):
                documents.page_selection(value, count)

    def test_upload_and_download_paths_are_confined(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(documents, 'DOCUMENTS', Path(folder)):
            job = documents.create_job('../../report.pdf', b'%PDF-1.7\n')
            self.assertEqual(job.parent, Path(folder))
            self.assertEqual(json.loads((job/'request.json').read_text())['name'], 'report.pdf')
            self.assertEqual(documents.download(f'/api/documents/{job.name}/document.md'), job/'document.md')
            for path in ['/api/documents/../../secret/document.md', f'/api/documents/{job.name}/source.pdf']:
                with self.assertRaises(ValueError):
                    documents.download(path)
            for name, content in [('x.exe', b'MZ'), ('x.pdf', b'plain text'), ('x.png', b'')]:
                with self.assertRaises(ValueError):
                    documents.create_job(name, content)

    def run_converter(self, needs_gpu=False, cancel=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'qwen').mkdir()
            code = "import json, sys\nprint(json.dumps(dict(type='phase',phase='document')),flush=True)\n"
            if needs_gpu:
                code += "print(json.dumps(dict(type='needs_ocr')),flush=True)\nassert sys.stdin.readline().strip()=='go'\n"
            code += "print(json.dumps(dict(type='document_done',text='Testo')),flush=True)\n"
            (root/'qwen/document_worker.py').write_text(code)
            engine = Engine(command=['unused'])
            engine.loaded = True
            self.assertTrue(engine.reserve())
            events = []
            def close():
                engine.loaded = False
            with patch.object(documents, 'PROJECT', root), patch.object(documents, 'OCR_PYTHON', Path(sys.executable)), \
                    patch.object(engine, 'close', side_effect=close) as closer:
                for event in documents.stream_document(engine, root):
                    events.append(event)
                    if cancel:
                        engine.cancel_requested = True
                self.assertEqual(closer.call_count, int(needs_gpu and not cancel))
            self.assertFalse(engine.gate.locked())
            self.assertIsNone(engine.document_process)
            self.assertEqual(engine.loaded, not (needs_gpu and not cancel))
            return events

    def test_native_pdf_preserves_loaded_chat(self):
        self.assertEqual(self.run_converter()[-1]['type'], 'document_done')

    def test_ocr_unloads_chat_before_granting_gpu(self):
        self.assertEqual(self.run_converter(needs_gpu=True)[-1]['type'], 'document_done')

    def test_cancel_releases_gate_and_terminates_worker(self):
        self.assertEqual(self.run_converter(needs_gpu=True, cancel=True)[-1]['type'], 'document_cancelled')


if __name__ == '__main__':
    unittest.main()
