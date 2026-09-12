import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from profiles import GIB, ROOT, table, select, configure, read_runtime, validate_runtime


class ProfilesTests(unittest.TestCase):
    def test_document_and_floor_selection(self):
        self.assertEqual(len(table()), 16)
        self.assertEqual(select(20*GIB, 80*GIB), select(16*GIB, 64*GIB))
        self.assertEqual(select(12282*2**20, 128*GIB)['vram_gib'], 12)
        self.assertEqual(select(48*GIB, 256*GIB), select(24*GIB, 128*GIB))
        for gpu, ram in ((7, 32), (12, 31)):
            with self.assertRaises(ValueError): select(gpu*GIB, ram*GIB)
        for p in table():
            self.assertEqual(select(p['vram_gib']*GIB, p['ram_gib']*GIB), p)
            self.assertEqual(p['ram_experts']+p['ssd_experts'], 512)
            self.assertGreaterEqual(p['ram_experts'], p['gpu_experts'])
            self.assertEqual(p['ngram'], 'ram' if p['ram_gib']==128 else 'ssd')
        self.assertEqual(select(24*GIB, 32*GIB)['prefill'], 8182)

    def test_all_profiles_generate_hotlists_and_storage(self):
        ranking=json.loads((ROOT/'expert_ranking.json').read_text())['generation']
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);shutil.copy2(ROOT/'expert_ranking.json',root)
            index=root/'model.index';index.write_text('verified test index')
            for profile in table():
                value=configure(root,index,profile,'GPU-test',8)
                self.assertEqual(read_runtime(root),value)
                hot=json.loads((root/'hotlist.json').read_text())
                self.assertEqual(hot['experts'],[r[:profile['gpu_experts']] for r in ranking])
                self.assertEqual(hot['model_index_sha256'],hashlib.sha256(index.read_bytes()).hexdigest())
                storage=json.loads((root/'storage.json').read_text())
                self.assertEqual(storage['ram_experts_per_layer'],profile['ram_experts'])
                value['profile']['context']+=1
                with self.assertRaises(ValueError): validate_runtime(value)

if __name__=='__main__': unittest.main()
