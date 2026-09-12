"""Storage policy and complete-ranking validation; no model or GPU allocation."""
import json
from pathlib import Path
import tempfile
import unittest
from storage import DEFAULT, ROOT, load_storage, load_ranking

class StorageTests(unittest.TestCase):
    def test_official_stays_full_ram(self):
        self.assertEqual(load_storage(),DEFAULT)

    def test_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'storage.json'
            for ram in (27,124,383,485,512):
                for ngram in ('ram','ssd'):
                    p.write_text(json.dumps(dict(ram_experts_per_layer=ram,ngram=ngram)))
                    self.assertEqual(load_storage(p)['ram_experts_per_layer'],ram)
            for bad in [dict(ram_experts_per_layer=26),dict(ram_experts_per_layer=513),
                        dict(ram_experts_per_layer=27.1),dict(ram_experts_per_layer=True),
                        dict(ngram='auto'),dict(buffer_mib=1),dict(io_threads=0),dict(unrecognized=1)]:
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_storage(p)

    def test_complete_ranking(self):
        hot=[e for row in json.loads((ROOT/'hotlist.json').read_text())['experts'] for e in row]
        ranking=load_ranking(hot)
        self.assertEqual(len(ranking),48*512)
        for l in range(48):self.assertEqual(sorted(ranking[l*512:(l+1)*512]),list(range(512)))
        hot[0],hot[1]=hot[1],hot[0]
        with self.assertRaises(ValueError):load_ranking(hot)

if __name__=='__main__':unittest.main()
