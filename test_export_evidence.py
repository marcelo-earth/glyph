import json
from pathlib import Path
import tempfile
import unittest
from export_evidence import export
from experiment import digest

class EvidenceTests(unittest.TestCase):
    def test_export_keeps_source_totals_and_original_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.json'
            result={'status':'complete','history':[{'validation':{'utf8_bytes':5,'nll_nats':3.,
                'documents':[{'block':'local detail'}],
                'source_documents':[{'utf8_bytes':2,'nll_nats':1.},{'utf8_bytes':3,'nll_nats':2.}]}}]}
            path.write_text(json.dumps(result));exported=export(path)
            self.assertEqual(exported['evidence_export']['original_metrics_sha256'],digest(path))
            self.assertNotIn('documents',exported['history'][0]['validation'])
            self.assertEqual(exported['history'][0]['validation']['source_documents'],result['history'][0]['validation']['source_documents'])
            result['history'][0]['validation']['utf8_bytes']=6;path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):export(path)

if __name__=='__main__':unittest.main()
