import tempfile
from pathlib import Path
import unittest
import numpy as np
from experiment import TrainingData
from exposure_baseline import replay_counts,score_counts
from test_experiment import tokenizer_fixture
from evaluation import prepare_evaluation

class ExposureBaselineTests(unittest.TestCase):
    def test_replay_wrap_and_byte_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok=tokenizer_fixture(Path(tmp)/'tok.json')
            data=TrainingData(tok,['abcdef🧬'],16,'raw')
            counts,exposure=replay_counts(data,5,2,42,260)
            self.assertEqual(int(counts.sum()),100)
            self.assertEqual(exposure['raw_bytes'],100)
            self.assertEqual(exposure['target_tokens'],100)
            self.assertEqual(exposure['raw_chars'],70)
            self.assertTrue(np.array_equal(counts,replay_counts(data,5,2,42,260)[0]))
            prepared=prepare_evaluation(tok,['abcdef🧬'])
            score=score_counts(np.zeros(260,dtype=np.int64),prepared)
            self.assertAlmostEqual(score['bits_per_utf8_byte'],np.log2(260))

if __name__=='__main__':unittest.main()
