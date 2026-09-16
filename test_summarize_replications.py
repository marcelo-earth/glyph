import math
import unittest
from summarize_replications import seed_summary


def score(bpb):
    return {'bits_per_utf8_byte':bpb,'source_documents':[{'sha256':'a','utf8_bytes':10,'nll_nats':bpb*10*math.log(2)}]}

class SeedSummaryTests(unittest.TestCase):
    def test_seed_and_source_uncertainty_are_separate(self):
        result=seed_summary([(42,score(2),score(3)),(43,score(2.5),score(3)),(44,score(3),score(3))],['repo'])
        self.assertEqual(result['seed_count'],3)
        self.assertEqual(result['fit_mean_bpb'],2.5)
        self.assertEqual(result['paired_mean_fit_minus_general_bpb'],-.5)
        self.assertEqual(result['paired_difference_sample_sd'],.5)
        # One identical source group has no resampling spread despite seed spread.
        for v in result['source_bootstrap_of_seed_mean']['fit_minus_general_bpb_95_interval']:
            self.assertAlmostEqual(v,-.5)
    def test_wrong_source_pair_fails(self):
        wrong=score(3);wrong['source_documents'][0]['sha256']='different'
        with self.assertRaises(ValueError):seed_summary([(42,score(2),wrong)],['repo'])

if __name__=='__main__':unittest.main()
