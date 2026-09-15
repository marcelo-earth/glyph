import math
import unittest
from analyze_matrix import group_bootstrap

class AnalysisTests(unittest.TestCase):
    def test_grouping_preserves_paired_ratio(self):
        fit=[{'sha256':str(i),'utf8_bytes':10,'nll_nats':10*math.log(2)} for i in range(4)]
        general=[dict(x,nll_nats=20*math.log(2)) for x in fit]
        out=group_bootstrap(fit,general,['repo-a','repo-a','repo-b','repo-b'],100)
        self.assertEqual(out['group_count'],2)
        for value in out['fit_minus_general_bpb_95_interval']:self.assertAlmostEqual(value,-1)
        general[0]['sha256']='other'
        with self.assertRaises(ValueError):group_bootstrap(fit,general,['a']*4)

if __name__=='__main__':unittest.main()
