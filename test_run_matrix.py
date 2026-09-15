import copy
import unittest
from run_matrix import check_pair


def fixture(label):
    return {'status':'complete','config':{'label':label,'tokenizer_sha256':label,'regime':'raw','steps':5},
            'parameters':100,'initial_weights_sha256':'shared',
            'exposure':{'raw_bytes':12,'raw_chars':11,'padded_positions':32},
            'history':[{'step':5,'exposure':{'raw_bytes':12,'raw_chars':11,'padded_positions':32},
                        'validation':{'utf8_bytes':10,'characters':9,'source_documents':[{'sha256':'document'}]}}]}

class PairTests(unittest.TestCase):
    def test_valid_pair(self):self.assertTrue(check_pair(fixture('fit'),fixture('general'))['passed'])
    def test_rejects_confounds_and_incomplete_evidence(self):
        edits=[lambda b:b['config'].update(steps=6),lambda b:b.update(status='running'),
               lambda b:b.update(initial_weights_sha256='different'),
               lambda b:b['exposure'].update(raw_bytes=13),lambda b:b['history'].append(b['history'][0]),
               lambda b:b['history'][0]['validation']['source_documents'][0].update(sha256='different'),
               lambda b:b['history'][0]['exposure'].update(raw_bytes=10)]
        for edit in edits:
            b=fixture('general');edit(b)
            with self.assertRaises(ValueError):check_pair(fixture('fit'),b)

if __name__=='__main__':unittest.main()
