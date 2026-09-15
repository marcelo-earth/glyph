import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from corpus_snapshot import prepare_snapshot
from experiment import raw_blocks, TrainingData, train_run, digest, atomic_json


def tokenizer_fixture(path):
    tok = Tokenizer(BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(['hello world \n 🧬 café'], BpeTrainer(vocab_size=260,
        initial_alphabet=ByteLevel.alphabet(),special_tokens=['[UNK]','[PAD]','[BOS]','[EOS]'],show_progress=False))
    tok.save(str(path)); return tok


class ExperimentTests(unittest.TestCase):
    def test_raw_blocks_preserve_unicode_and_parent_identity(self):
        texts=['a🧬bc漢é\n','short','']
        blocks,parents=raw_blocks(texts,5)
        self.assertTrue(all(len(t.encode('utf-8'))<=5 for t in blocks))
        for i,t in enumerate(texts): self.assertEqual(''.join(b for b,p in zip(blocks,parents) if p==i),t)

    def test_exposure_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok=tokenizer_fixture(Path(tmp)/'tok.json')
            texts=['abc🧬def漢é\n'*4,'second doc']
            raw=TrainingData(tok,texts,16,'raw')
            x,y,counts=raw.batch(list(range(raw.size)))
            self.assertEqual(counts['raw_bytes'],sum(len(t.encode('utf8')) for t in texts))
            self.assertEqual(counts['raw_chars'],sum(map(len,texts)))
            self.assertEqual(x.shape,y.shape)
            self.assertEqual(x.shape[1],16)
            token=TrainingData(tok,texts,16,'token')
            _,y,c=token.batch(list(range(token.size)))
            self.assertEqual(c['target_tokens'],token.size*16)
            self.assertEqual(int((y==-100).sum()),0)
            self.assertLessEqual(c['raw_bytes'],sum(len(t.encode('utf8')) for t in texts))

    def test_packed_unicode_counts_actual_bytes_not_offset_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok=tokenizer_fixture(Path(tmp)/'tok.json')
            data=TrainingData(tok,['abc🧬x'],4,'token')
            self.assertEqual(data.batch([0])[2]['raw_bytes'],4)
            self.assertEqual(data.batch([1])[2]['raw_bytes'],4)

    def test_checkpoint_resume_matches_uninterrupted_cpu(self):
        self.check_resume('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_checkpoint_resume_matches_uninterrupted_mps(self):
        self.check_resume('mps')

    def check_resume(self, device):
        import evaluation
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); tokpath=root/'fit.json';tokenizer_fixture(tokpath)
            (root/'general.json').write_bytes(tokpath.read_bytes())
            pools={'target':[f'hello world {i} abcdefghijklmnopqrstuvwxyz'*2 for i in range(20)],
                   'general':[f'different text {i} qwertyuiop'*2 for i in range(20)]}
            prepare_snapshot(root/'snapshot',corpus='target',general_corpus='general',
                max_docs=20,val_docs=2,test_docs=2,tokenizer_chars=100,loader=lambda n,**kw:pools[n],
                source_specs={'target':{'path':'offline-target'},'general':{'path':'offline-general'}})
            atomic_json(root/'manifest.json',{'snapshot_sha256':digest(root/'snapshot/manifest.json'),
                'vocab_size':260,'fit':{'sha256':digest(tokpath),'training_chars':100},
                'general':{'sha256':digest(root/'general.json'),'training_chars':100}})
            kwargs=dict(snapshot_dir=root/'snapshot',tokenizer_path=tokpath,label='fit',regime='raw',
                        dim=16,layers=1,heads=2,seq_len=16,batch_size=64,steps=4,eval_every=2,val_docs=1,device=device)
            full=train_run(out_dir=root/'full',**kwargs)
            calls=0;original=evaluation.evaluate_documents
            def interrupt(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('simulated interruption')
                return original(*a,**kw)
            with patch('evaluation.evaluate_documents',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    train_run(out_dir=root/'resume',**kwargs)
            resumed=train_run(out_dir=root/'resume',resume=True,**kwargs)
            a=torch.load(root/'full/checkpoint.pt',weights_only=False)
            b=torch.load(root/'resume/checkpoint.pt',weights_only=False)
            for key in a['model']:
                if device == 'cpu':
                    self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
                else:
                    # Repeated MPS runs show ~1e-8 FP32 variation; do not claim
                    # bitwise accelerator determinism. Missing RNG/optimizer
                    # restoration produces much larger errors than this bound.
                    if key.endswith('in_proj_bias'):
                        left_parts=a['model'][key].chunk(3);right_parts=b['model'][key].chunk(3)
                        for index,(lhs,rhs) in enumerate(zip(left_parts,right_parts)):
                            # Key bias adds the same logit shift to every key;
                            # softmax cancels it. Near-zero roundoff gradients
                            # are amplified by Adam, measured up to 2.4e-6.
                            torch.testing.assert_close(lhs,rhs,rtol=1e-6,atol=1e-5 if index==1 else 1e-7)
                    else:
                        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=1e-6,atol=1e-7)
            self.assertEqual(full['exposure'],resumed['exposure'])
            if device == 'cpu':
                self.assertEqual(full['history'][-1]['validation'],resumed['history'][-1]['validation'])
            else:
                left=full['history'][-1]['validation'];right=resumed['history'][-1]['validation']
                self.assertAlmostEqual(left['bits_per_utf8_byte'],right['bits_per_utf8_byte'],delta=1e-6)
                for lhs,rhs in zip(left['source_documents'],right['source_documents']):
                    self.assertEqual(lhs['sha256'],rhs['sha256'])
                    self.assertEqual(lhs['scored_tokens'],rhs['scored_tokens'])
                    self.assertEqual(lhs['utf8_bytes'],rhs['utf8_bytes'])
            self.assertEqual(full['initial_weights_sha256'],resumed['initial_weights_sha256'])
            self.assertEqual(train_run(out_dir=root/'full',**kwargs)['status'],'complete')
            with self.assertRaisesRegex(ValueError,'another configuration'):
                train_run(out_dir=root/'full',**dict(kwargs,steps=5))

if __name__=='__main__':unittest.main()
