"""Analytical and model-level checks for evaluation accounting; no downloads."""

import math
import unittest
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from evaluation import evaluate_documents, prepare_evaluation, evaluate_token_baselines
from model import GlyphGPT


class ByteTokenizer:
    """A lossless byte tokenizer; multi-byte Unicode exercises denominators."""
    def token_to_id(self, name):
        return {"[BOS]": 256, "[PAD]": 257, "[EOS]": 258}.get(name)

    def get_vocab_size(self):
        return 259

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")


class UniformModel(torch.nn.Module):
    seq_len = 4

    def forward(self, x):
        return torch.zeros((*x.shape, 259), device=x.device)


class RecordingModel(UniformModel):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.extend(x.tolist())
        return super().forward(x)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def test_uniform_loss_unicode_and_complete_remainder(self):
        texts = ["a", "café", "🦉z", "", "123456789"]
        result = evaluate_documents(UniformModel(), self.tokenizer, texts,
                                    context_length=4, batch_size=3)
        byte_count = sum(len(t.encode("utf-8")) for t in texts)
        char_count = sum(map(len, texts))
        self.assertEqual(result["scored_tokens"], byte_count)
        self.assertEqual(result["characters"], char_count)
        self.assertEqual(result["utf8_bytes"], byte_count)
        self.assertAlmostEqual(result["nll_nats"], byte_count * math.log(259), places=5)
        self.assertAlmostEqual(result["bits_per_utf8_byte"], math.log2(259), places=5)
        self.assertAlmostEqual(result["bits_per_character"],
                               byte_count * math.log2(259) / char_count, places=5)
        for text, document in zip(texts, result["documents"]):
            self.assertEqual(document["scored_tokens"], len(text.encode("utf-8")))
        self.assertEqual(result["documents"][3]["nll_nats"], 0)

    def test_first_token_bos_document_boundaries_and_chunk_shift(self):
        model = RecordingModel()
        result = evaluate_documents(model, self.tokenizer, ["abcdef", "z"],
                                    context_length=3, batch_size=1)
        self.assertEqual(model.inputs, [[256, 97, 98], [99, 100, 101], [256]])
        self.assertEqual(result["scored_tokens"], 7)

    def test_rolling_uses_all_available_context_and_scores_once(self):
        model = RecordingModel()
        result = evaluate_documents(model, self.tokenizer, ["abcde"],
                                    context_length=3, batch_size=1,
                                    context_policy="rolling")
        self.assertEqual(model.inputs, [[256], [256, 97], [256, 97, 98],
                                        [97, 98, 99], [98, 99, 100]])
        self.assertEqual(result["scored_tokens"], 5)
        self.assertAlmostEqual(result["nll_nats"], 5 * math.log(259), places=5)

    def test_eos_explicitly_adds_one_target_per_document(self):
        result = evaluate_documents(UniformModel(), self.tokenizer, ["a", ""],
                                    include_eos=True)
        self.assertEqual(result["text_tokens"], 1)
        self.assertEqual(result["scored_tokens"], 3)
        self.assertTrue(result["include_eos"])
        self.assertAlmostEqual(result["bits_per_character"], 3 * math.log2(259), places=5)

    def test_padding_batch_size_and_cached_encoding_invariance(self):
        torch.manual_seed(17)
        model = GlyphGPT(259, dim=8, n_heads=2, n_layers=1, seq_len=4, dropout=0.2)
        prepared = prepare_evaluation(self.tokenizer, ["abcdef", "a", "xyz", "🦉"])
        for policy in ["chunks", "rolling"]:
            alone = evaluate_documents(model, prepared=prepared, batch_size=1,
                                       context_policy=policy)
            padded = evaluate_documents(model, prepared=prepared, batch_size=5,
                                        context_policy=policy)
            self.assertTrue(model.training)
            self.assertAlmostEqual(alone["nll_nats"], padded["nll_nats"], places=5)
            for a, b in zip(alone["documents"], padded["documents"]):
                self.assertAlmostEqual(a["nll_nats"], b["nll_nats"], places=5)

    def test_length_one_scores_every_token(self):
        result = evaluate_documents(UniformModel(), self.tokenizer, ["abcdef", "g"],
                                    context_length=1, batch_size=100)
        self.assertEqual(result["window_count"], 7)
        self.assertEqual(result["scored_tokens"], 7)

    def test_empty_data_and_invalid_settings_fail_loudly(self):
        for texts in [[], [""], ["", ""]]:
            with self.assertRaises(ValueError):
                prepare_evaluation(self.tokenizer, texts)
        for kwargs in [{"batch_size": 0}, {"context_length": 0},
                       {"context_length": 5}, {"context_policy": "bad"}]:
            with self.assertRaises(ValueError):
                evaluate_documents(UniformModel(), self.tokenizer, ["a"], **kwargs)

    def test_nonfinite_loss_rejected_and_mode_restored(self):
        class BrokenModel(UniformModel):
            def forward(self, x):
                return super().forward(x) * float("nan")
        model = BrokenModel()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluate_documents(model, self.tokenizer, ["a"])
        self.assertTrue(model.training)

    def test_lossy_tokenizer_rejected(self):
        class LossyTokenizer(ByteTokenizer):
            def decode(self, ids, skip_special_tokens=False):
                return "incorrect"
        with self.assertRaisesRegex(ValueError, "not lossless"):
            prepare_evaluation(LossyTokenizer(), ["a"])

    def test_actual_byte_bpe_keeps_literal_special_spellings_as_text(self):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        spellings = ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]
        tokenizer.train_from_iterator(
            ["Ordinary text with punctuation [] and Unicode café 🦉."],
            trainer=trainers.BpeTrainer(vocab_size=300, special_tokens=spellings,
                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                       show_progress=False),
        )
        literal = "".join(spellings)
        controls = {tokenizer.token_to_id(text) for text in spellings}
        # Demonstrate that the otherwise-default behavior really is hazardous.
        self.assertEqual(set(tokenizer.encode(literal).ids), controls)
        prepared = prepare_evaluation(tokenizer, [literal])
        ids = prepared.documents[0].token_ids
        self.assertFalse(set(ids) & controls)
        self.assertEqual(tokenizer.decode(list(ids), skip_special_tokens=False), literal)
        prepared_eos = prepare_evaluation(tokenizer, [literal], include_eos=True)
        self.assertEqual(prepared_eos.documents[0].token_ids,
                         ids + (tokenizer.token_to_id("[EOS]"),))
        # Baseline training takes the same safe content path even if reset by a
        # caller between preparing validation data and fitting the unigram.
        tokenizer.encode_special_tokens = False
        baselines = evaluate_token_baselines(tokenizer, [literal], prepared=prepared)
        self.assertEqual(baselines["unigram"]["training_scored_tokens"], len(ids))
        self.assertTrue(tokenizer.encode_special_tokens)

    def test_unigram_smoothing_matches_analytic_loss(self):
        baselines = evaluate_token_baselines(self.tokenizer, ["aaa", "b"], ["az"])
        # Training total 4, vocabulary 259, add-one denominator 263.
        expected = -math.log(4 / 263) - math.log(1 / 263)
        self.assertAlmostEqual(baselines["unigram"]["nll_nats"], expected)
        self.assertAlmostEqual(baselines["uniform"]["nll_nats"], 2 * math.log(259))
        self.assertEqual(baselines["unigram"]["training_scored_tokens"], 4)
        self.assertEqual(baselines["unigram"]["characters"], 2)

    def test_baseline_eos_and_cached_targets(self):
        prepared = prepare_evaluation(self.tokenizer, ["a", ""], include_eos=True)
        baselines = evaluate_token_baselines(self.tokenizer, ["aa", ""], prepared=prepared)
        # a count 2, EOS count 2; both have add-one probability 3/263.
        self.assertAlmostEqual(baselines["unigram"]["nll_nats"], -3 * math.log(3 / 263))
        self.assertEqual(baselines["unigram"]["scored_tokens"], 3)
        for alpha in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                evaluate_token_baselines(self.tokenizer, ["a"], prepared=prepared, alpha=alpha)


if __name__ == "__main__":
    unittest.main()
