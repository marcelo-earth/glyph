"""CPU-only checkpoint/re-evaluation integrity checks; no external data."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from corpus_snapshot import prepare_snapshot
from evaluate_checkpoint import _identity, evaluate_checkpoint
from experiment import atomic_json, digest, train_run
from test_experiment import tokenizer_fixture


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer_path = self.root / "fit.json"
        self.snapshot = self.root / "snapshot"
        self.run = self.root / "training"
        self.checkpoint = self.run / "checkpoint.pt"
        self.output = self.root / "evaluation.json"
        pools = {"target": [f"Source {i}: café, 🦉 and words. " * 3 for i in range(12)],
                 "general": [f"Other {i}: alphabet and numbers. " * 3 for i in range(12)]}
        with contextlib.redirect_stdout(io.StringIO()):
            tokenizer_fixture(self.tokenizer_path)
            (self.root / "general.json").write_bytes(self.tokenizer_path.read_bytes())
            prepare_snapshot(self.snapshot, corpus="target", general_corpus="general",
                             max_docs=12, val_docs=3, test_docs=2, tokenizer_chars=100,
                             loader=lambda name, **_: pools[name],
                             source_specs={"target": {}, "general": {}})
            atomic_json(self.root / "manifest.json", {
                "snapshot_sha256": digest(self.snapshot / "manifest.json"), "vocab_size": 260,
                "fit": {"sha256": digest(self.tokenizer_path), "training_chars": 100},
                "general": {"sha256": digest(self.root / "general.json"), "training_chars": 100},
            })
            self.training = train_run(self.snapshot, self.tokenizer_path, self.run,
                                      label="fit", regime="raw", dim=8, layers=1, heads=2,
                                      seq_len=16, batch_size=4, steps=2, eval_every=2,
                                      val_docs=2, device="cpu")

    def evaluate(self, **kwargs):
        return evaluate_checkpoint(self.snapshot, self.tokenizer_path, self.checkpoint,
                                   self.output, device="cpu", **kwargs)

    def test_reloaded_endpoint_equals_training_evaluation_and_caches(self):
        result = self.evaluate(max_docs=2)
        self.assertEqual(result["scores"], self.training["history"][-1]["validation"])
        self.assertEqual(result["checkpoint_step"], 2)
        self.assertEqual(result, self.evaluate(max_docs=2))
        with self.assertRaisesRegex(ValueError, "another checkpoint or evaluation protocol"):
            self.evaluate(max_docs=1)

    def test_full_validation_blocks_preserve_all_source_characters(self):
        result = self.evaluate(block_bytes=7)
        scores = result["scores"]
        self.assertEqual(len(scores["source_documents"]), 3)
        self.assertEqual(scores["utf8_bytes"], sum(d["utf8_bytes"] for d in scores["source_documents"]))
        self.assertEqual(scores["characters"], sum(d["characters"] for d in scores["source_documents"]))
        self.assertTrue(all(d["utf8_bytes"] <= 7 for d in scores["documents"]))
        self.assertEqual(result["identity"]["protocol"]["source_selection"], "entire split")

    def test_uniform_checkpoint_has_analytic_nll_for_token_and_raw_modes(self):
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        checkpoint["model"] = {key: torch.zeros_like(value) for key, value in checkpoint["model"].items()}
        torch.save(checkpoint, self.checkpoint)
        for mode in ("raw", "token"):
            for policy in ("chunks", "rolling"):
                self.output = self.root / f"{mode}-{policy}.json"
                scores = self.evaluate(mode=mode, context_policy=policy)["scores"]
                self.assertAlmostEqual(scores["nats_per_token"], math.log(260), places=6)
                self.assertAlmostEqual(scores["nll_nats"], scores["scored_tokens"] * math.log(260),
                                       delta=scores["scored_tokens"] * 3e-7)
                self.assertEqual(len(scores["source_documents"]), 3)

    def test_snapshot_tokenizer_and_source_hash_mismatch_guards(self):
        original = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        for key, message in (("snapshot_sha256", "snapshot checksum"),
                             ("tokenizer_sha256", "tokenizer checksum")):
            changed = dict(original)
            changed["config"] = dict(original["config"], **{key: "invalid"})
            changed["config_hash"] = _identity(changed["config"])
            torch.save(changed, self.checkpoint)
            with self.assertRaisesRegex(ValueError, message):
                self.evaluate()
        for source in ("model.py", "evaluation.py", "experiment.py", "corpus_snapshot.py"):
            changed = dict(original)
            changed["config"] = dict(original["config"])
            changed["config"]["implementation_sha256"] = dict(original["config"]["implementation_sha256"])
            changed["config"]["implementation_sha256"][source] = "invalid"
            changed["config_hash"] = _identity(changed["config"])
            torch.save(changed, self.checkpoint)
            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                self.evaluate()

    def test_checkpoint_and_cached_payload_tampering_rejected(self):
        result = self.evaluate()
        result["scores"]["nll_nats"] += 1
        self.output.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "invalid payload checksum"):
            self.evaluate()
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        checkpoint["config"]["seed"] += 1
        torch.save(checkpoint, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "configuration checksum mismatch"):
            self.evaluate()

    def test_validation_evaluation_never_opens_sealed_test(self):
        original = Path.read_bytes

        def guarded(path):
            if path.name == "test.jsonl":
                raise AssertionError("sealed test opened")
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            result = self.evaluate(max_docs=1)
        self.assertEqual(result["identity"]["protocol"]["split"], "validation")

    def test_invalid_protocol_rejected(self):
        for kwargs in ({"max_docs": 0}, {"block_bytes": 17}, {"block_bytes": 3},
                       {"mode": "token", "block_bytes": 8}, {"batch_size": 0},
                       {"context_policy": "unknown"}, {"split": "train"}):
            with self.assertRaises(ValueError):
                self.evaluate(**kwargs)


if __name__ == "__main__":
    unittest.main()
