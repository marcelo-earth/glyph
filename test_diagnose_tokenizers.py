import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from corpus_snapshot import prepare_snapshot
from diagnose_tokenizers import load_diagnostic_partitions, paired_difference
from experiment import aggregate_documents


class DiagnosticTests(unittest.TestCase):
    def scores(self, nlls):
        texts = ["café", "owl 🦉"]
        records = [{"index": i, "nll_nats": nll, "text_tokens": 2, "scored_tokens": 2,
                    "characters": len(text), "utf8_bytes": len(text.encode("utf-8"))}
                   for i, (text, nll) in enumerate(zip(texts, nlls))]
        sources = aggregate_documents(records, [0, 1], texts)
        return {"source_documents": sources,
                **{key: sum(row[key] for row in records)
                   for key in ("nll_nats", "text_tokens", "characters", "utf8_bytes")}}

    def test_paired_totals_and_identical_bootstrap(self):
        fit, general = self.scores([2., 4.]), self.scores([3., 6.])
        result = paired_difference(fit, general, bootstrap_samples=100)
        self.assertEqual(result["nll_nats"], -3)
        self.assertAlmostEqual(result["bits_per_utf8_byte"], -3 / (13 * math.log(2)))
        same = paired_difference(fit, fit, bootstrap_samples=100)
        self.assertEqual(same["paired_source_bootstrap"]["bits_per_utf8_byte_percentile_95"], [0, 0])

    def test_mismatched_pair_and_wrong_totals_fail(self):
        original = self.scores([2., 4.])
        bad = copy.deepcopy(original)
        bad["source_documents"].reverse()
        with self.assertRaisesRegex(ValueError, "Paired source mismatch"):
            paired_difference(original, bad)
        bad = copy.deepcopy(original)
        bad["nll_nats"] += 1
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            paired_difference(original, bad)

    def test_validation_loader_never_opens_test_partition(self):
        pools = {"target": [f"target source number {i}" for i in range(5)],
                 "general": [f"general source number {i}" for i in range(5)]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot"
            prepare_snapshot(root, corpus="target", general_corpus="general", max_docs=5,
                             val_docs=1, test_docs=1, tokenizer_chars=30,
                             loader=lambda name, **_: pools[name],
                             source_specs={"target": {}, "general": {}})
            original = Path.read_bytes

            def guarded(path):
                if path.name == "test.jsonl":
                    raise AssertionError("sealed test was opened")
                return original(path)

            with patch.object(Path, "read_bytes", guarded):
                loaded = load_diagnostic_partitions(root)
            self.assertEqual(set(loaded), {"manifest", "train", "validation"})
            (root / "validation.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_diagnostic_partitions(root)


if __name__ == "__main__":
    unittest.main()
