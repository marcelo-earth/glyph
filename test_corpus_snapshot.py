"""Offline tests for split integrity, budget accounting, and snapshot identity."""

import json
from pathlib import Path
import tempfile
import unittest

from corpus_snapshot import exact_character_prefix, load_snapshot, prepare_snapshot, split_documents


class CorpusSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = [f"target {i}: " + "héllo 🐍 " * 8 for i in range(12)]
        self.general = [f"general {i}: " + "world 🌍 " * 8 for i in range(12)]
        self.sources = {"target": {"path": "offline/target", "split": "train"},
                        "general": {"path": "offline/general", "split": "train"}}

    def prepare(self, name="snapshot", **kwargs):
        data = {"target": self.target, "general": self.general}
        def loader(corpus, **options):
            self.assertTrue(options["streaming"])
            return data[corpus][:options["max_samples"]]
        options = dict(corpus="target", general_corpus="general", max_docs=100,
                       val_docs=2, test_docs=2, seed=19, loader=loader, source_specs=self.sources)
        options.update(kwargs)
        path = Path(self.temp.name) / name
        return path, prepare_snapshot(path, **options)

    def test_deterministic_split_deduplicates_before_holdout(self):
        texts = self.target + self.target[:3]
        first = split_documents(texts, 2, 3, seed=9)
        self.assertEqual(first, split_documents(texts, 2, 3, seed=9))
        self.assertNotEqual(first, split_documents(texts, 2, 3, seed=10))
        flattened = sum(first.values(), [])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(self.target))

    def test_exact_unicode_prefix(self):
        self.assertEqual(exact_character_prefix(["a🐍é", "世界abc"], 5), ["a🐍é", "世界"])
        self.assertEqual(exact_character_prefix(["a🐍é", "世界abc"], 3), ["a🐍é"])
        with self.assertRaises(ValueError):
            exact_character_prefix(["abc"], 4)
        with self.assertRaises(ValueError):
            exact_character_prefix(["abc"], 0)

    def test_snapshot_excludes_all_target_docs_from_general(self):
        self.target += self.target[:2]
        self.general += self.target + self.general[:3]
        path, result = self.prepare(tokenizer_chars=175)
        self.assertEqual(result, load_snapshot(path))
        all_target = set(result["train"] + result["validation"] + result["test"])
        self.assertTrue(all_target.isdisjoint(result["general"]))
        self.assertEqual(sum(map(len, result["fit_tokenizer"])), 175)
        self.assertEqual(sum(map(len, result["general_tokenizer"])), 175)
        self.assertEqual(result["fit_tokenizer"], exact_character_prefix(result["train"], 175))
        self.assertEqual(result["manifest"]["deduplication"]["target_removed"], 2)
        self.assertEqual(result["manifest"]["deduplication"]["general_removed"], 17)

    def test_identical_inputs_have_identical_snapshot_identity(self):
        _, one = self.prepare("one")
        _, two = self.prepare("two")
        self.assertEqual(one, two)
        _, different = self.prepare("different", seed=20)
        self.assertNotEqual(one["manifest"]["snapshot_sha256"], different["manifest"]["snapshot_sha256"])

    def test_tampered_partition_is_rejected(self):
        path, _ = self.prepare()
        with (path / "train.jsonl").open("a") as stream:
            stream.write(json.dumps({"text": "injected", "sha256": "bad"}) + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_snapshot(path)

    def test_tampered_manifest_is_rejected(self):
        path, _ = self.prepare()
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["split"]["seed"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_snapshot(path)

    def test_impossible_budgets_and_empty_partitions_fail_before_writing(self):
        with self.assertRaises(ValueError):
            self.prepare(tokenizer_chars=1_000_000)
        self.assertFalse((Path(self.temp.name) / "snapshot").exists())
        with self.assertRaises(ValueError):
            split_documents(["a", "b"], 1, 1)
        with self.assertRaises(ValueError):
            split_documents(self.target, 0, 1)
        self.general = self.target[:]
        with self.assertRaisesRegex(ValueError, "empty"):
            self.prepare()

    def test_cannot_overwrite_snapshot(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
