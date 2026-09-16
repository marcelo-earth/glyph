"""Offline controls for external tokenizer provenance and anchored deduplication."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from corpus_snapshot import _canonical_hash, text_sha256
from prepare_disjoint_tokenizer import (acquire_candidates, build_disjoint_pair, file_sha256,
                                        filter_against_references, load_reference,
                                        prepare_disjoint_corpus, verify_disjoint_corpus)
from prepare_python import prepare_python_snapshot, token_shingles


def record(i, text):
    return {"text": text, "sha256": text_sha256(text), "repo_name": f"owner/project{i}",
            "repository_family": f"project{i}", "path": "file.py"}


class DisjointTokenizerTests(unittest.TestCase):
    def test_excludes_families_exact_and_normalized_text(self):
        reference = record(0, "alpha beta gamma")
        rows = [
            {"repo_name": "fork/PROJECT0", "path": "a.py", "content": "completely new content"},
            {"repo_name": "owner/other", "path": "b.py", "content": "alpha beta gamma"},
            {"repo_name": "owner/other", "path": "c.py", "content": "alpha  beta\ngamma"},
            {"repo_name": "owner/allowed", "path": "d.py", "content": "unique unrelated document"},
        ]
        selected, stats = acquire_candidates(iter(rows), [reference], candidate_chars=20, min_chars=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["repository_family"], "allowed")
        self.assertEqual(stats["excluded_reference_family"], 1)
        self.assertEqual(stats["exact_duplicate"], 1)
        self.assertEqual(stats["whitespace_duplicate"], 1)

    def test_incomplete_acquisition_refused(self):
        with self.assertRaisesRegex(ValueError, "Incomplete acquisition"):
            acquire_candidates(iter([]), [], candidate_chars=100)

    def test_anchored_filter_matches_exhaustive_including_shorter_candidate(self):
        rng = random.Random(34)
        references = [record(i, " ".join(str(rng.randrange(100)) for _ in range(70))) for i in range(8)]
        candidates = []
        for i, reference in enumerate(references):
            candidates.extend([record(20 + 2 * i, reference["text"][:-4]),
                               record(21 + 2 * i, reference["text"] + " 9999")])
        candidates.append(record(200, "utterly different lexical content with many new identifiers"))
        candidates.append(record(201, candidates[-1]["text"] + " changed"))
        for threshold in (0.5, 0.85, 1.0):
            expected, anchors = [], [token_shingles(r["text"]) for r in references]
            for candidate in candidates:
                items = token_shingles(candidate["text"])
                if not any(len(items & anchor) / len(items | anchor) >= threshold for anchor in anchors):
                    expected.append(candidate)
                    anchors.append(items)
            actual, _ = filter_against_references(references, candidates, threshold)
            self.assertEqual(actual, expected)

    def make_reference(self, path):
        rows = [{"repo_name": f"owner/ref{i}", "path": "module.py",
                 "content": f"def function_{i}():\n    return [{i}, {i+5}, {i+9}]\n", "license": "mit"}
                for i in range(15)]
        prepare_python_snapshot(path, max_docs=15, max_scanned=20, min_chars=10,
                                max_chars=100, val_docs=2, test_docs=2, tokenizer_chars=100,
                                rows=iter(rows), general_texts=["general prose with many distinctive words " * 10])

    def make_corpus(self, root):
        snapshot, corpus = root / "lm", root / "external"
        self.make_reference(snapshot)
        rows = [{"repo_name": f"extra/newproject{i}", "path": "new.py", "license": "mit",
                 "content": f"class External{i}:\n    attribute_{i} = 'different_{i}'\n    constant_{i} = True\n"}
                for i in range(10)]
        prepare_disjoint_corpus(snapshot, corpus, tokenizer_chars=100, candidate_chars=300,
                                min_chars=10, rows=iter(rows))
        return snapshot, corpus

    def test_corpus_metadata_disjointness_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, corpus = self.make_corpus(Path(tmp))
            texts, manifest = verify_disjoint_corpus(corpus, snapshot)
            self.assertEqual(sum(map(len, texts)), 100)
            references, _ = load_reference(snapshot)
            metadata = [json.loads(line) for line in (corpus / "metadata.jsonl").read_text().splitlines()]
            self.assertFalse({r["repository_family"] for r in references} & {r["repository_family"] for r in metadata})
            self.assertEqual(manifest["lm_snapshot_manifest_file_sha256"], file_sha256(snapshot / "manifest.json"))
            with self.assertRaises(FileExistsError):
                prepare_disjoint_corpus(snapshot, corpus)

    def test_semantic_metadata_overlap_rejected_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, corpus = self.make_corpus(Path(tmp))
            references, _ = load_reference(snapshot)
            metadata_path = corpus / "metadata.jsonl"
            metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
            metadata[0]["repo_name"] = references[0]["repo_name"]
            metadata[0]["repository_family"] = references[0]["repository_family"]
            metadata_path.write_text("".join(json.dumps(row) + "\n" for row in metadata))
            manifest_path = corpus / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["metadata"]["file_sha256"] = file_sha256(metadata_path)
            manifest.pop("corpus_sha256")
            manifest["corpus_sha256"] = _canonical_hash(manifest)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "family overlap"):
                verify_disjoint_corpus(corpus, snapshot)

    def test_pair_copies_general_bytes_and_binds_external_provenance(self):
        from build_tokenizer import build_bpe
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, corpus = self.make_corpus(root)
            original, out = root / "original", root / "newpair"
            original.mkdir()
            build_bpe(["general prose " * 10], 260, str(original / "general.json"))
            original_manifest = {"snapshot_sha256": file_sha256(snapshot / "manifest.json"),
                                 "vocab_size": 260, "settings": {"min_frequency": 2, "byte_level": True,
                                                                 "add_prefix_space": False},
                                 "general": {"sha256": file_sha256(original / "general.json"),
                                             "training_chars": 100, "training_bytes": 100}}
            (original / "manifest.json").write_text(json.dumps(original_manifest))
            manifest = build_disjoint_pair(snapshot, corpus, original, out)
            self.assertEqual((out / "general.json").read_bytes(), (original / "general.json").read_bytes())
            self.assertEqual(manifest["fit"]["condition"], "disjoint_in_domain")
            self.assertEqual(manifest["fit"]["training_chars"], manifest["general"]["training_chars"])
            self.assertEqual(manifest["snapshot_sha256"], original_manifest["snapshot_sha256"])
            self.assertEqual(manifest["fit"]["corpus_manifest_file_sha256"], file_sha256(corpus / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
