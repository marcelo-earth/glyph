import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analyze_checkpoint_pairs import analyze
from experiment import atomic_json, digest


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CheckpointPairAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        sources = ["first", "second"]
        partition = "\n".join(json.dumps({"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
                              for text in sources).encode()
        (self.snapshot / "validation.jsonl").write_bytes(partition)
        metadata = "\n".join(json.dumps({"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                           "repository_family": f"repo-{index}"})
                             for index, text in enumerate(sources)).encode()
        (self.snapshot / "validation_repositories.jsonl").write_bytes(metadata)
        atomic_json(self.snapshot / "manifest.json", {
            "repository_metadata": {"validation": {"file": "validation_repositories.jsonl",
                                                       "file_sha256": hashlib.sha256(metadata).hexdigest()}},
        })
        self.sources = [{"sha256": hashlib.sha256(text.encode()).hexdigest(), "utf8_bytes": len(text),
                         "nll_nats": 2.0 + index} for index, text in enumerate(sources)]

    def evaluation(self, path, nlls, protocol=None):
        protocol = protocol or {"split": "validation", "mode": "raw", "raw_block_bytes": 128}
        sources = [dict(source, nll_nats=nll) for source, nll in zip(self.sources, nlls)]
        result = {"status": "complete", "identity": {
            "snapshot_manifest_sha256": digest(self.snapshot / "manifest.json"), "protocol": protocol},
            "scores": {"source_documents": sources,
                       "bits_per_utf8_byte": sum(nlls) / sum(s["utf8_bytes"] for s in sources),
                       "characters": sum(s["utf8_bytes"] for s in sources),
                       "utf8_bytes": sum(s["utf8_bytes"] for s in sources)}}
        result["payload_sha256"] = payload_hash(result)
        atomic_json(path, result)

    def test_analyzes_compatible_authenticated_pair(self):
        fit, general = self.root / "fit.json", self.root / "general.json"
        self.evaluation(fit, [2.0, 3.0])
        self.evaluation(general, [3.0, 5.0])
        result = analyze(fit, general, self.snapshot)
        self.assertLess(result["fit_minus_general_bpb"], 0)
        self.assertEqual(result["bootstrap"]["group_count"], 2)

    def test_rejects_mismatched_protocol(self):
        fit, general = self.root / "fit.json", self.root / "general.json"
        self.evaluation(fit, [2.0, 3.0])
        self.evaluation(general, [3.0, 5.0], {"split": "validation", "mode": "token"})
        with self.assertRaisesRegex(ValueError, "different protocols"):
            analyze(fit, general, self.snapshot)


if __name__ == "__main__":
    unittest.main()
