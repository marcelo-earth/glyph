"""Summarize a compatible pair of full-split checkpoint evaluations.

The training-matrix analysis summarizes the fixed development monitor that was
scored during training.  This tool separately analyzes authenticated posthoc
checkpoint evaluations, preserving their declared context policy and grouping
uncertainty at the original Python repository-family unit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_matrix import group_bootstrap
from experiment import atomic_json, digest


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _load(path):
    path = Path(path)
    result = json.loads(path.read_text())
    payload = {key: value for key, value in result.items() if key != "payload_sha256"}
    if result.get("status") != "complete" or result.get("payload_sha256") != _canonical_hash(payload):
        raise ValueError(f"Invalid or incomplete checkpoint evaluation: {path}")
    return result


def _repository_groups(snapshot_dir, sources):
    root = Path(snapshot_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = manifest.get("repository_metadata", {}).get("validation")
    if metadata is None:
        raise ValueError("Snapshot has no validation repository metadata")
    raw = (root / metadata["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["file_sha256"]:
        raise ValueError("Repository metadata checksum differs")
    records = [json.loads(line) for line in raw.decode().splitlines()]
    if len(records) != len(sources):
        raise ValueError("Repository metadata source count differs")
    if [record["sha256"] for record in records] != [source["sha256"] for source in sources]:
        raise ValueError("Repository metadata source order differs")
    return [record["repository_family"] for record in records]


def analyze(fit_path, general_path, snapshot_dir):
    fit, general = _load(fit_path), _load(general_path)
    fit_identity, general_identity = fit["identity"], general["identity"]
    if fit_identity["snapshot_manifest_sha256"] != digest(Path(snapshot_dir) / "manifest.json"):
        raise ValueError("Fit evaluation belongs to another snapshot")
    if general_identity["snapshot_manifest_sha256"] != fit_identity["snapshot_manifest_sha256"]:
        raise ValueError("Evaluations use different snapshots")
    if fit_identity["protocol"] != general_identity["protocol"]:
        raise ValueError("Evaluations use different protocols")
    fit_scores, general_scores = fit["scores"], general["scores"]
    groups = _repository_groups(snapshot_dir, fit_scores["source_documents"])
    bootstrap = group_bootstrap(fit_scores["source_documents"], general_scores["source_documents"], groups)
    for key in ("characters", "utf8_bytes"):
        if fit_scores[key] != general_scores[key]:
            raise ValueError(f"Paired evaluation denominator differs: {key}")
    difference = fit_scores["bits_per_utf8_byte"] - general_scores["bits_per_utf8_byte"]
    return {
        "scope": "development only; posthoc checkpoint evaluation, no final test inference",
        "bootstrap_unit": "repository_family",
        "fit_evaluation_sha256": digest(fit_path),
        "general_evaluation_sha256": digest(general_path),
        "protocol": fit_identity["protocol"],
        "fit_bpb": fit_scores["bits_per_utf8_byte"],
        "general_bpb": general_scores["bits_per_utf8_byte"],
        "fit_minus_general_bpb": difference,
        "relative_fit_improvement": -difference / general_scores["bits_per_utf8_byte"],
        "bootstrap": bootstrap,
        "interpretation": [
            "Negative fit-minus-general BPB favors the corpus-fit tokenizer.",
            "The bootstrap covers validation repository-family sampling only, not training-seed or tokenizer-sample variation.",
            "Canonical-token NLL is not exact decoded-string probability marginalized over tokenizations.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--general", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(args.fit, args.general, args.snapshot)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
