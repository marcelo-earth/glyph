"""Training-only tokenizer baselines and paired held-out coding diagnostics.

The test partition is never opened unless --split test is explicitly supplied.
Whole-document and shared-raw-block encodings are separate experiments because
cutting text before BPE changes canonical segmentation. Neither coding metric
is the marginal probability of the raw string.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from evaluation import evaluate_token_baselines, prepare_evaluation
from experiment import aggregate_documents, atomic_json, digest, raw_blocks


def _canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_diagnostic_partitions(snapshot_dir, split="validation"):
    """Verify the manifest and selected partitions without opening sealed test.

    Full cross-partition/provenance validation belongs to snapshot creation. Here
    we authenticate the selected files against that immutable snapshot manifest.
    """
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or explicitly requested test")
    root = Path(snapshot_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    identity = {k: v for k, v in manifest.items() if k != "snapshot_sha256"}
    if manifest["snapshot_sha256"] != _canonical_hash(identity):
        raise ValueError("Snapshot manifest checksum mismatch")
    result = {"manifest": manifest}
    for name in ("train", split):
        spec = manifest["partitions"][name]
        if spec["file"] != f"{name}.jsonl":
            raise ValueError("Unexpected partition filename")
        raw = (root / spec["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["file_sha256"]:
            raise ValueError(f"Partition file checksum mismatch: {name}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        texts = [row["text"] for row in rows]
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        if hashes != [row["sha256"] for row in rows]:
            raise ValueError(f"Document checksum mismatch: {name}")
        actual = (len(texts), sum(map(len, texts)),
                  sum(len(t.encode("utf-8")) for t in texts), _canonical_hash(hashes))
        expected = (spec["documents"], spec["characters"], spec["utf8_bytes"],
                    spec["ordered_text_sha256"])
        if actual != expected:
            raise ValueError(f"Partition statistics mismatch: {name}")
        result[name] = texts
    return result


def vocabulary_usage(tokenizer, training, evaluation):
    train_counts = Counter(t for d in training.documents for t in d.token_ids)
    eval_counts = Counter(t for d in evaluation.documents for t in d.token_ids)
    controls = {tokenizer.token_to_id(t) for t in ("[BOS]", "[EOS]", "[PAD]", "[UNK]")}
    controls.discard(None)
    if controls & (train_counts.keys() | eval_counts.keys()):
        raise ValueError("Control token occurred in ordinary source content")
    vocab_size = tokenizer.get_vocab_size()

    def describe(counts, prepared):
        total = sum(counts.values())
        characters = sum(d.characters for d in prepared.documents)
        nbytes = sum(d.utf8_bytes for d in prepared.documents)
        entropy = -math.fsum((c / total) * math.log(c / total) for c in counts.values())
        return {
            "documents_or_blocks": len(prepared.documents), "tokens": total,
            "characters": characters, "utf8_bytes": nbytes,
            "characters_per_token": characters / total, "utf8_bytes_per_token": nbytes / total,
            "tokens_per_utf8_byte": total / nbytes,
            "unique_content_ids": len(counts),
            "fraction_total_vocabulary_used": len(counts) / vocab_size,
            "fraction_noncontrol_vocabulary_used": len(counts) / (vocab_size - len(controls)),
            "empirical_token_entropy_bits": entropy / math.log(2),
            "effective_vocabulary_exp_entropy": math.exp(entropy),
            "top_tokens": [{"id": token, "symbol": tokenizer.id_to_token(token), "count": count}
                           for token, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:20]],
        }

    unseen = eval_counts.keys() - train_counts.keys()
    return {
        "train": describe(train_counts, training), "evaluation": describe(eval_counts, evaluation),
        "evaluation_ids_unseen_in_lm_train": len(unseen),
        "evaluation_tokens_unseen_in_lm_train": sum(eval_counts[t] for t in unseen),
        "evaluation_fraction_tokens_unseen_in_lm_train":
            sum(eval_counts[t] for t in unseen) / sum(eval_counts.values()),
    }


def paired_difference(fit, general, *, bootstrap_samples=2000, seed=0):
    """Fit minus general; resample independent source documents, not raw blocks."""
    a, b = fit["source_documents"], general["source_documents"]
    if not a or len(a) != len(b):
        raise ValueError("Paired source counts differ or are empty")
    for left, right in zip(a, b):
        for key in ("index", "sha256", "characters", "utf8_bytes"):
            if left[key] != right[key]:
                raise ValueError(f"Paired source mismatch: {key}")
    for metrics in (fit, general):
        for key in ("nll_nats", "text_tokens", "characters", "utf8_bytes"):
            if not math.isclose(math.fsum(d[key] for d in metrics["source_documents"]),
                                metrics[key], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError(f"Aggregate source accounting mismatch: {key}")
    deltas = np.array([x["nll_nats"] - y["nll_nats"] for x, y in zip(a, b)])
    nbytes = np.array([x["utf8_bytes"] for x in a])
    nchars = np.array([x["characters"] for x in a])
    result = {
        "direction": "fit minus general; negative NLL difference favors fit",
        "nll_nats": float(deltas.sum()),
        "bits_per_utf8_byte": float(deltas.sum() / (nbytes.sum() * math.log(2))),
        "bits_per_character": float(deltas.sum() / (nchars.sum() * math.log(2))),
        "fit_over_general_token_count": fit["text_tokens"] / general["text_tokens"],
        "fit_token_count_reduction_fraction": 1 - fit["text_tokens"] / general["text_tokens"],
        "source_documents": [
            {"index": x["index"], "sha256": x["sha256"], "nll_nats_difference": float(delta),
             "characters": x["characters"], "utf8_bytes": x["utf8_bytes"]}
            for x, delta in zip(a, deltas)
        ],
    }
    if bootstrap_samples:
        indices = np.random.default_rng(seed).integers(0, len(a), (bootstrap_samples, len(a)))
        boot = deltas[indices].sum(axis=1) / (nbytes[indices].sum(axis=1) * math.log(2))
        result["paired_source_bootstrap"] = {
            "replicates": bootstrap_samples, "seed": seed,
            "bits_per_utf8_byte_percentile_95": np.quantile(boot, [.025, .975]).tolist(),
            "scope": "source-document sampling only; does not represent LM training-seed uncertainty",
        }
    return result


def diagnose_tokenizers(snapshot_dir, tokenizers_dir, *, split="validation", max_docs=100,
                        block_bytes=128):
    if max_docs < 1:
        raise ValueError("max_docs must be positive")
    snapshot = load_diagnostic_partitions(snapshot_dir, split)
    root = Path(tokenizers_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["snapshot_sha256"] != digest(Path(snapshot_dir) / "manifest.json"):
        raise ValueError("Tokenizer manifest belongs to another snapshot")
    train_texts, eval_texts = snapshot["train"], snapshot[split][:max_docs]
    train_blocks, _ = raw_blocks(train_texts, block_bytes)
    eval_blocks, parents = raw_blocks(eval_texts, block_bytes)
    modes = {
        "whole_documents": (train_texts, eval_texts, list(range(len(eval_texts)))),
        "shared_raw_blocks": (train_blocks, eval_blocks, parents),
    }
    result = {
        "schema_version": 1, "snapshot_manifest_sha256": digest(Path(snapshot_dir) / "manifest.json"),
        "tokenizers_manifest_sha256": digest(root / "manifest.json"),
        "diagnostics_implementation_sha256": digest(__file__),
        "split": split, "evaluation_documents": len(eval_texts), "max_docs": max_docs,
        "block_bytes": block_bytes, "include_eos": False,
        "metric": "canonical_token_sequence_nll",
        "notes": [
            "Whole documents and shared raw blocks are tokenized independently; block boundaries change BPE merges.",
            "Unigram counts use every LM training document, with matching whole-document or raw-block segmentation.",
            "These baselines include no inserted BOS/EOS counts and use add-one smoothing over the full model vocabulary.",
            "Unigram training counts need not match exposure in a token-budget-limited LM run.",
            "Vocabulary utilization is a diagnostic; uniform code-length reductions alone do not demonstrate learned modeling gains.",
        ],
        "tokenizers": {}, "paired_differences": {},
    }
    for label in ("fit", "general"):
        path = root / f"{label}.json"
        if digest(path) != manifest[label]["sha256"]:
            raise ValueError(f"Tokenizer checksum mismatch: {label}")
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.encode_special_tokens = True
        entry = {"sha256": digest(path), "vocab_size": tokenizer.get_vocab_size(), "segmentations": {}}
        for mode, (training_texts, scoring_texts, source_ids) in modes.items():
            print(f"Diagnosing {label}: {mode}", flush=True)
            training = prepare_evaluation(tokenizer, training_texts)
            evaluation = prepare_evaluation(tokenizer, scoring_texts)
            baselines = evaluate_token_baselines(tokenizer, training_texts, prepared=evaluation)
            for scores in baselines.values():
                scores["source_documents"] = aggregate_documents(scores.pop("documents"), source_ids, eval_texts)
            entry["segmentations"][mode] = {
                "vocabulary_and_compression": vocabulary_usage(tokenizer, training, evaluation),
                "baselines": baselines,
            }
        examples = []
        for index, text in enumerate(eval_texts[:3]):
            short = text[:120]
            encoded = tokenizer.encode(short, add_special_tokens=False)
            examples.append({"source_index": index, "text": short, "truncated": len(text) > 120,
                             "token_ids": encoded.ids, "token_symbols": encoded.tokens,
                             "offsets": encoded.offsets,
                             "roundtrip_exact": tokenizer.decode(encoded.ids, skip_special_tokens=False) == short})
        entry["segmentation_examples"] = examples
        result["tokenizers"][label] = entry
    fit, general = result["tokenizers"]["fit"], result["tokenizers"]["general"]
    if fit["vocab_size"] != general["vocab_size"]:
        raise ValueError("Paired tokenizer vocabularies differ")
    for mode in modes:
        result["paired_differences"][mode] = {
            name: paired_difference(fit["segmentations"][mode]["baselines"][name],
                                    general["segmentations"][mode]["baselines"][name])
            for name in ("uniform", "unigram")
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tokenizers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--block-bytes", type=int, default=128)
    args = parser.parse_args()
    result = diagnose_tokenizers(args.snapshot, args.tokenizers, split=args.split,
                                 max_docs=args.max_docs, block_bytes=args.block_bytes)
    atomic_json(args.out, result)
    print(json.dumps({mode: {name: {k: v for k, v in values.items()
                                   if k != "source_documents"}
                            for name, values in pairs.items()}
                      for mode, pairs in result["paired_differences"].items()}, indent=2))


if __name__ == "__main__":
    main()
