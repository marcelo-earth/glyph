"""Posthoc scoring of trusted local Glyph checkpoints with an explicit protocol.

Validation is the default; the test split is opened only with --split test.
Use --mode raw to retokenize shared UTF-8-bounded blocks, or --mode token for
canonical whole-document encodings with token-count-limited model context.
Checkpoints contain pickle data and must come from this trusted experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from tokenizers import Tokenizer

from diagnose_tokenizers import load_diagnostic_partitions
from evaluation import evaluate_documents, prepare_evaluation
from experiment import aggregate_documents, atomic_json, digest, raw_blocks, verify_tokenizer_manifest
from model import GlyphGPT


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _repository_metadata(root):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        return {"git_commit": revision, "git_dirty": bool(status),
                "git_status_sha256": hashlib.sha256(status.encode()).hexdigest()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None, "git_status_sha256": None}


def evaluate_checkpoint(snapshot_dir, tokenizer_path, checkpoint_path, out_path, *,
                        split="validation", max_docs=None, mode="raw", block_bytes=None,
                        context_policy="chunks", batch_size=None, device="cpu"):
    """Score all selected sources; cache only an identical, authenticated protocol.

    Model architecture and vocabulary come from the checkpoint. Its snapshot,
    tokenizer, training config hash, and all four recorded implementation files
    must match current inputs/source. Cache identity includes checkpoint bytes,
    protocol, runtime, and scoring implementation hashes. Existing unrelated or
    altered outputs are rejected. Exact reruns return their existing result.
    """
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or explicitly requested test")
    if mode not in ("raw", "token"):
        raise ValueError("mode must be raw or token")
    if context_policy not in ("chunks", "rolling"):
        raise ValueError("context_policy must be chunks or rolling")
    if max_docs is not None and max_docs < 1:
        raise ValueError("max_docs must be positive or omitted for the full split")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if mode == "token" and block_bytes is not None:
        raise ValueError("block_bytes applies only to raw mode")
    source_root = Path(__file__).resolve().parent
    checkpoint_path, out_path = Path(checkpoint_path), Path(out_path)
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if digest(checkpoint_path) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    config = checkpoint["config"]
    config_hash = _identity(config)
    if config_hash != checkpoint["config_hash"]:
        raise ValueError("Checkpoint configuration checksum mismatch")
    if config["snapshot_sha256"] != digest(Path(snapshot_dir) / "manifest.json"):
        raise ValueError("Checkpoint snapshot checksum mismatch")
    if config["tokenizer_sha256"] != digest(tokenizer_path):
        raise ValueError("Checkpoint tokenizer checksum mismatch")
    expected_source = config["implementation_sha256"]
    for name in ("experiment.py", "model.py", "evaluation.py", "corpus_snapshot.py"):
        if expected_source.get(name) != digest(source_root / name):
            raise ValueError(f"Checkpoint source checksum mismatch: {name}")
    verify_tokenizer_manifest(snapshot_dir, tokenizer_path, config["label"])
    if not isinstance(checkpoint["step"], int) or checkpoint["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    seq_len = config["seq_len"]
    if mode == "raw":
        block_bytes = seq_len if block_bytes is None else block_bytes
        if not 4 <= block_bytes <= seq_len:
            raise ValueError("block_bytes must be at least 4 and at most model seq_len")
    batch_size = config["batch_size"] if batch_size is None else batch_size
    snapshot = load_diagnostic_partitions(snapshot_dir, split)
    texts = snapshot[split] if max_docs is None else snapshot[split][:max_docs]
    if not texts:
        raise ValueError("Selected evaluation split is empty")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.encode_special_tokens = True
    if tokenizer.get_vocab_size() != config["vocab_size"]:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    evaluation_texts, parents = (raw_blocks(texts, block_bytes) if mode == "raw"
                                 else (texts, list(range(len(texts)))))
    prepared = prepare_evaluation(tokenizer, evaluation_texts)
    if mode == "raw" and any(len(d.token_ids) > seq_len for d in prepared.documents):
        raise ValueError("Raw block token count exceeds model context capacity")
    protocol = {
        "split": split, "source_selection": "ordered prefix" if max_docs is not None else "entire split",
        "max_docs": max_docs, "source_documents": len(texts),
        "split_file_sha256": snapshot["manifest"]["partitions"][split]["file_sha256"],
        "ordered_source_sha256": _identity([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]),
        "mode": mode, "raw_block_bytes": block_bytes, "context_policy": context_policy,
        "context_length": seq_len, "include_eos": False, "first_token_context": "BOS",
        "special_token_literals": "ordinary source text",
        "denominators": "all selected source Unicode code points and UTF-8 bytes",
        "metric": "canonical_token_sequence_nll", "batch_size": batch_size,
        "device": str(device), "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "torch_threads": torch.get_num_threads(), "machine": platform.machine(),
        "scoring_implementation_sha256": {
            name: digest(source_root / name) for name in
            ("evaluate_checkpoint.py", "diagnose_tokenizers.py", "experiment.py", "evaluation.py", "model.py")
        },
    }
    identity = {"schema_version": 1, "checkpoint_sha256": checkpoint_hash,
                "checkpoint_step": checkpoint["step"], "training_config_sha256": config_hash,
                "snapshot_manifest_sha256": config["snapshot_sha256"],
                "tokenizer_sha256": config["tokenizer_sha256"], "protocol": protocol}
    identity_hash = _identity(identity)
    if out_path.exists():
        cached = json.loads(out_path.read_text())
        if cached.get("identity_sha256") != identity_hash or cached.get("identity") != identity:
            raise ValueError("Output belongs to another checkpoint or evaluation protocol")
        payload = {k: v for k, v in cached.items() if k != "payload_sha256"}
        if cached.get("payload_sha256") != _identity(payload) or cached.get("status") != "complete":
            raise ValueError("Existing output is incomplete or has an invalid payload checksum")
        return cached

    model = GlyphGPT(config["vocab_size"], config["dim"], config["heads"],
                     config["layers"], seq_len, dropout=config["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    start = time.perf_counter()
    scores = evaluate_documents(model, prepared=prepared, batch_size=batch_size,
                                device=device, context_length=seq_len,
                                context_policy=context_policy)
    scores["source_documents"] = aggregate_documents(scores["documents"], parents, texts)
    if mode == "raw":
        scores["raw_block_bytes"] = block_bytes
    result = {
        "status": "complete", "identity": identity, "identity_sha256": identity_hash,
        "training_config": config, "checkpoint_step": checkpoint["step"],
        "checkpoint_result_status": checkpoint.get("result", {}).get("status"),
        "checkpoint_exposure": checkpoint.get("result", {}).get("exposure"),
        "evaluation_seconds": time.perf_counter() - start,
        "repository": _repository_metadata(source_root), "scores": scores,
        "interpretation": [
            "Canonical-token NLL is not the probability summed over every tokenization of the string.",
            "Raw mode uses shared byte-bounded segments; token mode uses tokenizer-dependent context spans.",
            "Source-document totals support paired resampling; blocks from one source are not independent observations.",
        ],
    }
    result["payload_sha256"] = _identity(result)
    atomic_json(out_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--mode", choices=("raw", "token"), default="raw")
    parser.add_argument("--block-bytes", type=int)
    parser.add_argument("--context-policy", choices=("chunks", "rolling"), default="chunks")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.snapshot, args.tokenizer, args.checkpoint, args.out,
                                 split=args.split, max_docs=args.max_docs, mode=args.mode,
                                 block_bytes=args.block_bytes, context_policy=args.context_policy,
                                 batch_size=args.batch_size, device=args.device)
    print(json.dumps({"checkpoint_step": result["checkpoint_step"],
                      "protocol": result["identity"]["protocol"],
                      "scores": {k: v for k, v in result["scores"].items()
                                 if k not in ("documents", "source_documents")}}, indent=2))


if __name__ == "__main__":
    main()
