"""An in-domain tokenizer trained on Python families absent from the LM corpus.

The language-model snapshot and the original general tokenizer stay fixed.
Only the fit-labeled tokenizer changes; its data come from additional pinned
CodeParrot files, with family, exact-text, and lexical-near-duplicate exclusions.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

from corpus_snapshot import _canonical_hash, _write_partition, exact_character_prefix, load_snapshot, text_sha256
from prepare_python import CODEPARROT_REVISION, iter_python_rows, repository_family, token_shingles


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    snapshot = load_snapshot(snapshot_dir)
    records = []
    for partition in ("train", "validation", "test"):
        metadata = [json.loads(line) for line in (snapshot_dir / f"{partition}.metadata.jsonl").read_text().splitlines()]
        for text, row in zip(snapshot[partition], metadata):
            if row["repository_family"] != repository_family(row["repo_name"]):
                raise ValueError("Reference family identity disagrees with repository")
            records.append(dict(row, text=text, partition=partition))
    return records, snapshot["manifest"]


def acquire_candidates(rows, references, *, candidate_chars, max_scanned=100_000,
                       min_chars=200, max_chars=20_000):
    if min(candidate_chars, max_scanned, min_chars) < 1 or max_chars < min_chars:
        raise ValueError("Invalid acquisition limits")
    families = {record["repository_family"] for record in references}
    exact = {text_sha256(record["text"]) for record in references}
    normalized = {text_sha256(" ".join(record["text"].split())) for record in references}
    selected, characters, stats = [], 0, Counter()
    try:
        for row in rows:
            stats["scanned"] += 1
            text = row.get("content")
            try:
                family = repository_family(row.get("repo_name"))
            except ValueError:
                family = None
            if family is None:
                stats["missing_repository"] += 1
            elif family in families:
                stats["excluded_reference_family"] += 1
            elif not isinstance(text, str) or not min_chars <= len(text) <= max_chars:
                stats["outside_size_bounds"] += 1
            elif not str(row.get("path", "")).lower().endswith(".py"):
                stats["non_python_path"] += 1
            else:
                digest = text_sha256(text)
                normal_digest = text_sha256(" ".join(text.split()))
                if digest in exact:
                    stats["exact_duplicate"] += 1
                elif normal_digest in normalized:
                    stats["whitespace_duplicate"] += 1
                else:
                    selected.append({"text": text, "sha256": digest, "repo_name": row["repo_name"],
                                     "repository_family": family, "path": row["path"],
                                     "license": row.get("license"), "source_shard": row.get("source_shard"),
                                     "source_line": row.get("source_line")})
                    exact.add(digest)
                    normalized.add(normal_digest)
                    characters += len(text)
            if characters >= candidate_chars or stats["scanned"] >= max_scanned:
                break
    finally:
        close = getattr(rows, "close", None)
        if close:
            close()
    if characters < candidate_chars:
        raise ValueError(f"Incomplete acquisition: {characters} candidate characters, need {candidate_chars}")
    stats.update(selected_documents=len(selected), selected_characters=characters)
    return selected, dict(stats)


def filter_against_references(references, candidates, threshold=0.85):
    """Exact lexical Jaccard search; all reference files are immutable anchors.

    One global rare-first shingle order gives complete prefix candidate search.
    Unlike ordinary greedy deduplication, a shorter candidate cannot displace a
    reference file. Accepted candidates also act as anchors for later candidates.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Jaccard threshold must be in (0, 1]")
    records = references + candidates
    vocabulary, sets = {}, []
    for record in records:
        items = set()
        for shingle in sorted(token_shingles(record["text"])):
            if shingle not in vocabulary:
                vocabulary[shingle] = len(vocabulary)
            items.add(vocabulary[shingle])
        sets.append(items)
    del vocabulary
    frequencies = Counter(item for items in sets for item in items)
    index = defaultdict(list)
    kept, rejections, comparisons = [], [], 0
    for i, items in enumerate(sets):
        prefix_size = len(items) - math.ceil(threshold * len(items)) + 1
        prefix = sorted(items, key=lambda item: (frequencies[item], item))[:prefix_size]
        rejected = None
        if i >= len(references):
            possible = {j for item in prefix for j in index[item]
                        if min(len(sets[j]), len(items)) >= threshold * max(len(sets[j]), len(items))}
            for j in sorted(possible):
                comparisons += 1
                overlap = len(items.intersection(sets[j]))
                similarity = overlap / (len(items) + len(sets[j]) - overlap)
                if similarity >= threshold:
                    rejected = {"candidate_sha256": records[i]["sha256"],
                                "anchor_sha256": records[j]["sha256"], "jaccard": similarity,
                                "anchor_is_lm_reference": j < len(references)}
                    break
        if rejected:
            rejections.append(rejected)
        else:
            if i >= len(references):
                kept.append(records[i])
            for item in prefix:
                index[item].append(i)
    return kept, {"method": "exact lexical 5-token-shingle Jaccard, complete rare-first prefix index",
                  "threshold": threshold, "reference_documents": len(references),
                  "candidate_documents": len(candidates), "accepted_documents": len(kept),
                  "verified_candidate_pairs": comparisons, "rejected_documents": len(rejections),
                  "reference_near_duplicates": sum(row["anchor_is_lm_reference"] for row in rejections),
                  "rejections": rejections,
                  "limitation": "Surface lexical similarity only; renamed forks and semantic clones may remain"}


def prepare_disjoint_corpus(snapshot_dir, out, *, tokenizer_chars=4_000_000,
                            candidate_chars=8_000_000, max_scanned=100_000,
                            min_chars=200, max_chars=20_000, seed=2026,
                            near_duplicate_threshold=0.85, rows=None):
    out, snapshot_dir = Path(out), Path(snapshot_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite disjoint corpus: {out}")
    if candidate_chars < tokenizer_chars:
        raise ValueError("Candidate budget must cover tokenizer character budget")
    references, reference_manifest = load_reference(snapshot_dir)
    if tokenizer_chars != reference_manifest["tokenizer_training"]["characters_per_corpus"]:
        raise ValueError("Tokenizer character budget must equal the existing experiment budget")
    candidates, selection = acquire_candidates(iter_python_rows() if rows is None else rows, references,
                                               candidate_chars=candidate_chars, max_scanned=max_scanned,
                                               min_chars=min_chars, max_chars=max_chars)
    print(f"Acquired {len(candidates)} disjoint-family candidates; checking against {len(references)} LM files", flush=True)
    accepted, near_duplicates = filter_against_references(references, candidates, near_duplicate_threshold)
    random.Random(seed).shuffle(accepted)
    texts = exact_character_prefix([record["text"] for record in accepted], tokenizer_chars)
    retained = accepted[:len(texts)]
    excluded_families = sorted({record["repository_family"] for record in references})
    retained_families = sorted({record["repository_family"] for record in retained})
    if set(excluded_families).intersection(retained_families):
        raise ValueError("Disjoint corpus overlaps an LM repository family")
    manifest = {
        "schema_version": 1, "purpose": "disjoint_in_domain_tokenizer_ablation",
        "lm_snapshot_manifest_file_sha256": file_sha256(snapshot_dir / "manifest.json"),
        "lm_snapshot_content_sha256": reference_manifest["snapshot_sha256"],
        "source": {"path": "codeparrot/codeparrot-clean", "revision": CODEPARROT_REVISION,
                   "split": "train", "shards_read": sorted({r["source_shard"] for r in candidates if r["source_shard"]})},
        "selection": {"seed": seed, "min_chars": min_chars, "max_chars": max_chars,
                      "candidate_chars": candidate_chars, "max_scanned": max_scanned, "statistics": selection},
        "exclusion": {"lm_partitions": ["train", "validation", "test"],
                      "reference_family_count": len(excluded_families),
                      "reference_families_sha256": _canonical_hash(excluded_families),
                      "retained_family_count": len(retained_families), "family_overlap_count": 0,
                      "reference_texts_sha256": _canonical_hash(sorted(r["sha256"] for r in references)),
                      "exact_and_whitespace_normalized_texts": True},
        "near_duplicates": near_duplicates,
        "tokenizer_training": {"characters": tokenizer_chars, "unit": "Unicode code points",
                               "selection": "seeded file order; exact prefix; final file may be truncated"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as tmp:
        staging = Path(tmp) / "corpus"
        staging.mkdir()
        manifest["training"] = _write_partition(staging / "training.jsonl", texts)
        manifest["full_source"] = _write_partition(staging / "full_source.jsonl", [r["text"] for r in retained])
        metadata = staging / "metadata.jsonl"
        with metadata.open("w", encoding="utf-8") as stream:
            for record, text in zip(retained, texts):
                row = {key: value for key, value in record.items() if key not in ("text", "sha256")}
                row.update(sha256=text_sha256(text), full_source_sha256=record["sha256"],
                           training_characters=len(text), full_source_characters=len(record["text"]))
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        manifest["metadata"] = {"file": metadata.name, "file_sha256": file_sha256(metadata)}
        manifest["corpus_sha256"] = _canonical_hash(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_disjoint_corpus(staging, snapshot_dir)
        os.rename(staging, out)
    return manifest


def verify_disjoint_corpus(path, snapshot_dir):
    path, snapshot_dir = Path(path), Path(snapshot_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["corpus_sha256"] != _canonical_hash({k: v for k, v in manifest.items() if k != "corpus_sha256"}):
        raise ValueError("Disjoint corpus manifest checksum mismatch")
    if manifest["lm_snapshot_manifest_file_sha256"] != file_sha256(snapshot_dir / "manifest.json"):
        raise ValueError("Disjoint corpus references a different LM snapshot")
    loaded = {}
    for key in ("training", "full_source", "metadata"):
        spec = manifest[key]
        if spec["file"] != {"training": "training.jsonl", "full_source": "full_source.jsonl", "metadata": "metadata.jsonl"}[key]:
            raise ValueError("Unexpected disjoint corpus filename")
        file = path / spec["file"]
        if file_sha256(file) != spec["file_sha256"]:
            raise ValueError(f"Disjoint corpus {key} checksum mismatch")
        loaded[key] = [json.loads(line) for line in file.read_text().splitlines()]
        if key != "metadata":
            if any(text_sha256(row["text"]) != row["sha256"] for row in loaded[key]):
                raise ValueError("Disjoint corpus document checksum mismatch")
            if sum(len(row["text"]) for row in loaded[key]) != spec["characters"]:
                raise ValueError("Disjoint corpus character accounting mismatch")
    references, _ = load_reference(snapshot_dir)
    excluded = {row["repository_family"] for row in references}
    exact = {row["sha256"] for row in references}
    normalized = {text_sha256(" ".join(row["text"].split())) for row in references}
    metadata, sources = loaded["metadata"], loaded["full_source"]
    if len(metadata) != len(sources) or len(sources) != len(loaded["training"]):
        raise ValueError("Disjoint corpus metadata length mismatch")
    for row, source, training in zip(metadata, sources, loaded["training"]):
        if row["repository_family"] != repository_family(row["repo_name"]) or row["repository_family"] in excluded:
            raise ValueError("Disjoint corpus repository family overlap or mislabeling")
        if source["sha256"] in exact or text_sha256(" ".join(source["text"].split())) in normalized:
            raise ValueError("Disjoint corpus source text overlaps LM reference")
        if row["sha256"] != training["sha256"] or row["full_source_sha256"] != source["sha256"]:
            raise ValueError("Disjoint corpus metadata alignment mismatch")
    texts = [row["text"] for row in loaded["training"]]
    if texts != exact_character_prefix([row["text"] for row in sources], manifest["tokenizer_training"]["characters"]):
        raise ValueError("Disjoint corpus budget or source prefix mismatch")
    return texts, manifest


def build_disjoint_pair(snapshot_dir, corpus_dir, original_pair_dir, out):
    from build_tokenizer import build_bpe
    from tokenizers import Tokenizer
    out, original_pair_dir = Path(out), Path(original_pair_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite tokenizer pair: {out}")
    texts, corpus_manifest = verify_disjoint_corpus(corpus_dir, snapshot_dir)
    original = json.loads((original_pair_dir / "manifest.json").read_text())
    if original["snapshot_sha256"] != file_sha256(Path(snapshot_dir) / "manifest.json"):
        raise ValueError("Original tokenizer pair references a different snapshot")
    if original["settings"] != {"min_frequency": 2, "byte_level": True, "add_prefix_space": False}:
        raise ValueError("Unsupported original tokenizer settings")
    if original["general"]["training_chars"] != sum(map(len, texts)):
        raise ValueError("General/disjoint tokenizer character budgets differ")
    if file_sha256(original_pair_dir / "general.json") != original["general"]["sha256"]:
        raise ValueError("Original general tokenizer checksum mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as tmp:
        staging = Path(tmp) / "pair"
        staging.mkdir()
        tokenizer = build_bpe(texts, original["vocab_size"], str(staging / "fit.json"))
        tokenizer.encode_special_tokens = True
        if tokenizer.get_vocab_size() != original["vocab_size"]:
            raise ValueError("Disjoint corpus cannot support the original vocabulary size")
        for text in texts:
            if tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False) != text:
                raise ValueError("Disjoint tokenizer failed exact round trip")
        shutil.copyfile(original_pair_dir / "general.json", staging / "general.json")
        if Tokenizer.from_file(str(staging / "general.json")).get_vocab_size() != original["vocab_size"]:
            raise ValueError("General vocabulary size differs")
        manifest = {"snapshot_sha256": original["snapshot_sha256"], "vocab_size": original["vocab_size"],
                    "settings": original["settings"], "general": original["general"],
                    "fit": {"sha256": file_sha256(staging / "fit.json"), "training_chars": sum(map(len, texts)),
                            "training_bytes": sum(len(t.encode("utf-8")) for t in texts),
                            "condition": "disjoint_in_domain", "training_source": "external_python_repository_families",
                            "corpus_manifest_file_sha256": file_sha256(Path(corpus_dir) / "manifest.json"),
                            "corpus_content_sha256": corpus_manifest["corpus_sha256"],
                            "reference_family_overlap_count": 0},
                    "ablation": "Same LM snapshot and general tokenizer; fit tokenizer uses disjoint in-domain files"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.rename(staging, out)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="data/python-v1")
    parser.add_argument("--out", default="data/python-disjoint-tokenizer-v1")
    parser.add_argument("--original-pair", default="tokenizers/python-v1-4096")
    parser.add_argument("--tokenizer-out", default="tokenizers/python-disjoint-v1-4096")
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--reuse-corpus", action="store_true")
    parser.add_argument("--candidate-chars", type=int, default=8_000_000)
    parser.add_argument("--max-scanned", type=int, default=100_000)
    parser.add_argument("--attempt-log", default="research/disjoint_tokenizer_attempts.jsonl")
    args = parser.parse_args()
    attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args)}
    start = time.monotonic()
    try:
        if not args.reuse_corpus:
            manifest = prepare_disjoint_corpus(args.snapshot, args.out, candidate_chars=args.candidate_chars,
                                                max_scanned=args.max_scanned)
        else:
            _, manifest = verify_disjoint_corpus(args.out, args.snapshot)
        attempt["corpus_sha256"] = manifest["corpus_sha256"]
        if not args.corpus_only:
            pair = build_disjoint_pair(args.snapshot, args.out, args.original_pair, args.tokenizer_out)
            attempt["fit_tokenizer_sha256"] = pair["fit"]["sha256"]
        attempt["status"] = "success"
        print(json.dumps(attempt, indent=2), flush=True)
    except Exception as error:
        attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        attempt["elapsed_seconds"] = time.monotonic() - start
        attempt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        log = Path(args.attempt_log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write(json.dumps(attempt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
