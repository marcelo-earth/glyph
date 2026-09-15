"""Freeze disjoint text splits and equal-sized tokenizer training corpora.

Splits are sampled from the configured source's training pool, not its official
benchmark splits. Exact text deduplication precedes splitting; near duplicates
are not detected. A snapshot, rather than a mutable upstream dataset, is the
authoritative input for subsequent experiments.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile


SCHEMA_VERSION = 1
PARTITIONS = ("train", "validation", "test", "general", "fit_tokenizer", "general_tokenizer")


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text_sha256(encoded)


def _unique_documents(texts, excluded=()):
    seen = set(excluded)
    unique = []
    removed = 0
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every document must be a nonempty text string")
        digest = text_sha256(text)
        if digest in seen:
            removed += 1
        else:
            unique.append(text)
            seen.add(digest)
    return unique, removed


def split_documents(texts, val_docs, test_docs, seed=0):
    """Deduplicate, shuffle deterministically, and split once at document level."""
    if val_docs < 1 or test_docs < 1:
        raise ValueError("Validation and test must each contain at least one document")
    unique, _ = _unique_documents(texts)
    if len(unique) <= val_docs + test_docs:
        raise ValueError("Not enough unique target documents to leave a training split")
    random.Random(seed).shuffle(unique)
    return {
        "validation": unique[:val_docs],
        "test": unique[val_docs:val_docs + test_docs],
        "train": unique[val_docs + test_docs:],
    }


def exact_character_prefix(texts, n_chars):
    """Take exactly n Unicode code points, preserving document boundaries.

    Only the last retained document may be truncated. No separators are added;
    a Python string slice cannot split a UTF-8 encoded code point. This budget
    counts code points, not bytes or grapheme clusters.
    """
    if n_chars < 1:
        raise ValueError("Character budget must be positive")
    result = []
    remaining = n_chars
    for text in texts:
        if remaining == 0:
            break
        part = text[:remaining]
        if part:
            result.append(part)
            remaining -= len(part)
    if remaining:
        raise ValueError(f"Corpus has {n_chars - remaining} characters; requested {n_chars}")
    return result


def _write_partition(path, texts):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text, "sha256": text_sha256(text)}, ensure_ascii=False) + "\n")
    return {
        "file": path.name,
        "documents": len(texts),
        "characters": sum(map(len, texts)),
        "utf8_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "ordered_text_sha256": _canonical_hash([text_sha256(text) for text in texts]),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def prepare_snapshot(out, *, corpus, general_corpus, max_docs, val_docs, test_docs,
                     seed=0, general_max_docs=None, tokenizer_chars=None, loader=None,
                     source_specs=None):
    """Write an immutable snapshot and return its verified loaded contents.

    max_docs/general_max_docs cap documents retained by load_corpus *before*
    exact deduplication. loader/source_specs can be injected for offline tests.
    The default tokenizer budget is the smaller available character count;
    tokenizer_chars requests an explicit budget and fails if either is short.
    """
    if max_docs < 1 or (general_max_docs is not None and general_max_docs < 1):
        raise ValueError("Document limits must be positive")
    if corpus == general_corpus:
        raise ValueError("Target and general corpora must have different names")
    if loader is None:
        from data import CORPORA, load_corpus
        loader = load_corpus
        if source_specs is None:
            source_specs = CORPORA
    if source_specs is None:
        raise ValueError("An injected loader requires source_specs for provenance")
    target_spec = dict(source_specs[corpus])
    general_spec = dict(source_specs[general_corpus])
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to replace snapshot: {out}")
    general_max_docs = max_docs if general_max_docs is None else general_max_docs
    target_raw = list(loader(corpus, max_samples=max_docs, min_chars=50, streaming=True))
    general_raw = list(loader(general_corpus, max_samples=general_max_docs, min_chars=50, streaming=True))
    target_unique, target_removed = _unique_documents(target_raw)
    target_hashes = {text_sha256(text) for text in target_unique}
    general_unique, general_removed = _unique_documents(general_raw, excluded=target_hashes)
    if not general_unique:
        raise ValueError("General corpus is empty after deduplication and target exclusion")
    splits = split_documents(target_unique, val_docs, test_docs, seed)
    random.Random(seed + 1).shuffle(general_unique)
    splits["general"] = general_unique
    available_chars = {key: sum(map(len, splits[key])) for key in ("train", "general")}
    budget = min(available_chars.values()) if tokenizer_chars is None else tokenizer_chars
    splits["fit_tokenizer"] = exact_character_prefix(splits["train"], budget)
    splits["general_tokenizer"] = exact_character_prefix(splits["general"], budget)
    sources = {
        "target": {"name": corpus, "spec": target_spec, "max_docs": max_docs},
        "general": {"name": general_corpus, "spec": general_spec, "max_docs": general_max_docs},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "source_specs_sha256": _canonical_hash(sources),
        "split": {"seed": seed, "val_docs": val_docs, "test_docs": test_docs,
                  "method": "Python random.Random shuffle of unique source-training documents"},
        "loading": {"streaming": True, "min_chars": 50},
        "deduplication": {
            "method": "exact UTF-8 text SHA-256; target first, general excludes all target documents",
            "target_loaded": len(target_raw), "target_removed": target_removed,
            "general_loaded": len(general_raw), "general_removed": general_removed,
            "near_duplicates_checked": False,
        },
        "tokenizer_training": {
            "characters_per_corpus": budget,
            "character_unit": "Unicode code points; no inserted separators",
            "selection": "ordered prefix; final document truncated if necessary",
            "fit_source": "train", "general_source": "general",
            "available_characters": available_chars,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete snapshot, leaving no half-written destination.
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as tmp:
        staging = Path(tmp) / "snapshot"
        staging.mkdir()
        manifest["partitions"] = {
            name: _write_partition(staging / f"{name}.jsonl", splits[name])
            for name in PARTITIONS
        }
        manifest["snapshot_sha256"] = _canonical_hash(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staging, out)
    return load_snapshot(out)


def load_snapshot(path, verify=True):
    """Load {train, validation, test, general, *_tokenizer, manifest}.

    Verification checks all recorded content hashes, cross-split disjointness,
    the fit tokenizer's training-only provenance, and equal character budgets.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported snapshot schema")
    if verify:
        identity = {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
        if manifest["snapshot_sha256"] != _canonical_hash(identity):
            raise ValueError("Snapshot manifest checksum mismatch")
        if manifest["source_specs_sha256"] != _canonical_hash(manifest["sources"]):
            raise ValueError("Source specification checksum mismatch")
    result = {"manifest": manifest}
    for name in PARTITIONS:
        spec = manifest["partitions"][name]
        if spec["file"] != f"{name}.jsonl":
            raise ValueError(f"Unexpected partition filename: {name}")
        raw = (path / spec["file"]).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        texts = [row["text"] for row in rows]
        if verify:
            if hashlib.sha256(raw).hexdigest() != spec["file_sha256"]:
                raise ValueError(f"Partition file checksum mismatch: {name}")
            hashes = [text_sha256(text) for text in texts]
            if any(row["sha256"] != digest for row, digest in zip(rows, hashes)):
                raise ValueError(f"Document checksum mismatch: {name}")
            actual = (len(texts), sum(map(len, texts)),
                      sum(len(text.encode("utf-8")) for text in texts), _canonical_hash(hashes))
            expected = (spec["documents"], spec["characters"], spec["utf8_bytes"], spec["ordered_text_sha256"])
            if actual != expected:
                raise ValueError(f"Partition statistics mismatch: {name}")
        result[name] = texts
    if verify:
        seen = set()
        for name in ("train", "validation", "test", "general"):
            texts = result[name]
            unique, removed = _unique_documents(texts, excluded=seen)
            if removed or not unique:
                raise ValueError(f"Empty, duplicated, or overlapping partition: {name}")
            seen.update(text_sha256(text) for text in texts)
        budget = manifest["tokenizer_training"]["characters_per_corpus"]
        for name, source in (("fit_tokenizer", "train"), ("general_tokenizer", "general")):
            if result[name] != exact_character_prefix(result[source], budget):
                raise ValueError(f"Tokenizer training provenance mismatch: {name}")
        if "repository_metadata" in manifest:
            repo_seen, family_seen = set(), set()
            for name in ("train", "validation", "test"):
                spec = manifest["repository_metadata"][name]
                if spec["file"] != f"{name}.metadata.jsonl":
                    raise ValueError(f"Unexpected metadata filename: {name}")
                raw = (path / spec["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != spec["file_sha256"]:
                    raise ValueError(f"Repository metadata checksum mismatch: {name}")
                rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
                if [row["sha256"] for row in rows] != [text_sha256(text) for text in result[name]]:
                    raise ValueError(f"Repository metadata alignment mismatch: {name}")
                repos = {row["repo_name"].casefold() for row in rows}
                families = {row["repository_family"] for row in rows}
                if not all(repos) or not all(families):
                    raise ValueError(f"Missing repository identity: {name}")
                if repo_seen.intersection(repos) or family_seen.intersection(families):
                    raise ValueError(f"Repository or repository-family overlap: {name}")
                if len(repos) != spec["repositories"] or len(families) != spec["repository_families"]:
                    raise ValueError(f"Repository metadata statistics mismatch: {name}")
                repo_seen.update(repos)
                family_seen.update(families)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--general-corpus", required=True)
    parser.add_argument("--max-docs", type=int, required=True)
    parser.add_argument("--general-max-docs", type=int)
    parser.add_argument("--val-docs", type=int, required=True)
    parser.add_argument("--test-docs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer-chars", type=int)
    parser.add_argument("--out", required=True)
    args = vars(parser.parse_args())
    out = args.pop("out")
    snapshot = prepare_snapshot(out, **args)
    print(json.dumps(snapshot["manifest"], indent=2))


if __name__ == "__main__":
    main()
