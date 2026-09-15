"""Prepare a pinned Python-vs-prose experiment with repository-disjoint holdouts.

CodeParrot-clean supplies GitHub Python files and repository identifiers. This
preparer keeps complete files within explicit size bounds, deduplicates before
splitting, and groups all repositories with the same case-folded basename.
The basename rule conservatively groups many forks, but cannot identify renamed
forks. Content filtering removes exact, whitespace-normalized, and high-Jaccard
5-token-shingle duplicates; it does not establish semantic independence.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import urllib.request

from corpus_snapshot import (PARTITIONS, SCHEMA_VERSION, _canonical_hash,
                             _unique_documents, _write_partition,
                             exact_character_prefix, load_snapshot, text_sha256)


CODEPARROT_REVISION = "35a59fb025bc0a102f7d96eac09d145b896d487b"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CODEPARROT_BASE = f"https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/{CODEPARROT_REVISION}"
WIKITEXT_URL = (f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKITEXT_REVISION}/"
                "wikitext-2-raw-v1/train-00000-of-00001.parquet")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def repository_family(repo_name):
    """Conservative fork-family proxy; different owners with same basename group."""
    if not isinstance(repo_name, str) or repo_name.count("/") != 1:
        raise ValueError("Expected owner/repository identity")
    owner, name = repo_name.split("/")
    if not owner or not name:
        raise ValueError("Empty repository identity")
    return name.casefold()


def iter_python_rows():
    """Read pinned gzip shards sequentially; closing stops the network stream."""
    for shard in range(1, 55):
        filename = f"file-{shard:012d}.json.gz"
        with urllib.request.urlopen(f"{CODEPARROT_BASE}/{filename}", timeout=90) as response:
            with gzip.GzipFile(fileobj=response) as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    row["source_shard"] = filename
                    row["source_line"] = line_number
                    yield row


def load_general_texts():
    """The small pinned WikiText2 parquet needs one bounded (~6 MB) download."""
    import pyarrow.parquet as pq
    with urllib.request.urlopen(WIKITEXT_URL, timeout=90) as response:
        raw = response.read()
    table = pq.read_table(io.BytesIO(raw), columns=["text"])
    return [text for text in table.column("text").to_pylist() if text and len(text.strip()) >= 50]


def select_files(rows, max_docs=10_000, max_scanned=100_000, min_chars=200, max_chars=20_000):
    """Bound acquisition and reject whole files; never truncate model documents."""
    if min(max_docs, max_scanned, min_chars) < 1 or max_chars < min_chars:
        raise ValueError("Invalid positive corpus/size bounds")
    selected, seen, normalized_seen = [], set(), set()
    stats = Counter()
    try:
        for row in rows:
            stats["scanned"] += 1
            text = row.get("content")
            if not isinstance(text, str) or not min_chars <= len(text) <= max_chars:
                stats["outside_size_bounds"] += 1
            elif not str(row.get("path", "")).lower().endswith(".py"):
                stats["non_python_path"] += 1
            else:
                try:
                    family = repository_family(row.get("repo_name"))
                except ValueError:
                    stats["missing_repository"] += 1
                    family = None
                if family is not None:
                    digest = text_sha256(text)
                    normalized = text_sha256(" ".join(text.split()))
                    if digest in seen:
                        stats["exact_duplicates"] += 1
                    elif normalized in normalized_seen:
                        stats["whitespace_duplicates"] += 1
                    else:
                        selected.append({"text": text, "sha256": digest,
                                         "repo_name": row["repo_name"], "repository_family": family,
                                         "path": row["path"], "license": row.get("license"),
                                         "source_shard": row.get("source_shard"),
                                         "source_line": row.get("source_line")})
                        seen.add(digest)
                        normalized_seen.add(normalized)
            if len(selected) >= max_docs or stats["scanned"] >= max_scanned:
                break
    finally:
        close = getattr(rows, "close", None)
        if close:
            close()
    if len(selected) < max_docs:
        raise ValueError(f"Only {len(selected)} acceptable files after scanning {stats['scanned']}; requested {max_docs}")
    stats["selected_before_shingle_filter"] = len(selected)
    return selected, dict(stats)


def token_shingles(text, width=5):
    tokens = TOKEN_PATTERN.findall(text)
    if len(tokens) < width:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def deduplicate_shingles(records, threshold=0.85):
    """Greedy exact set-Jaccard filtering with complete prefix candidate search.

    For a pair with Jaccard >= t, intersection >= ceil(t * size) for both
    sets. Its first shared shingle must therefore occur within each set's
    prefix of length size-ceil(t*size)+1, under one global shingle order.
    Every qualifying pair against a retained record is thus considered. The
    filter compares surface lexical 5-grams, including comments/string text;
    syntactic/semantic code clones with low surface overlap remain possible.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Jaccard threshold must be in (0, 1]")
    # Intern strings/tuples to compact integer IDs while retaining exact equality.
    vocabulary, sets = {}, []
    for record in records:
        ids = set()
        for shingle in sorted(token_shingles(record["text"])):
            if shingle not in vocabulary:
                vocabulary[shingle] = len(vocabulary)
            ids.add(vocabulary[shingle])
        sets.append(ids)
    frequencies = Counter(item for items in sets for item in items)
    index = defaultdict(list)
    retained, removed, comparisons = [], [], 0
    for i in sorted(range(len(records)), key=lambda j: (len(sets[j]), j)):
        items = sets[i]
        prefix_length = len(items) - math.ceil(threshold * len(items)) + 1
        prefix = sorted(items, key=lambda item: (frequencies[item], item))[:prefix_length]
        candidates = {j for item in prefix for j in index[item]
                      if len(sets[j]) >= threshold * len(items)}
        duplicate = None
        for j in sorted(candidates):
            comparisons += 1
            overlap = len(items.intersection(sets[j]))
            similarity = overlap / (len(items) + len(sets[j]) - overlap)
            if similarity >= threshold:
                duplicate = {"removed_sha256": records[i]["sha256"],
                             "retained_sha256": records[j]["sha256"],
                             "jaccard": similarity,
                             "cross_repository": records[i]["repo_name"] != records[j]["repo_name"]}
                break
        if duplicate:
            removed.append(duplicate)
        else:
            retained.append(i)
            for item in prefix:
                index[item].append(i)
    return [records[i] for i in sorted(retained)], {
        "method": "exact lexical 5-token-shingle set Jaccard with complete global-prefix candidate index",
        "threshold": threshold, "removed": len(removed), "verified_candidate_pairs": comparisons,
        "cross_repository_removed": sum(item["cross_repository"] for item in removed),
        "removals": removed,
        "limitation": "No semantic-clone detection; comments and strings contribute lexical tokens",
    }


def split_repositories(records, val_docs=300, test_docs=300, seed=2026):
    """Assign whole repository families, meeting minimum holdout file counts."""
    if min(val_docs, test_docs) < 1:
        raise ValueError("Positive validation and test document minima required")
    groups = defaultdict(list)
    for record in records:
        groups[record["repository_family"]].append(record)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    result = {"train": [], "validation": [], "test": []}
    for name in names:
        split = ("validation" if len(result["validation"]) < val_docs else
                 "test" if len(result["test"]) < test_docs else "train")
        result[split].extend(groups[name])
    if not result["train"] or len(result["validation"]) < val_docs or len(result["test"]) < test_docs:
        raise ValueError("Insufficient repository families for nonempty disjoint splits")
    for i, records_in_split in enumerate(result.values()):
        random.Random(seed + i + 10).shuffle(records_in_split)
    return result


def prepare_python_snapshot(out, *, max_docs=10_000, max_scanned=100_000,
                            val_docs=300, test_docs=300, seed=2026,
                            min_chars=200, max_chars=20_000, tokenizer_chars=4_000_000,
                            near_duplicate_threshold=0.85, rows=None, general_texts=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to replace snapshot: {out}")
    acquired, loading_stats = select_files(iter_python_rows() if rows is None else rows,
                                          max_docs, max_scanned, min_chars, max_chars)
    print(f"Acquired {len(acquired)} Python files; checking lexical near duplicates", flush=True)
    records, near_duplicates = deduplicate_shingles(acquired, near_duplicate_threshold)
    grouped = split_repositories(records, val_docs, test_docs, seed)
    splits = {name: [record["text"] for record in values] for name, values in grouped.items()}
    general = load_general_texts() if general_texts is None else general_texts
    general, general_removed = _unique_documents(general, excluded={record["sha256"] for record in acquired})
    random.Random(seed + 1).shuffle(general)
    splits["general"] = general
    for name, source in (("fit_tokenizer", "train"), ("general_tokenizer", "general")):
        splits[name] = exact_character_prefix(splits[source], tokenizer_chars)
    sources = {
        "target": {"name": "python", "spec": {"path": "codeparrot/codeparrot-clean",
                   "revision": CODEPARROT_REVISION, "split": "train", "text_column": "content",
                   "shards_read": sorted({record["source_shard"] for record in acquired if record["source_shard"]})},
                   "max_docs": max_docs},
        "general": {"name": "wikitext2", "spec": {"path": "Salesforce/wikitext",
                    "revision": WIKITEXT_REVISION, "name": "wikitext-2-raw-v1", "split": "train",
                    "text_column": "text", "url": WIKITEXT_URL}},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "sources": sources,
        "source_specs_sha256": _canonical_hash(sources),
        "split": {"seed": seed, "val_docs_minimum": val_docs, "test_docs_minimum": test_docs,
                  "method": "Shuffle sorted repository families, assign whole families to holdout minima then train",
                  "group": "case-folded repository basename; conservatively groups same-name forks",
                  "limitation": "Renamed forks can evade family grouping; near-duplicate filtering adds content control",
                  "repository_overlap_count": 0, "repository_family_overlap_count": 0},
        "loading": {"streaming": True, "min_chars": min_chars, "max_chars": max_chars,
                    "max_scanned": max_scanned, "sampling": "bounded sequential prefix of pinned shards",
                    "selection_stats": loading_stats, "file_truncation": False},
        "deduplication": {"target_exact_and_whitespace": loading_stats,
                          "general_removed": general_removed, "near_duplicates_checked": True,
                          "near_duplicates": near_duplicates},
        "tokenizer_training": {"characters_per_corpus": tokenizer_chars,
                               "character_unit": "Unicode code points; no inserted separators",
                               "selection": "ordered prefix; final document truncated if necessary",
                               "fit_source": "train", "general_source": "general",
                               "available_characters": {key: sum(map(len, splits[key])) for key in ("train", "general")}},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as tmp:
        staging = Path(tmp) / "snapshot"
        staging.mkdir()
        manifest["partitions"] = {name: _write_partition(staging / f"{name}.jsonl", splits[name]) for name in PARTITIONS}
        metadata = {}
        for name, values in grouped.items():
            file = staging / f"{name}.metadata.jsonl"
            with file.open("w", encoding="utf-8", newline="\n") as stream:
                for record in values:
                    stream.write(json.dumps({k: v for k, v in record.items() if k != "text"}, sort_keys=True) + "\n")
            metadata[name] = {"file": file.name, "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                              "repositories": len({record["repo_name"].casefold() for record in values}),
                              "repository_families": len({record["repository_family"] for record in values})}
        manifest["repository_metadata"] = metadata
        manifest["snapshot_sha256"] = _canonical_hash(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Verify group identities and every content hash before publishing.
        load_snapshot(staging)
        os.rename(staging, out)
    return load_snapshot(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/python-v1")
    for name, default in (("max-docs", 10_000), ("max-scanned", 100_000), ("val-docs", 300),
                          ("test-docs", 300), ("seed", 2026), ("min-chars", 200),
                          ("max-chars", 20_000), ("tokenizer-chars", 4_000_000)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--attempt-log", default="research/python_preparation_attempts.jsonl")
    options = vars(parser.parse_args())
    attempt_log = Path(options.pop("attempt_log"))
    attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "options": options}
    try:
        snapshot = prepare_python_snapshot(**options)
        attempt.update(status="success", snapshot_sha256=snapshot["manifest"]["snapshot_sha256"],
                       partition_documents={name: len(snapshot[name]) for name in PARTITIONS})
        print(json.dumps(attempt, indent=2), flush=True)
    except Exception as error:
        attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        attempt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        with attempt_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(attempt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
