# Glyph experiment log

## 2026-09-15 — Audit and protocol repair (in progress)

Objective: isolate the practical and learned effects of corpus-fit byte BPE at fixed vocabulary and transformer architecture. Distinguish compression, raw-text exposure, context span, and learned modeling effects. The sibling Vowel varies vocabulary size and model width; its results are background, not a quantitative baseline for Glyph.

Starting state: commit `07cd5ab`; pre-existing uncommitted `data.py` change switches WikiText to Salesforce parquet mirror, preserved. No Glyph results or checkpoints in repository. Existing fit TinyStories tokenizer cannot be reused without a split/provenance manifest.

Confirmed defects: fit-tokenizer held-out contamination; validation drops windows/batches and scales by unscored full-corpus token/character ratio; empty evaluation reports zero loss; vocabulary shortfall can alter architecture; documentation reverses characters/token interpretation; checkpoint/tokenizer cache provenance incomplete; token cap is computed on different docs from the nonstreaming fallback split.

Offline evidence: with 1,000 validation tokens, length 128, batch 16, existing evaluator scores zero targets; with 3,000 tokens it scores only 2,048. MPS was unavailable inside earlier sandbox but is available with current unrestricted environment (torch 2.8.0, Apple M3, 16GB).

Planned gates:
1. Independent data/evaluation verification and analytic tests.
2. Immutable train/validation/test snapshot, deduplicated, equal raw-character tokenizer training budgets.
3. Same actual vocabulary, architecture, initialization seed and explicit exposure accounting.
4. Uniform and train-unigram coding baselines; validation pilot before test evaluation.
5. Paired-seed equal-token and equal-raw-block experiments. Evaluate common raw blocks as context control. Compare learned savings beyond baselines; do not equate canonical-path NLL with marginal string likelihood.
6. Iterate model/budget/data controls based on evidence; record failures and uncertainty. Keep test held out from tuning.

No results or conclusions claimed yet.

## 2026-09-15 — Data preparation and first corrected pilot

Commits: `ddd19a3` exact evaluation and protocol; `488907e` paired training, snapshots and diagnostics; `2f24dde` Python preparation. Existing WikiText-source change was committed separately as `25280a3` while this task was working; preserved.

Primary Python source failure: Stack-smol requires authentication. Replaced the source for the new explicit snapshot, not silently in the legacy loader, with pinned CodeParrot-clean. `data/python-v1` has 9,158 training files, 301 validation files, 300 test files. Whole repository families are disjoint; 241 lexical near-duplicates removed (239 across repositories). Same-name forks group together; renamed/semantic clones remain a limitation. Python train totals 50,553,666 characters. Each tokenizer sees 4m characters.

Secondary snapshot `data/tinystories-v1`: 11,000 train, 500 validation, 500 test stories; exact deduplication only. Frozen content hashes, but original source revision was not recorded, so final primary evidence prioritizes pinned Python. Both secondary tokenizers have exactly 4,096 types. Old tokenizer files and newly observed legacy `results.json` are not corrected-pipeline evidence and are preserved without modification.

30 offline tests now cover scoring, literal controls, Unicode byte accounting, full CPU checkpoint replay, snapshot integrity and repository grouping. The lexical prefix candidate filter matches exhaustive greedy Jaccard filtering on generated test cases.

### TinyStories raw-block pilot, seed 42, 500 updates

Same 2,590,848-parameter architecture, same initialized weight hash, 128-byte raw blocks, 16 blocks/update, byte-normalized objective. Validation: first 100 fixed validation stories (80,650 bytes). Final test not scored.

| Coding score (bits/UTF-8 byte) | Fit | Mismatched WikiText tokenizer |
|---|---:|---:|
| Uniform | 3.14024 | 3.99355 |
| Train unigram on matching raw blocks | 2.30855 | 2.86163 |
| Initialized transformer | 3.14328 | 4.00879 |
| Transformer after 500 updates | 2.08189 | 2.59608 |
| Gain over unigram | 0.22666 | 0.26555 |

Interpretation: fit has a useful absolute coding advantage, but its transformer gap (0.51419 BPB) is smaller than its static-unigram gap (0.55308 BPB). This pilot does not support an additional advantage from contextual learning. It is one early-training seed on a secondary corpus, not a final conclusion. The cosine schedule reaches zero at 500; its flat endpoint does not establish convergence. Extend with a fresh shared longer schedule if informative.

Next action: complete 500-update paired Python development pilots under raw-block and packed-token training, compare static gains, then choose adequate longer schedules and replication. Do not select on held-out test scores.

### Python development pilot, seed 42, 500 updates — completed

Both paired invariant audits passed. Fixed 50-file development subset: 273,747 UTF-8 bytes. Same 2.59M architecture and 128-byte evaluation blocks.

| BPB ↓ | Fit | Mismatched |
|---|---:|---:|
| Uniform, shared blocks | 4.10806 | 7.43215 |
| Full-train unigram, shared blocks | 3.29542 | 4.30680 |
| Raw-block trained transformer | 3.24036 | 3.70862 |
| Packed-token trained transformer | 3.13334 | 3.61708 |

Raw-block transformer gap = -0.46825 BPB; packed-token gap = -0.48374 BPB (fit minus mismatched). Both are smaller than the -1.01138 static-unigram gap. The fit raw-block transformer improves only ~0.055 BPB over a full-training unigram. This fails the intended substantial contextual-learning gate for a final inference. The early advantage is not sufficient to choose an endpoint conclusion.

Baseline-data-access check added: `exposure_baseline.py` reconstructs the exact sampler and verifies all exposure counters before fitting an add-one unigram on the actually trained targets. The fit raw-block exposure unigram scores 3.29989 BPB (vs full-train 3.29542), so its small learned gain persists under that correction. Other arms' checks in progress. This distinguishes baseline access to the full corpus from the ~1m-byte raw pilot exposure.

MPS and CPU checkpoint replay now pass exact-weight tests across sampler wraparound. The fixed-token pair trains precisely 1,024,000 targets each; raw-block pair sees precisely 1,014,716 bytes each, with different target token counts.

Decision before further evaluation: run a new 5,000-update development matrix, seed 42, both regimes, same architecture/data and 50 validation files, evaluate every 500. Fresh initialization and shared 5,000-step cosine schedule; do not extend a schedule already annealed to zero. This tenfold budget tests whether the tokenizer ranking and learned gains persist as optimization proceeds. Replicated seeds and a scale/context control remain required after choosing an adequate budget. Final test remains unscored.

### Correction: accelerator replay precision

The first isolated MPS replay passed exact equality, but full-suite/repeated checks subsequently failed bitwise equality. Three repeated first-weight maximum differences were 7.45e-9, 1.49e-8, and 9.31e-10. CPU remains exact. The GPU regression now requires all weights within atol=1e-7/rtol=1e-6, score within 1e-6 BPB, and exact exposure/source/initialization identity. This documents FP32 numerical replay rather than claiming bitwise MPS determinism. Completed pilot models were uninterrupted; this does not invalidate their scoring. Longer-run numerical drift is not bounded by this four-update regression and remains a reproducibility caveat.

Further five-repeat measurement isolated larger drift to attention projection biases (max 2.32e-6); all other parameter differences were <=2.98e-8, and the strict score/counter checks passed. The test now keeps 1e-7 absolute tolerance for all ordinary parameters and Q/V biases, allowing 1e-5 only on attention **key** biases. A shared key bias cancels from attention softmax mathematically; numerical near-zero gradients can be amplified by Adam. This is a specific numerical tolerance, not permission for general weight drift. The measurements are saved in `research/mps_replay_diagnostic.json`; that diagnostic replaced weight assertions with measurement and is not itself the final passing test.
