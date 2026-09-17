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

### Longer development matrix in progress

The 5,000-update Python raw-fit arm completed at 2.37403 BPB, vs 3.24036 in the 500-update pilot. Its exposure-matched unigram is 3.29542, so learned gain is now 0.92139 BPB. The same-schedule general arm must finish before comparing endpoints. The matrix continues sequentially through both packed-token arms. No ranking or stopping decision is made from a lone completed arm. Draft PR #1 contains the pipeline and pilot evidence; `research/STATUS.md` records verified process handles and remaining scientific gates.

Visualization verification: the Python pilot SVG/PNG was rendered and visually inspected; final labels, common scales, baseline encodings and captions were checked against source evidence. The figure is intentionally restricted to the registered 500-update pilot and has a machine-readable source hash/row companion.

## 5,000-update Python matrix completed; seed replication started

All four seed-42 arms completed, with both paired audits passing. Raw-control BPB: fit 2.3740306391, mismatched 2.5643821724 (delta -0.1903515333). Packed-token BPB on the same raw-context evaluation: fit 2.2913040422, mismatched 2.5057602091 (delta -0.2144561670). Both models now beat static unigrams substantially; the general tokenizer makes up much of its larger initial/static gap. This is a descriptive decomposition, not a causal claim that a percentage of the advantage is due to compression, nor a claim that general tokenization has superior semantic modeling.

The same 5,000-update matrix is now being replicated with seeds 43 and 44 in `experiments/python-replication-5k`. Keep both complete seed pairs regardless of outcomes. The independent review prioritizes a fresh 20,000-update schedule afterward: current raw runs see ~20% of available source bytes, and cosine endpoint flatness is not convergence evidence. Full-validation endpoints, context-policy sensitivity and the disjoint-in-domain tokenizer ablation are being prepared independently. Test remains unscored.

User explicitly requested committing all this work. The historical root `results.json` is preserved in Git as legacy smoke evidence; its prototype leakage/evaluation limitations remain clearly documented and it is not used in corrected analyses. Large local data/checkpoints retain existing ignore rules; verified source-level evidence and exact tokenizers are versioned.

## Full validation and disjoint tokenizer preparation

Full validation on all 301 Python files (1,654,670 UTF-8 bytes) confirms the seed-42 raw-pair ordering: fit 2.297952253 versus general 2.495968825 BPB, difference -0.198016572. Scores were computed on CPU with two threads while seed replication continued on MPS. These are validation results, not final test results. Source uncertainty does not establish training-seed robustness.

The packed 5k exposure unigrams are 3.300873660 (fit) and 4.307984404 (general), giving model gains 1.009569617 and 1.802224195 BPB. Regenerated paired analysis includes both.

Prepared the disjoint in-domain tokenizer using the same pinned CodeParrot revision: 808 files, 790 repository families, exactly 4 million Unicode characters, no family overlap with any LM split. Filtering removed 129 lexical near duplicates (124 matching LM files). General tokenizer is unchanged. This is preparation only; no ablation model has been trained. The bounded source prefix, family-name heuristic and lexical clone threshold remain limitations.

All 15 new evaluator, disjoint-preparation and seed-summary tests passed together on CPU. New implementation, artifacts and independent interim review are committed as requested.

## Seed-43/44 replication matrix complete

The replication matrix (session 88636) finished all eight arms: raw/token regimes crossed
with fit/general tokenizers at seeds 43 and 44. Every pair audit passed and each seed pair
shares its initial weight hash, matching the seed-42 provenance already recorded.

`summarize_replications.py` pools seeds 42/43/44 (`research/snapshots/python-seed-replication-summary.json`):
raw paired difference −0.190495 BPB (seed range −0.192797 to −0.188337, sample SD 0.002233);
token paired difference −0.210112 BPB (seed range −0.214456 to −0.203319, sample SD 0.005959).
The corpus-fit tokenizer's advantage over the general tokenizer is consistent in sign and
magnitude across three independent training seeds, in both regimes. This is a development
finding at 5,000 updates only; the independent review's recommended 20k-update replication
and the disjoint-in-domain tokenizer ablation remain outstanding, and the final test is
still unscored.

## Fresh 20,000-update development matrix and extension decision

All four seed-42 arms of `experiments/python-development-20k` completed with a
new 20,000-step cosine schedule. This is not a resume of the 5k checkpoint.
Both pair audits pass: raw fit/general share 40,490,022 raw bytes and padded
positions, while packed fit/general share 40,960,000 target tokens and padded
positions. Exposure-matched unigram gains on the 50-file monitor are 1.3431 /
2.2592 BPB for raw fit/general and 1.4349 / 2.2610 BPB for packed fit/general.

The complete 301-file, 204-family development validation split was scored on
CPU in the primary 128-byte shared-raw-block protocol. Raw fit/general score
1.887215393 / 1.981981061 BPB, a fit-minus-general difference of −0.094765668
BPB. Packed-trained fit/general score 1.813698833 / 1.981879587 BPB under that
same evaluation, a difference of −0.168180754 BPB. Repository-family bootstrap
95% intervals are [−0.104921, −0.084891] and [−0.178282, −0.158056],
respectively. They quantify validation-source variation only, not training-seed
or tokenizer-sample uncertainty.

The raw-control gap has fallen from the 5k three-seed monitor mean of about
−0.1905 BPB to −0.0948 BPB on full validation, while both raw arms improve by
well over 1% relative BPB from 5k to 20k. The interim-review extension condition
is therefore met. Decision before seeing any 40k endpoint: run a fresh seed-42
40k shared-raw-block pair with the same architecture, data, optimizer and
paired controls. Score full validation at the endpoint. Do not open the final
test, and do not treat this single 20k seed as a replacement for a longer-budget
seed replication.

## Full-validation context and exposure diagnostics at 20k

Shorter shared raw-block evaluations were run on all four completed checkpoints
with matching full-training static baselines. Raw-trained fit-minus-general
differences are −0.094766, −0.075889 and −0.039154 BPB at 128, 64 and 32 bytes.
Packed-trained differences under those same raw evaluations are −0.168181,
−0.113953 and −0.064358 BPB. Their repository-family bootstrap intervals all
exclude zero, but they reflect development-source sampling conditional on one
training seed. The fit tokenizer's full-training unigram advantage is −0.965104
BPB at 64-byte blocks and −0.892986 at 32-byte blocks, so static coding remains
a substantial part of the complete-system difference under each segmentation.

An explicitly named 128-token chunk evaluation of whole source documents gives
a contrasting result. The raw-trained pair has fit-minus-general +0.379614 BPB,
whereas the packed-trained pair has −0.301310 BPB. This endpoint permits
tokenizer-dependent raw history, and raw-trained checkpoints were not optimized
for it. It is therefore a context-policy diagnostic rather than a replacement
for the primary raw-block result, a semantic-quality endpoint, or a causal
decomposition of the tokenizer effect.

## Fresh 40,000-update raw-control extension

The predeclared fresh 40k raw-control pair completed at seed 42. Pair auditing
confirms identical raw-character/raw-byte exposure (80,932,794 / 80,988,420)
and padded positions (81,920,000), with shared initialized weights. On the
50-file during-training monitor, fit/general score 1.760387 / 1.800995 BPB.
Exposure-matched unigram gains are 1.535043 / 2.505784 BPB.

On the authenticated full 301-file validation endpoint, fit/general score
1.699065437 / 1.744520098 BPB, a fit-minus-general difference of −0.045454661
BPB. The repository-family bootstrap 95% interval is [−0.053228, −0.037953].
This is a development-source interval conditional on one training seed.

Both 40k arms improve materially from 20k, yet the primary raw-control gap
narrows again, from −0.094766 to −0.045455 BPB. Decision after this endpoint:
run the matching fresh 40k packed-token pair before choosing a longer-budget
replication or a model-scale check. Do not open the final test or recast the
raw-only extension as a complete practical-regime result.

## Fresh 40,000-update packed-token extension

The matching fresh 40k packed pair completed at seed 42 with exactly
81,920,000 target tokens and padded positions per arm. The 50-file monitor
ends at 1.722764 / 1.819660 BPB for fit/general. Exposure-matched unigram gains
are 1.577989 / 2.488503 BPB.

Full validation under the primary 128-byte raw-block endpoint gives fit/general
1.673130757 / 1.769944707 BPB, a fit-minus-general difference of −0.096813950
BPB and a repository-family bootstrap 95% interval [−0.104338, −0.089188]. The
separate 128-token chunk diagnostic scores 1.326565350 / 1.556944776 BPB, a
difference of −0.230379426. Those intervals condition on this single training
seed; the token endpoint permits tokenizer-dependent raw history.

Both packed arms improve materially from their 20k values, but the primary
raw-block advantage narrows from −0.168181 to −0.096814 BPB. Decision before
seeing another endpoint: run a fresh seed-42 80k shared-raw-block pair under
the same architecture, optimizer and evaluation monitor. Score full validation
at its endpoint, preserve all outcomes, and keep the final test sealed.

## Disjoint in-domain tokenizer static check

Before allocating a model-training run, the prepared disjoint tokenizer pair
was diagnosed on full validation in 128-byte raw blocks. The disjoint fit
tokenizer uses exactly 4 million Python characters from repository families
that are absent from all LM train/validation/test splits; the general WikiText
tokenizer and LM snapshot are unchanged. The full-training unigram endpoint is
3.228509 BPB for disjoint fit and 4.241392 for general, a fit-minus-general
difference of −1.012883 BPB with a source-bootstrap 95% interval
[−1.046051, −0.984261]. This records a static coding diagnostic only. A paired
model-training ablation is still required to test whether that result survives
contextual learning with no tokenizer/LM family overlap.
