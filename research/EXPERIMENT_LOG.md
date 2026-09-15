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
