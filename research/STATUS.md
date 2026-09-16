# Active research status

This is a progress checkpoint, not a completion claim. The full goal remains active.

## Current work

- Branch: `codex/longer-budget-controls`
- Earlier PR reference is no longer resolvable through GitHub; current work continues on the branch above.
- Completed: scientific/Vowel audit; corrected data/evaluation/training; six 500-update pilot runs; matched full-data and exposure unigrams; paired-budget audits; source-level uncertainty; scientific plot.
- Completed 5k matrix, seed 42: raw fit/general **2.3740306391 / 2.5643821724**; token fit/general **2.2913040422 / 2.5057602091**. Both paired audits passed.
- Final test has not been scored.

## Replication matrix (complete)

The seed-43/44 replication matrix (session 88636) finished all eight arms: raw and token
regimes, fit and general tokenizers, seeds 43 and 44. Both pair audits pass for every arm
and each seed pair shares its initial weight hash. No live job remains.

`summarize_replications.py` combines seeds 42/43/44 into `research/snapshots/python-seed-replication-summary.json`:

- Raw regime: fit mean 2.367082, general mean 2.557577 BPB, paired difference −0.190495
  (seed range −0.192797 to −0.188337, sample SD 0.002233).
- Token regime: fit mean 2.294199, general mean 2.504311 BPB, paired difference −0.210112
  (seed range −0.214456 to −0.203319, sample SD 0.005959).

The paired fit advantage is consistent in sign and magnitude across all three independent
training seeds. This satisfies next scientific action 1 below. Independent review, posthoc
checkpoint evaluation and the disjoint-in-domain tokenizer remain implemented and verified.
Full validation (301 files) raw seed-42 scores: fit 2.297952253 versus general 2.495968825
BPB. The disjoint tokenizer has exactly 4 million training characters and no
repository-family overlap with the LM snapshot. No final-test scoring authorized by the
current development protocol.

## Next scientific actions

1. ~~Complete the seed-43/44 replication matrix, verify paired audits and export evidence.~~
   Done: all three seeds agree in sign and magnitude for both regimes (see above).
2. Decide adequate training budget from development learning curves and static gains. The 500-update fit model was undertrained; the longer fit arm learns substantially more. A cosine endpoint flattening is not a convergence proof.
3. Run the fresh 20k-update schedule the independent review recommends (raw first, then packed), holding architecture fixed, before committing to a larger model. Evaluate full validation at endpoints, preserving repository grouping.
4. Train the disjoint-in-domain tokenizer ablation now that preparation is verified, and add context-policy diagnostics (shorter raw blocks, explicit token-context scorer) as inexpensive follow-ups.
5. Lock final evaluation choices before scoring test. Finish only after the protocol's robustness, mechanism and uncertainty gates; preserve genuine null/inconclusive results.

## Reproducibility details

36 tests exist. Latest full suite (35 before export test) passed; the export regression passed separately. CPU resume is exact. MPS requires FP32 tolerances; key-bias drift is specifically documented in `EXPERIMENT_LOG.md` and `mps_replay_diagnostic.json`. Do not claim bitwise GPU determinism.

Per-block `metrics.json` and checkpoints remain local. Committed `evidence.json` preserves all per-source statistics and original metrics hashes. Published evidence was checked to reproduce the complete local Python pilot analysis exactly. Exact tokenizer artifacts are committed; their builder intentionally refuses overwrites.

Pre-existing/newly observed root `results.json` belongs to the legacy prototype and has been left untouched. It is not corrected-pipeline evidence. The earlier WikiText source fix was independently committed as `25280a3` and preserved.

New verification: the full suite (51 tests, including checkpoint-evaluator, disjoint-tokenizer
and seed-summary tests) passed together on CPU. Full-validation checkpoint artifacts retain
authenticated protocol and source scores.

## Fresh 20k schedule launched

Unified exec session 99111 runs `run_matrix.py --snapshot data/python-v1 --tokenizers tokenizers/python-v1-4096 --out experiments/python-development-20k --steps 20000 --eval-every 2000 --val-docs 50 --seeds 42`. Fresh initialization and cosine schedule; raw pair first, packed pair second. Verify process before restarting. Core source files remain frozen during the matrix. This extends development evidence; final test remains unscored.
