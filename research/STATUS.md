# Active research status

This is a progress checkpoint, not a completion claim. The full goal remains active.

## Current work

- Branch: `codex/reliable-tokenizer-experiments`
- Draft PR: https://github.com/marcelo-earth/glyph/pull/1
- Completed: scientific/Vowel audit; corrected data/evaluation/training; six 500-update pilot runs; matched full-data and exposure unigrams; paired-budget audits; source-level uncertainty; scientific plot.
- Completed 5k matrix, seed 42: raw fit/general **2.3740306391 / 2.5643821724**; token fit/general **2.2913040422 / 2.5057602091**. Both paired audits passed.
- Final test has not been scored.

## Live job (verify before acting)

The original 5k matrix (session 57918, PID 97571) is terminal/completed. Do not restart it.

New replication matrix launched in unified exec session **88636**:

```
.venv/bin/python run_matrix.py --snapshot data/python-v1 --tokenizers tokenizers/python-v1-4096 --out experiments/python-replication-5k --steps 5000 --eval-every 500 --val-docs 50 --seeds 43 44
```

Verify its session/process before acting. Order: raw seed43 pair, raw seed44 pair, token seed43 pair, token seed44 pair. Frozen core implementation files must not change while this matrix is live, because each new arm hashes them for provenance.

Independent agents are reviewing interpretation (`research/INTERIM_REVIEW.md`), adding posthoc checkpoint evaluation (`evaluate_checkpoint.py`), and preparing a disjoint-in-domain tokenizer (`prepare_disjoint_tokenizer.py`). Parent handles commits after verification. No final-test scoring authorized by the current development protocol.

## Next scientific actions

1. Complete the 5k matrix, verify both pair audits, export completed evidence, analyze against the reused matching full-training diagnostics and exposure-matched baselines.
2. Decide adequate training budget from development learning curves and static gains. The 500-update fit model was undertrained; the longer fit arm learns substantially more. A cosine endpoint flattening is not a convergence proof.
3. Replicate the decisive comparison over at least three paired seeds. Evaluate full validation at endpoints, preserving repository grouping. Add longer-budget/model/context controls and disjoint-domain tokenizer ablation if informative.
4. Lock final evaluation choices before scoring test. Finish only after the protocol's robustness, mechanism and uncertainty gates; preserve genuine null/inconclusive results.

## Reproducibility details

36 tests exist. Latest full suite (35 before export test) passed; the export regression passed separately. CPU resume is exact. MPS requires FP32 tolerances; key-bias drift is specifically documented in `EXPERIMENT_LOG.md` and `mps_replay_diagnostic.json`. Do not claim bitwise GPU determinism.

Per-block `metrics.json` and checkpoints remain local. Committed `evidence.json` preserves all per-source statistics and original metrics hashes. Published evidence was checked to reproduce the complete local Python pilot analysis exactly. Exact tokenizer artifacts are committed; their builder intentionally refuses overwrites.

Pre-existing/newly observed root `results.json` belongs to the legacy prototype and has been left untouched. It is not corrected-pipeline evidence. The earlier WikiText source fix was independently committed as `25280a3` and preserved.
