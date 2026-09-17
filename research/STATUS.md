# Active research status

This is a progress checkpoint, not a completion claim. The full goal remains active.

## Current work

- Branch: `codex/reliable-tokenizer-experiments`
- Draft PR: https://github.com/marcelo-earth/glyph/pull/1
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

## 20k development extension (complete)

A new seed-42 20,000-update schedule completed all four arms with fresh cosine
schedules, not a continuation of the 5k checkpoint. The raw and packed pair
audits pass. On the during-training 50-file monitor, raw fit/general are
1.952328 / 2.047652 BPB and packed fit/general are 1.865939 / 2.047112 BPB.

Authenticated CPU evaluation of the complete 301-file validation split under
the primary 128-byte raw-block protocol gives raw fit/general 1.887215 /
1.981981 BPB (difference −0.094766, repository-family bootstrap 95% interval
[−0.104921, −0.084891]) and packed fit/general 1.813699 / 1.981880 BPB
(difference −0.168181, interval [−0.178282, −0.158056]). These intervals cover
only development-source sampling; this schedule has one training seed. The
20k raw gap is materially smaller than the three-seed 5k monitor gap, so the
budget is not locked and the final test remains sealed.

## 20k context diagnostics (complete)

The full validation raw-block endpoint remains fit-favoring as raw context is
shortened, but the difference decreases with the available history. Raw-trained
fit-minus-general BPB is −0.094766, −0.075889 and −0.039154 at 128, 64 and 32
bytes. Packed-trained fit-minus-general BPB is −0.168181, −0.113953 and
−0.064358. All six repository-family bootstrap intervals exclude zero, but all
refer to this one training seed and development set.

The static full-training unigram advantage also remains large at the short
contexts: −0.965104 BPB at 64 bytes and −0.892986 at 32 bytes. The explicit
128-token chunk endpoint changes the intervention: it favors general for the
raw-trained pair (+0.379614 BPB fit minus general) and fit for the packed-trained
pair (−0.301310). Those are context-policy diagnostics, not alternate primary
endpoints or estimates of a tokenizer's intrinsic modeling capability.

## Disjoint tokenizer static diagnostic (complete)

The prepared disjoint-in-domain tokenizer was checked on all 301 validation
files in shared 128-byte blocks before model training. It uses 4 million
characters from repository families absent from every LM split. Its full-training
unigram scores 3.228509 BPB versus 4.241392 for the fixed WikiText tokenizer,
a difference of −1.012883 BPB (source-bootstrap interval [−1.046051,
−0.984261]). Its compression and static code advantage therefore resemble the
original in-domain fit condition. This is not an ablation result until paired
LM training uses the same snapshot and declared budget.

## 40k raw-control extension (complete)

The fresh seed-42 40,000-update raw pair passes its paired audit with exactly
80,988,420 raw bytes and 81,920,000 padded positions in each arm. The full
validation primary endpoint is 1.699065 fit versus 1.744520 general BPB, a
fit-minus-general difference of −0.045455 BPB (204-family bootstrap 95% interval
[−0.053228, −0.037953]). Both arms improve by roughly 10% to 12% relative BPB
from the 20k endpoint, but the controlled gap falls again from −0.094766 BPB.

The source-level interval excludes zero only conditional on this seed. This
continued shrinking gap means 40k is not an adequate final budget choice and
the packed 40k pair is required before selecting longer-budget seed replication
or a larger-model scope check. Final test remains sealed.

## Next scientific actions

1. ~~Complete the seed-43/44 replication matrix, verify paired audits and export evidence.~~
   Done: all three seeds agree in sign and magnitude for both regimes (see above).
2. Run a fresh 40k packed-token seed-42 pair, then score its full-validation raw-block and fixed-token-context endpoints. The 40k raw control still improves materially but its fit advantage shrinks, so both training regimes are needed before deciding on a final budget or seed replication.
3. Train the disjoint-in-domain tokenizer ablation now that preparation is verified. Do not compare a reduced-LM-data variant with the current runs as though only its tokenizer changed.
5. Lock final evaluation choices before scoring test. Finish only after the protocol's robustness, mechanism and uncertainty gates; preserve genuine null/inconclusive results.

## Reproducibility details

36 tests exist. Latest full suite (35 before export test) passed; the export regression passed separately. CPU resume is exact. MPS requires FP32 tolerances; key-bias drift is specifically documented in `EXPERIMENT_LOG.md` and `mps_replay_diagnostic.json`. Do not claim bitwise GPU determinism.

Per-block `metrics.json` and checkpoints remain local. Committed `evidence.json` preserves all per-source statistics and original metrics hashes. Published evidence was checked to reproduce the complete local Python pilot analysis exactly. Exact tokenizer artifacts are committed; their builder intentionally refuses overwrites.

Pre-existing/newly observed root `results.json` belongs to the legacy prototype and has been left untouched. It is not corrected-pipeline evidence. The earlier WikiText source fix was independently committed as `25280a3` and preserved.

New verification: the full suite (51 tests, including checkpoint-evaluator, disjoint-tokenizer
and seed-summary tests) passed together on CPU. Full-validation checkpoint artifacts retain
authenticated protocol and source scores.
