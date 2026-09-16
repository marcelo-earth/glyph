# Independent interim scientific review: Python at 5,000 updates

Date: 2026-09-15. Scope: the four completed seed-42 runs in
`experiments/python-development-5k`, their evidence exports, local metrics,
pair audits, tokenizer diagnostics and current research protocol. No model
training or final-test evaluation was performed for this review.

## Assessment

**The experiment now supports a useful, restricted development finding: the
corpus-fit tokenizer has lower canonical coding cost after training, including
when raw LM exposure, raw context boundaries and padded model shapes are
matched. It does not yet establish an asymptotic advantage, an independent
semantic-modeling improvement, or a causal decomposition of the gain. Further
budget and replication experiments have substantial value.**

The four exports say `complete` at step 5,000. Their local metrics hashes match
the hashes in `analysis.json`. Both pair audits pass and all arms share the
recorded initialized weight hash. The common development endpoint contains
50 files, 43 repository families and 273,747 UTF-8 bytes, scored in identical
128-byte raw blocks. This verifies the reported comparisons' provenance;
it is not a substitute for broader validation or seed replication.

| Training regime | Fit BPB | Mismatched BPB | Fit minus mismatched | Relative fit improvement |
| --- | ---: | ---: | ---: | ---: |
| Shared raw blocks | 2.374031 | 2.564382 | −0.190352 | 7.42% |
| Packed fixed-token budget | 2.291304 | 2.505760 | −0.214456 | 8.56% |

The repository-family bootstrap intervals are respectively
`[-0.215962, -0.163067]` and `[-0.241352, -0.186748]` BPB. These quantify
development-source sampling conditional on one trained pair. They do not
include training-seed, tokenizer-training-sample or source-snapshot variation.
The final test remains outside the evidence inspected here.

## Why 5,000 updates is not yet an adequate stopping point

The raw runs have consumed the same 10,121,659 bytes, only **20.0%** of the
50,588,403-byte LM training pool. They used 80,000 of 399,792 shuffled raw
blocks, with no full pass completed. Packed fit has consumed 31,684,649 bytes
(62.6% of the pool); packed mismatched has consumed 17,128,839 bytes (33.9%).
These exposure fractions measure the present stage of training, not a
theoretical requirement to finish an epoch.

The raw fit advantage is still changing substantially through the learning
trajectory:

| Updates on the 5k schedule | Raw fit BPB | Raw mismatched BPB | Fit advantage |
| --- | ---: | ---: | ---: |
| 500 | 3.028892 | 3.439151 | 0.410259 |
| 2,500 | 2.460997 | 2.686680 | 0.225683 |
| 5,000 | 2.374031 | 2.564382 | 0.190352 |

The endpoint is flat partly because cosine annealing drives the learning rate
to zero. It does not demonstrate that more training at a useful learning rate
would stop helping, or that the tokenizer gap has reached a stable limit. Both
models now clearly beat unigram baselines, so the earlier basic-learnability
failure has been resolved. The remaining question is how the advantage evolves
with a materially larger budget.

**Recommendation:** complete the already planned seeds 43 and 44 at 5k, then
run fresh 20k schedules, with shared raw blocks first and packed runs next.
Do not revive the zero-learning-rate 5k checkpoint as if it were a continuous
20k schedule. At 20k the raw run will have seen about 80% of its block pool;
the packed arms will have repeated some source data. Record those repetitions
and avoid calling cumulative bytes unique data coverage.

Use the existing 50-file subset for intermediate monitoring if needed, but
score the **full 301-file / 204-family validation split** at the chosen endpoints
before making further design choices. This is development evaluation, not a
reason to open the final test.

## What the static-to-transformer comparison means

On the same raw-block development text, the full-training unigrams score
3.295421 BPB for fit and 4.306804 BPB for mismatched, a 1.011383 BPB advantage
before contextual modeling. After raw-block training, the corresponding
transformer gains over unigram are 0.921390 and 1.742421 BPB. Exposure-matched
unigrams are nearly identical here (3.295416 and 4.306672), so access to the
full unigram-count dataset does not explain this diagnostic.

The arithmetic identity is useful:

```text
transformer fit advantage
  = unigram fit advantage − (mismatched learning gain − fit learning gain)
  = 1.011383 − (1.742421 − 0.921390)
  = 0.190352 BPB.
```

This identity is **not** a mediation experiment. It does not establish that a
certain percentage of the advantage was caused by compression or that the
mismatched tokenizer produces a better contextual model. The two unigram
baselines operate over different symbol systems: smaller pieces leave more
within-word or within-identifier regularity for a contextual model to learn,
while a learned merge can put some of that regularity inside one symbol.
Baseline subtraction cannot uniquely separate those mechanisms.

Likewise, the smaller final transformer gap than the initial/uniform gap does
not make the final coding benefit an artifact. A corpus-fit tokenizer can be a
useful learned static code, and the combined tokenizer-plus-transformer system
can genuinely encode the held-out text more efficiently. The defensible claim
is about this complete system and measured budget. An additional claim about
semantic capability or latent “model intelligence” needs a different endpoint.

## Priority of subsequent experiments

### 1. Longer budget before larger architecture

**Highest value:** 20k paired schedules with the current model/tokenizers,
initially seed 42, both regimes. Replicate the decisive longer-budget comparison
with seeds 43/44 once its endpoint is selected. This directly tests the evident
gap shrinkage while holding architecture fixed.

Predeclare an extension rule before seeing the 20k outcome. For example, extend
to 40k or 50k if either arm improves by more than 1% relative BPB over 5k and the
paired advantage changes by a practically meaningful amount (the protocol's
1%-relative-BPB criterion is a reasonable reference). If absolute quality keeps
improving but the advantage is stable across budgets, that supports a stable
budget-specific tokenizer conclusion without requiring optimization to an
unattainable global minimum. A sign reversal or continued substantial gap
shrinkage warrants further training rather than an early endpoint declaration.

The extension rule is a proposed decision aid, not a formal statistical test.
Record the decision and all outcomes, including unfavorable ones.

### 2. Context diagnostics on existing endpoints, then targeted retraining

Evaluate the same validation documents with shorter shared raw blocks (for
example 32, 64, 128 bytes) and an explicitly named fixed-token-context scorer.
These are inexpensive compared with new model training, and can proceed after
the current GPU queue is clear. This review did not execute them.

Interpretation requires care: retokenizing shorter blocks changes both the BPE
segmentation and context resets. Recompute uniform/unigram baselines for every
block policy. A context-cropping evaluation retaining an existing canonical
tokenization can complement this check, but raw-history matching around
different token boundaries is not exact. Distinguish a diagnostic perturbation
from a clean causal intervention.

If results are very sensitive to short blocks, train a second context condition
while keeping model width/layers fixed. Avoid changing width, context, batch
size and learning budget simultaneously. The current primary endpoint excludes
the packed fit tokenizer's inference-time benefit of fitting more raw text in
128 token positions; a token-context endpoint measures a different practical
quantity and should be reported separately.

### 3. Disjoint in-domain tokenizer before claiming an intrinsic domain benefit

This is a high-value mechanism experiment once the budget is adequate. The fit
tokenizer currently learns from 4m characters in 723 LM training files. Both
LM arms have the same allowed train split, but at a partial-data budget much of
the tokenizer's in-domain training text may not yet have been sampled by the
LM. This is valid training data, not held-out leakage; it is nevertheless part
of the complete system's data access and static prior.

Construct three tokenizer arms with the same 4m-character budget: fitted to LM
training documents, fitted to disjoint Python repository families, and fitted
to prose. Keep the LM training pool identical across all three arms. Two valid
designs are:

* Acquire a fresh pinned Python pool outside all existing train/validation/test
  repository families, with content near-deduplication, for the disjoint arm.
* Reserve about 4m characters from the current **training** pool exclusively for
  tokenizer fitting, then retrain all three LM arms on the common remainder.

Do not use existing validation/test documents for this tokenizer ablation, and
do not compare a reduced-data disjoint run directly with the current larger
LM pool as though only the tokenizer changed. A disjoint-domain win similar to
the current fit arm supports transfer of distribution-specific segmentation
beyond sharing exact training examples.

### 4. One larger model as a scope check

After the longer-budget result stabilizes, use one approximately 10–15M model
with the same vocabulary and context, holding that larger architecture constant
within each pair. Train it long enough to beat its own useful baselines; copying
5k updates blindly can create a new undertraining artifact. Start with the
decisive raw control, then add the practical regime if the result changes or a
broad conclusion requires it.

This establishes whether the 2.59M result is strongly capacity-dependent. It
does not replace the budget check, and a large multi-size grid is lower value
than one interpretable replication. Do not change vocabulary size here: that
would return to Vowel's different intervention.

## Unresolved confounds and claim boundaries

1. **One matched tokenizer pair:** training seeds vary the transformer and
   sampler, not tokenizer-training corpus selection. Replication across
   in-domain tokenizer samples is separate uncertainty. The current baseline
   is WikiText-2 prose, not a representative commercial general-purpose
   tokenizer; code/prose mismatch is intentionally strong.
2. **Regime comparison changes several things:** raw blocks reset BOS, omit EOS
   targets and use byte-normalized loss; packed windows cross EOS boundaries
   and use token-normalized loss. The difference between their final gaps
   (~0.0241 BPB at seed 42) cannot be assigned solely to exposure or context.
3. **Optimizer recipe:** equal hyperparameters establish an intervention under
   one shared training recipe. They do not show each tokenizer's best achievable
   performance. If longer-budget results depend strongly on learning rate or
   clipping, run a small symmetric learning-rate sensitivity check and record
   gradient/clipping diagnostics before claiming intrinsic superiority.
4. **Dataset scope:** a pinned bounded prefix of one CodeParrot shard, file sizes
   200–20,000 characters, repository-family grouping and lexical near-dedup are
   substantial controls. Renamed/semantic clones and broader code-domain
   representativeness remain limitations. Broader validation reduces sampling
   noise within this snapshot; it does not solve distributional scope.
5. **Canonical coding metric:** these are normalized probabilities of one
   canonical token path, not exact marginalized decoded-string probabilities.
   This already appears correctly in the protocol and should remain explicit.
6. **Timing:** equal padded shapes and updates are not equal wall time or energy.
   Raw fit recorded 485.8 seconds versus 264.3 seconds for general despite the
   matched shapes; packed runs recorded 246.3 and 232.4 seconds. Treat these as
   observed run timings, not an established speed effect. A practical speed
   claim requires warm, isolated, repeated measurements with randomized order.
7. **Character-matched tokenizer data:** 4,001,559 versus 4,007,911 source bytes
   differ by about 0.16%; this is small and disclosed, but do not describe these
   as exactly byte-matched tokenizer corpora.
8. **Development selection:** repeated monitoring and next-experiment choices
   use this development subset. Repository bootstrap intervals are not
   selection-adjusted final-test evidence.

## Stopping audit for the full goal

The goal is not complete at this checkpoint. A defensible finish still requires:

* completed paired-seed replication;
* a justified budget based on at least one substantial extension, including
  full-development endpoints rather than only the 50-file monitor;
* context/exposure diagnostics with their limits stated;
* a decisive larger-model or comparable scope replication;
* the disjoint-domain control if making an interpretation beyond exact shared
  training-corpus fit, or an explicit narrower measured interpretation;
* a locked final-test protocol and final held-out measurements;
* reproducible source-backed reporting of all runs, failures and uncertainty.

There is no evidence that the next budget, context and disjoint-domain
experiments have diminishing value yet. The project has moved from a pipeline
demonstration to a promising controlled study; the next work should determine
where its initial advantage persists and what the coding metric can actually
support.
