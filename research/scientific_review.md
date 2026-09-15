# Scientific review and decision gates

Date: 2026-09-15. This review inspected Glyph at `07cd5ab` and the local
`../vowel` checkout, including its saved results and tokenizer artifacts. It is
a design review, not a report of new transformer experiments. Implementation
may subsequently supersede the defects recorded here.

## The question worth answering

Glyph asks whether changing **the corpus used to train a tokenizer**, while
holding tokenizer algorithm, vocabulary size and language model architecture
fixed, improves a small model of a target corpus. Vowel instead changes
vocabulary size and reallocates a fixed parameter budget between embeddings
and transformer layers. These are distinct interventions.

The initial README's two-outcome interpretation is too strong: lower raw-text
loss does not by itself isolate a benefit from learned sequence modeling.
Compression can improve a useful coding metric before any model training.
For a uniform predictor over V tokens and a text encoded into N tokens,

`uniform_bits_per_byte = N * log2(V) / UTF8_bytes(text)`.

At the same V, fewer tokens immediately lowers this score. This is not a
numerical evaluation bug; it is a genuine property of the coding scheme.
Likewise, seeing more raw text and having a longer raw context are genuine
practical benefits, but they cannot establish an effect independent of those
mechanisms. The project should quantify those benefits and test the mechanisms
separately.

Predeclare three estimands:

1. **Practical benefit at a fixed model-token budget:** change in held-out
   canonical bits per byte (BPB), with identical architecture, token positions
   processed, batch shapes, update count and scheduler. Different raw data
   exposure and raw context are allowed and measured.
2. **Benefit with raw data and context controlled:** change in BPB when both
   models see the same raw chunks in the same batches, the same number of times,
   with the same raw context boundary and no truncation. This deliberately
   removes the fit tokenizer's ability to use its saved positions for extra
   text. Fixed shared padding can also hold dense tensor shapes constant.
3. **Learning beyond a static tokenizer code:** contextual-model coding gain
   over train-fitted token unigram/bigram baselines, together with untrained
   model and context-ablation measurements. These are explanatory diagnostics,
   not replacements for the absolute BPB result or proof of general capability.

The headline should name the corpus, model size, training budget, evaluation
protocol and seed variation. A tie is informative only when uncertainty is
small enough to exclude the effect size considered practically meaningful.

## What the existing evidence does and does not show

### Vowel

The local `../vowel/results.json` has four completed-looking result records and
matching checkpoint/tokenizer files. Its README still says the sweep is being
rerun. The records give the following final **reported** BPC and training time:

| Requested vocabulary | Reported BPC | Sum of recorded epoch time |
| --- | ---: | ---: |
| 1,000 | 1.7994 | 1.01 hours |
| 4,000 | 1.8500 | 0.67 hours |
| 8,000 | 1.8058 | 0.72 hours |
| 32,000 | 1.7535 | 3.90 hours |

These are historical artifacts, not validated comparable raw-text likelihoods.
`../vowel/train.py` uses `Whitespace`, caps the initial alphabet at 256, and has
no lossless decoder. Loading its actual 1K tokenizer on this machine produced:

```text
input:   "def x():\n    return  1\n"
decoded: "de f x ( ) : return 1"

input:   "漢字 🧬"
tokens:  ["[UNK]", "[UNK]", "[UNK]"]
decoded: ""
```

Whitespace is unmodeled and some Unicode content is collapsed; the raw-text
denominator includes information that is not encoded. Vowel also drops
evaluation tails, has one seed, changes model width with vocabulary, and uses
unequal token/update budgets. Its results motivate Glyph and provide rough
runtime history, but should not be pooled with Glyph as a baseline or cited as
an established vocabulary-size conclusion. Recovering a scientifically sound
Vowel sweep is a separate follow-up.

### Glyph at the reviewed commit

* The fit tokenizer is trained before the model split is materialized. With
  smoke defaults it reads the first 4,000 documents, while LM training uses the
  first 3,000 and validation the next 2,000: 1,000 validation documents are in
  tokenizer training. Non-streaming fallback splits can also leak.
* Tokenizer sample count and model sample count differ. Default fit tokenization
  sees extra in-domain documents not used for LM training; this is extra data,
  even when it is not evaluation leakage. Equal document counts between prose
  and code also need not equal text volume.
* Token windows cross document boundaries with no EOS/BOS. They drop the first
  target and residual stream tail; evaluation additionally drops the final
  incomplete batch. The numerator consequently does not describe the same
  text as the full-validation raw-character denominator.
* A zero-batch loader produces zero reported loss because of `max(vb, 1)`.
* The equal-token cap is calculated from a separately loaded corpus, before
  training's split and window/batch truncation; requested caps are not proof of
  equal actual trained tokens. Prefix truncation may change source composition.
* Vocabulary undershoot changes dimensions instead of rejecting an invalid
  comparison. Equal approximate parameter totals are not equal architectures.
* Tokenizer caches identify only label, corpus and requested vocabulary, not
  dataset revision, source hashes, sample budget or settings.
* `characters/token` is incorrectly described as lower being more compressed.
  Use explicit names (`bytes_per_token`, `chars_per_token`); larger means fewer
  tokens for the same raw text. Fit BPE is expected, not guaranteed, to compress
  an unseen target set better; training fit and held-out fit are distinct.
* Epoch-dependent cosine schedules and dropped batches obscure comparisons at
  intermediate budgets. Checkpoint state lacks enough metadata to reproduce
  or safely resume a run.

## Measurement contract

Use UTF-8 bytes for the primary denominator and Unicode code points for a
secondary BPC output. On an identical test text they differ by a fixed factor;
BPB is convenient for exact byte-level reversibility. Neither should be used
to compare intrinsic difficulty across languages without qualification.

For every raw test document or shared raw chunk, score every encoded content
token once from an explicit BOS or available within-document context. Sum
negative log probabilities in nats, divide by `log(2)` and the exact raw bytes
represented by those targets. Include the first content token; retain partial
batches. Decide whether EOS is scored and report it separately or identify the
boundary-inclusive metric explicitly. BOS and PAD are never content targets.
Report the precise context/reset policy and evaluation stride.

For a streaming scorer with token-specific windows, use all target tokens but
recognize that equal token context means unequal raw context. For a controlled
scorer, partition the raw text identically *before* tokenization, preserve all
bytes, reset each shared chunk and ensure both encodings fit. Chunking itself
changes the task; label these results as chunk-conditioned coding scores.

At minimum save per-document or per-original-source totals: nats, bytes,
characters, token count, boundary losses, and chunk count. Aggregate by sums,
not averages of document ratios. Bootstrap paired original documents, not
individual correlated tokens or chunks. If files share repository ancestry,
prefer repository-level split/bootstrap. Seed-to-seed variation is separate
from document-bootstrap uncertainty: show individual paired-seed deltas and
their mean, and do not call a document-only interval total uncertainty.

**Canonical-token caveat:** BPB from the deterministic encoder evaluates one
token path. Several token sequences can decode to the same string, so exact
decoded-string probability sums over those paths. Canonical BPB is a valid
lossless code score given the tokenizer and boundary protocol; it is not an
exact marginal string likelihood. Cao and Rimell found meaningful differences
in some settings; Geh et al. show exact marginalization is hard and canonical
and marginal values are often close in their setting. Neither result settles
this question for Glyph's small models. [Cao and Rimell (2021)](https://aclanthology.org/2021.emnlp-main.161/),
[Geh et al. (2024)](https://aclanthology.org/2024.emnlp-main.230/).

Cheap mandatory diagnostics:

* Lossless encode/decode checks over every prepared document, Unicode, repeated
  whitespace and literal strings resembling special tokens; zero content UNKs.
* Exact requested/actual vocabulary size and shared model configuration/hash.
* Uniform predictor BPB, actual initialized-model BPB, and smoothed unigram BPB
  fit only on the allowed training split. A bigram is a useful next baseline.
* Learned gain `baseline_BPB - model_BPB`, with baseline and model values shown
  separately. Comparing these gains is descriptive, not an independent metric
  of semantic understanding.
* Raw bytes/token, type usage, unused vocabulary fraction, token frequency
  histogram and segmentation examples on both train and held-out text.
* Actual optimizer updates, nonpadding training targets, processed padded
  positions, raw bytes exposed, unique data coverage, wall time and model size.
* Per-length or domain/structure-stratum loss to identify effects concentrated
  in long contexts, code indentation, repetitive names, or rare tokens.

## Feasible experiment matrix

Hardware was checked directly: Apple M3, 16 GiB unified memory, macOS 26.5.
Do not infer full-run speed from a tiny correctness run or from Vowel's
unrecorded hardware context. Train one model at a time, measure warm steady
state for both tokenizers, and record memory pressure and timing.

### Stage A: freeze the evidence substrate

Materialize and hash train, development and final test splits before any
tokenizer training. Pin dataset revision/config and filtering. Deduplicate
across splits; random document splitting does not remove near-duplicate or
same-repository leakage. Prefer repository-separated Python data when metadata
permits. Use WikiText as a **mismatched prose tokenizer**, not an all-domain
commercial tokenizer. Bound its training text by the same raw-byte budget as
the fit tokenizer. Save the exact sample identities; avoid mutable downloads
and hidden tokenizer-cache reuse.

Use development data for budget/model choices. Open a locked final test only
after these choices; adaptive experiments that inspect test outcomes must be
labeled exploratory and independently replicated.

### Stage B: learn cheaply before scaling

Start with 4K vocabulary, roughly 4M parameters, fixed architecture and context
128 or 256. Pilot one pair on the intended Python/prose contrast. A TinyStories
pair is a useful secondary contrast and diagnostic, but success on children's
stories does not replace the primary code experiment.

Record 0-step, early, middle and final development scores. Train enough for the
transformer to beat static baselines and for the paired difference to become
stable; an early-training win alone does not establish an endpoint advantage.
Budget extensions must use a predeclared scheduler policy and valid resume
state, or restart both arms with the extended schedule.

### Stage C: primary paired controls

| Family | Held constant | Allowed difference | Recommended initial replication |
| --- | --- | --- | --- |
| Fixed-token budget | Architecture, fixed token tensor shapes, updates, target-token budget | Raw exposure and raw context | 3 paired seeds |
| Shared raw chunks | Identical raw chunks, batch order, data passes, context boundaries, updates; shared fixed padding | Number of nonpadding targets and their segmentation | 3 paired seeds |
| Context ablation | Same trained checkpoint and held-out chunks | Available preceding raw context | Evaluate both families; no retraining |
| Static baselines | Same training split and held-out chunks | Tokenizer-induced code and frequency model | Once per tokenizer/split |

For shared chunks, choose a Unicode-safe byte limit and deterministically
subdivide any chunk that exceeds the shared token capacity under either
tokenizer. Apply resulting identical chunks to both arms. Use the same padded
tensor length in both arms if claiming equal dense-compute budget; dynamic
padding is acceptable if measuring, rather than equating, compute. Train loss
should use a declared normalization: summed content loss divided by raw batch
bytes makes each batch's source text carry the same scale. The conventional
token-mean objective changes gradient scaling when token counts differ; retain
that alternative as a sensitivity check if the outcome is small or unstable.

An equal-raw-data experiment with separately packed token windows is also
useful, but does not control context or update count and should not be confused
with the stronger shared-chunk experiment.

### Stage D: follow the result

* If the fixed-token gain disappears with shared raw chunks, quantify data and
  context contributions with exposure-matched and context-limited evaluations.
  Do not call the original gain fake: report the useful mechanism.
* If a gain survives the raw-chunk control, check that it exceeds a static-code
  baseline difference, survives a larger budget and appears across seeds.
* If the sign changes across seeds or budgets, add seeds and extend both arms
  before adding new tokenizer algorithms. Five paired seeds provide a clearer
  variance view than three, though neither is large-sample evidence.
* Repeat the decisive comparison at one larger architecture (roughly 12–15M
  parameters) and, if informative, a second vocabulary such as 8K. This tests
  whether the result is a very-small-model or vocabulary-allocation artifact.
* Add an in-domain tokenizer trained on **disjoint** in-domain documents to
  separate distribution fit from sharing exact lexical examples with LM
  training. A half-target/half-prose tokenizer is a useful dose-response check
  if the fit/mismatch result is large and consistent.
* Replicate on one different corpus only after the first contrast is understood.
  Avoid a broad low-budget grid of undertrained models.

The direct precedent is Dagan et al.'s controlled code-tokenizer work: tokenizer
training data and design can affect effective context, memory and downstream
behavior. It supports testing these mechanisms, not assuming their direction
at Glyph's scale. [Dagan et al. (2024)](https://proceedings.mlr.press/v235/dagan24a.html).
Goldman et al.'s compression-as-uniform-code framing motivates the mandatory
zero-gram baseline; their downstream findings do not make compression alone a
sufficient evaluation for this project. [Goldman et al. (2024)](https://aclanthology.org/2024.findings-acl.134/).

## Decision gates and stopping rule

1. **Correctness:** exact raw accounting, no held-out tokenizer exposure,
   lossless encoding, same actual vocabulary/config, finite losses, nonempty
   loaders, reproducible manifests, tested resume. Any failure invalidates
   downstream inference until corrected.
2. **Learnability:** both models improve substantially over their initialized
   scores and beat useful static baselines on development data. Otherwise
   investigate optimization/data before declaring a tokenizer conclusion.
3. **Fair budget:** actual counters prove the named budget. Equal token count
   is not automatically equal wall time; equal epochs are not equal updates.
4. **Robustness:** show all paired seeds, stable late learning curves, and the
   decisive raw-context/exposure control. Predeclare a meaningful difference,
   for example 1% relative BPB as an operational threshold, rather than equating
   non-significance with equivalence. This threshold is a proposed choice,
   not a literature-established constant.
5. **Mechanism:** characterize how much of the result is visible before
   contextual learning and how the result changes with context/exposure.
   If mechanisms cannot be uniquely separated, say so rather than overclaim.
6. **Scope:** the conclusion survives one scale or corpus replication, or is
   explicitly restricted to the measured regime with that limitation justified.
7. **Diminishing returns:** stop only when the likely interpretation and useful
   effect bound remain unchanged after the predeclared replication/control
   checks. Keep inconclusive small-effect outcomes inconclusive. Archive all
   failed and superseded runs with their invalidation reasons.

## Follow-up issues worth preserving

* Rebuild Vowel with a reversible byte-level tokenizer and exact raw scoring;
  its current historical BPC is not a clean source-text coding metric.
* Quantify canonical versus marginalized string probabilities on a tractable
  tiny-string subset; do not claim general marginalization from sampled paths.
* Test domain-mixture and disjoint in-domain tokenizer training to map the
  difference between memorized lexicon and distribution adaptation.
* If language-modeling gains are strong, add an appropriate tokenization-free
  behavioral endpoint. Tiny models may be at floor on HumanEval; do not use a
  floor-level benchmark or handpicked generations as evidence of equivalence.
* Generalize source-group deduplication and repository-separated code splits
  before claiming broad code generalization.
