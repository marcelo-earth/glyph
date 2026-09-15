# Review of the auditable experiment runner

Reviewed 2026-09-15, while `experiment.py`, `corpus_snapshot.py` and
`evaluation.py` were actively being developed. This report reviews the
`experiment.py` contents with `train_run` at line 158 and final completion at
line 266; line numbers will drift after fixes. No implementation was edited.
The evaluation module was mid-refactor, so its temporary syntax/structure
problems were deliberately not treated as runner findings.

## Required fixes before inferential runs

### 1. Resume identity excludes the implementation and backend

**Priority: high. Locations:** configuration at lines 169–176 and restore at
209–222.

The configuration hash contains data and tokenizer hashes but not source-code
hashes, device, or package/runtime compatibility. Editing model, training or
evaluation code between runs leaves the hash unchanged. A resumed run then
silently mixes implementations. Restoring the old `result` also replaces the
newly collected git/runtime metadata, hiding the change. A changed completed
run can likewise be returned as if its results belong to current code.

**Fix:** record hashes of all executed project modules and relevant dependency
versions in the run identity. Require compatible device/backend for deterministic
resume, or explicitly fork and label a nonidentical continuation. Hash content,
not just HEAD: the runner currently permits dirty trees. Save a code snapshot
or exact diff if results are intended to survive later edits. Strict source
identity may include comments; a restart requirement is preferable to silent
mixture, and a deliberate override should create a new run identity.

**Gate:** interrupt after a checkpoint, resume with identical code/backend, and
compare final model, scheduler, optimizer, order/cursor, RNG and exposure with
an uninterrupted run. Test across a data-permutation rollover. A source change
must reject resume. CPU exact equality and MPS reproducibility/tolerance should
be reported separately; restoring RNG state alone does not prove identical
backend arithmetic.

### 2. The runner never validates the tokenizer's training provenance

**Priority: high. Locations:** `build_tokenizers` manifest at 79–99;
`train_run` loading at 167–175.

Tokenizer construction records the snapshot and file hash, but training does
not read this manifest. Any byte-level tokenizer file—including one built on
test data—can be passed with `label='fit'`. Its hash provides identity, not
proof that its training corpus was allowed. Single-run calls also do not
enforce matching vocabulary/architecture/initial weights between the arms.

**Fix:** for primary experiments require a verified tokenizer manifest linking
the selected label, tokenizer hash, settings and vocabulary to the current
snapshot. Add a paired-run validator (or orchestration gate) that checks shared
architecture, actual vocabulary, initial weight hash, snapshot, steps, batch
shape, scheduler and regime. For raw runs also require equal raw exposures and
the same raw-batch order/hash. Exploratory external tokenizers can use a clearly
separate provenance mode.

**Gate:** mismatched snapshots, swapped labels, altered tokenizer files, unequal
vocabularies and deliberately changed paired hyperparameters fail before
training or before any paired result is labeled valid.

### 3. Raw evaluation drops original document/group identity

**Priority: high for uncertainty claims. Locations:** lines 195–197 and 224–227.

`raw_blocks(validation, seq_len)` returns `parents`, but the runner never uses
or saves it. `evaluate_documents` therefore sees each chunk as an independent
document. Later bootstrapping these records treats correlated chunks from the
same file as independent observations, overstates sample size and obscures
source-group leakage. `val_docs` in configuration no longer matches the
reported evaluation document count.

**Fix:** retain block-to-document identity, original document hashes and the
source-group identity from the snapshot. Aggregate per-block nats/counts by
original document for reporting, and retain source groups for grouped
bootstrap. Preserve block sufficient statistics for inspection if useful.

**Gate:** changing the chunk length changes block count but leaves original
document/group counts and total raw bytes invariant. Paired bootstrap samples
the same original units for both tokenizers and keeps all blocks together.

### 4. Packed-run byte exposure is wrong at multibyte boundaries

**Priority: medium/high for exposure attribution. Locations:** offsets at
124–129 and prefix differences at 152–153.

ByteLevel offsets refer to Unicode character spans; byte tokens within one
code point can share an offset. The monotone-end rule allocates every byte of
that code point to its first byte token. Summing an entire document conserves
bytes, but shuffled or budget-limited windows do not represent that allocation.

Empirical reproduction with a 260-entry byte vocabulary and `seq_len=4`:

```text
raw source: abc🧬x
window 0 targets: a b c <first emoji byte>
reported raw_bytes: 7; actual target bytes: 4

window 1 targets: <remaining three emoji bytes> x
reported raw_bytes: 1; actual target bytes: 4
```

**Fix:** count exact source bytes represented by the byte-level token pieces,
using the reversible ByteLevel alphabet mapping and zero bytes for inserted
boundaries. Special-token recognition must first be controlled. For characters,
choose and name a convention because a partial UTF-8 character has no natural
integer character count: report completed characters, proportional byte
coverage, or keep character counts only for whole raw blocks. Also report the
discarded stream tail and unique coverage; cumulative exposure counts repeats.

**Gate:** test multibyte characters crossing windows, a budget ending inside
one, repeated windows and stream tails. Byte target lengths must equal raw UTF-8
bytes when all content targets are included.

### 5. Roundtrip checks do not prevent literal control-token recognition

**Priority: medium; potentially higher on code with sentinel strings.
Locations:** lines 89–93 and every `tokenizer.encode` call.

The comment says special-token strings in text must not become controls, but
`decode(..., skip_special_tokens=False) == text` does not enforce this. Direct
verification found `[EOS]` encoding as ID 3 and decoding back to `[EOS]`, so it
passes. Content can therefore reuse the same IDs as inserted EOS/BOS/PAD.

**Fix:** set `tokenizer.encode_special_tokens = True` wherever a tokenizer is
loaded/used (the name means tokenize special strings through the ordinary
model), then reject special IDs in encoded raw content and verify roundtrip.
In the installed library this turns literal `[EOS]` into ordinary byte IDs.
`add_special_tokens=False` alone disables post-processing; it does not solve
special-string recognition. Verify whether the flag survives serialization;
otherwise reset it explicitly after `Tokenizer.from_file`.

**Gate:** all four literal special strings in raw train/evaluation text
roundtrip and contain no control IDs. Manually inserted BOS/EOS remain valid.

### 6. Tokenizer construction overwrites supposedly reusable artifacts

**Priority: medium. Locations:** `build_tokenizers` lines 78–99.

An existing output directory is accepted and both JSONs and the manifest are
overwritten. If training settings, source snapshot or library version changes,
old runs can lose their referenced tokenizer artifact. If construction fails
after writing one tokenizer, the directory can contain a mismatched pair.

**Fix:** stage the complete pair and manifest, then atomically publish to a new
directory. Refuse replacement of an existing artifact, or verify and reuse
only an exact matching artifact. Prefer immutable identifiers as with corpus
snapshots.

## Scientific interpretation and reporting fixes

### Evaluation currently measures only shared short raw contexts

Both training regimes are evaluated using common `seq_len`-byte blocks, not
natural documents or fixed-token contexts. This is a valid controlled endpoint
and makes the result particularly interpretable for the raw regime. However,
the practical fixed-token regime's inference-context advantage is deliberately
removed by this endpoint. Its result cannot quantify the full practical
benefit of fitting more raw text into the token window.

Save both (a) the common-raw-block score and (b) an exact whole-document scorer
with an explicitly declared fixed-token context/reset policy. Give them
different names. Neither is universally more correct: they answer different
questions. Evaluate static baselines and initialization under both policies
as appropriate; the runner currently records initialization but never calls
the static-baseline evaluator.

### Tokenizer training budgets match characters, not bytes

`corpus_snapshot.py` and `build_tokenizers` enforce equal Unicode code points.
This is a coherent, documented budget; the earlier scientific review proposed
equal bytes. Report both totals and explicitly predeclare the choice. If the
byte imbalance is material, use an equal-byte sensitivity check. Do not claim
the present character-matched tokenizers saw identical byte volumes.

### Regime differences extend beyond exposure and context

Packed training scores inserted EOS tokens and crosses document boundaries;
raw training resets BOS per raw block and omits EOS. Packed training uses
token-mean loss; raw training uses loss per byte. These are intentional and
reasonable within-arm controls, but changing regimes changes several things
at once. A disappearing effect across regimes is evidence compatible with an
exposure/context mechanism, not a unique mediation estimate.

Record EOS target counts and report content targets separately. If the regime
comparison becomes decisive, add a targeted boundary/normalization sensitivity
check rather than attributing the entire difference to context or exposure.

### Runtime is model-step time, not end-to-end time

The timer starts after batch construction and device transfer; evaluation,
checkpointing, tokenization and data loading are excluded. This is a useful
training-kernel measure. Name it accordingly and record total wall time when
making practical throughput claims. Synchronization around steps is already
present and is a good choice.

### Preserve input validation and termination semantics

Reject nonpositive steps, batch size, evaluation interval, sequence length,
model dimensions and validation count, invalid regimes passed through Python,
and nonfinite/nonpositive learning rates. In particular, `steps=0` currently
marks an initialized model complete and `eval_every=0` fails mid-run. Validate
before creating output artifacts. If a run fails after metrics are written,
record its failure reason while retaining the last valid checkpoint.

## What is already sound in the runner

* Snapshot loading verifies content hashes and exact cross-split disjointness;
  fit-tokenizer inputs are reconstructed from training only. This repairs the
  original held-out-tokenizer leakage for honest, verified snapshot artifacts.
* Raw block boundaries depend only on raw UTF-8 text and `seq_len`, so both
  tokenizers see the same chunks. The byte cap is conservative: with byte-level
  BPE each ordinary token represents at least one byte, so content fits the
  target capacity even under a byte vocabulary.
* Raw batches use the same seeded permutations and constant padded tensor
  shapes. Subject to validated pair configuration, identical chunk ordering
  controls raw exposure and update counts.
* Right-padding labels use `-100`. Causal attention prevents right padding from
  affecting preceding valid token logits; lack of a key-padding mask is not a
  defect for these sequences. Confirm with a padding-invariance test.
* Raw loss divides summed content NLL by the exact raw batch bytes. Packed
  loss divides by actual targets including EOS. These are explicit objectives;
  their numerical values must not be compared as the same unit.
* Model, optimizer, scheduler, data permutation/cursor and backend torch RNG
  states are checkpointed. The save order is mostly conservative: a crash
  after metrics but before checkpoint publication replays from the prior
  checkpoint rather than skipping unpersisted updates. Tests still need to
  prove continuous-versus-resumed equality.
* Checkpoint and metrics writes use replace-on-complete temporary files.

## Suggested integration-test priorities

1. Continuous versus interrupted/resumed training across an epoch/permutation
   boundary, plus rejection of source/backend/provenance mismatch.
2. Identical raw-regime batch indices, byte exposure, padded positions, shared
   initialization and architecture for fit/general.
3. Parent-document aggregation and complete raw accounting on Unicode, short
   documents, partial windows and unequal encoded lengths.
4. Byte-level exposure with a multibyte character crossing packed windows.
5. Literal special strings and padding invariance.
6. Tokenizer immutability and incomplete-pair construction recovery.

These gates target concrete scientific failure modes; passing a tiny
end-to-end training run alone does not establish them.
