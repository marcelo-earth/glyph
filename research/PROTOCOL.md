# Glyph experimental protocol — version 1

Status: development pilots, no final test opened for model selection.

## Hypothesis and scope

At fixed BPE algorithm, vocabulary size and transformer architecture, a tokenizer learned from the target training distribution may reduce held-out canonical coding cost relative to one learned from mismatched prose. This can happen through static compression, data exposure, effective context, or differences in learned conditional prediction. Lower bits per byte alone does not separate these mechanisms.

Vowel varies vocabulary size and trades embedding capacity for transformer width. Glyph fixes both. Historical Vowel raw-text results are not a validated baseline (see scientific_review.md).

## Primary and secondary data

Primary target: Python code. The original Stack-smol source is gated and failed without authentication on 2026-09-15. Prepare a pinned, accessible CodeParrot snapshot, retaining repository provenance and grouping before splitting. Secondary target: TinyStories. Mismatched tokenizer source: WikiText-2 prose (not a commercial general-purpose tokenizer).

Target train, development and test splits precede tokenizer fitting. Freeze exact text files and checksummed manifests. Fit and mismatched tokenizers each receive exactly 4,000,000 Unicode code points. Record UTF-8 bytes separately. Differences in token counts, unique types and source complexity are part of the domain intervention, not matched quantities. The fit tokenizer uses a subset of LM training text only. A later disjoint-in-domain ablation can distinguish domain fit from exact shared lexical examples.

## Interventions

Both tokenizers: reversible byte-level BPE, vocabulary exactly 4,096 including four control tokens, minimum merge frequency 2, same byte alphabet and pretokenizer. Literal control-token strings are encoded as ordinary text. Any vocabulary shortfall fails the comparison.

Pilot model: width 192, 4 heads, 4 transformer layers, tied input/output embeddings, context capacity 128, dropout 0.1 (2,590,848 parameters). Paired seed 42 for development. AdamW learning rate 0.0003, weight decay 0.01, norm clipping 1.0, cosine schedule defined over the whole run. Pilot: 500 updates, batch 16; evaluate at 0, 100, 200, 300, 400 and 500 updates. This is a learning/optimization diagnostic, not the final budget. Longer runs restart both arms with the same longer schedule.

1. **Packed fixed-token run:** full 128-target windows in a concatenated training stream with EOS boundaries. Same number of updates, targets, padded positions and architecture. Dataset windows shuffled without replacement, cycling as necessary. Residual stream tail is untrained, recorded by dataset length. Raw byte exposure differs; counters reflect actual token byte content including partial Unicode code points, with zero raw bytes for control targets. Character exposure counts code points completed by a target token. Prefix fragments are not independently decodable examples.
2. **Shared raw-block control:** split source documents into identical contiguous blocks of at most 128 UTF-8 bytes, without splitting a code point. Both byte BPE encodings fit the context. Same blocks, same batch order, same update count and fixed 128-position padded tensor shapes. BOS per block, content-only targets. Sum token NLL divided by raw batch bytes. This holds training text exposure and available raw context boundaries constant; padding deliberately spends saved positions. It is a distinct training task from packed windows; cross-family differences cannot be attributed solely to exposure.

Paired seeds share initialization hashes, but token IDs name different strings: identical weights are a controlled initialization, not a semantic embedding alignment. GPU reproducibility is checked empirically; CPU checkpoint resume has exact-weight regression coverage. Checkpoints record model, optimizer, scheduler, RNGs, sampler, counters, implementation hashes and runtime. Resume rejects incompatible configurations.

## Evaluation contract

Primary controlled development score: all content tokens in identical shared 128-byte raw blocks from the same documents, each conditioned on BOS and preceding tokens within its block. Sum NLL in nats; divide by log(2) and exact original UTF-8 bytes. Also report Unicode-codepoint BPC. No EOS scored; no omitted first token or partial batch. Aggregate blocks back to source documents/repositories for uncertainty. A token-context score and shorter raw-context evaluations are diagnostics, labeled separately.

This is the probability of one deterministic token encoding, not the marginal probability of a decoded string across all tokenizations. Report compression/uniform predictor, train-fitted smoothed unigram and initialized-transformer scores. Show absolute BPB and gains over static baselines; such gains are descriptive and do not establish semantic capability.

Development pilots use a fixed prefix of 100 validation documents; full validation is used for later confirmation. Final test remains unused until budgets, seeds and decisive controls are chosen. Test data may be hashed/validated during loading but is not scored or used to tune choices.

## Decision rules

- Fix correctness failures before interpreting any runs. Preserve failed/superseded evidence and reasons.
- Both transformer arms should beat useful static baselines; otherwise investigate optimization or extend both schedules.
- Replicate decisive comparisons with at least three paired seeds. Report every paired difference and seed variation separately from document/repository bootstrap intervals.
- Use 1% relative BPB as an operational meaningful-effect threshold, not a universal scientific constant. Non-significance is not equivalence.
- If results survive the raw-block control, test a larger budget and model, and consider a disjoint-in-domain tokenizer. If they disappear, quantify context/exposure mechanisms without calling practical compression benefits artificial.
- Finish only when the controls, replication and uncertainty support a restricted defensible conclusion and additional planned experiments have diminishing interpretive value. Track promising extensions as issues/TODOs.
