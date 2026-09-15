# Research follow-ups and completion gates

## Required before a final Glyph conclusion

- [x] Independent Vowel and Glyph scientific audit.
- [x] Exact reversible coding evaluation, raw-byte control, static baselines.
- [x] Immutable Python snapshot with repository-family split and lexical near-deduplication.
- [x] CPU checkpoint resume regression; code/runtime/snapshot identities enforced.
- [x] Verify MPS checkpoint resume across sampler wraparound within declared FP32 tolerances; CPU exact. Longer-run numerical drift remains a caveat.
- [ ] Complete paired development pilots in both regimes on Python; measure learning vs unigram.
- [ ] Choose adequate full training budget on development only, replicate ≥3 paired seeds.
- [ ] Context and exposure mechanism diagnostics; uncertainty at original repository units for Python.
- [ ] Decisive comparison at larger model/budget; domain-disjoint fit tokenizer if informative.
- [ ] Lock final evaluation choice, score held-out test once, report all seeds and uncertainty.
- [ ] Reproducibility package, inspectable plots, honest restricted conclusion and stopping audit.

## Promising separate research issues

1. Rebuild Vowel's vocabulary-size sweep with reversible byte tokenization. Existing saved tokenizer discards whitespace and collapses Unicode; prior BPC does not score all source information.
2. Measure exact string marginalization on a tractable toy subset to quantify canonical-path evaluation bias.
3. Train domain-mixed and disjoint-in-domain tokenizers with matched text budgets to distinguish distribution adaptation from exact lexical overlap.
4. Add a tokenization-independent behavioral endpoint only once model quality exceeds the task floor; handpicked generations are insufficient.
5. Broaden code-family grouping beyond same-name repository forks and surface lexical near duplicates.
