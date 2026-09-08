# Glyph

Does a corpus-fit tokenizer help a small language model, or does it just make the sequences shorter?

## What is this?

Train the same small transformer twice on the same corpus. The only difference is the tokenizer:

- **fit**: a BPE trained on the model's own training corpus.
- **general**: a BPE trained on a broad, different corpus, with the same vocab size and the same settings.

A corpus-fit tokenizer packs more text into each token on that corpus, so it always wins on fertility (characters per token). The question is whether the model that reads those tokens is actually better, or whether the only thing that changed is sequence length.

## How this differs from [Vowel](https://github.com/marcelo-earth/vowel)

Vowel varied the vocab **size** and held parameters constant. Glyph holds the vocab size constant and varies the tokenizer's **fit** to the corpus.

## How we measure

Per-token loss is in a different unit for each tokenizer, so it is not comparable between the two runs. The headline metric is **bits per character**: the same loss divided by the raw character count of the validation text, which is the same denominator for both runs.

Reading the result:

- `fit` and `general` land at the same bits per character, `fit` has lower fertility -> the fit tokenizer only bought shorter sequences.
- `fit` also has lower bits per character -> the fit tokenizer did real work.

## Setup

```bash
pip install -r requirements.txt

# both tokenizers, both training runs, writes results.json
python run_experiment.py

# defaults: model trains on Python code, general tokenizer is trained on WikiText
python run_experiment.py --train-corpus python --general-corpus wikitext --vocab-size 32000
```

Single pieces:

```bash
python build_tokenizer.py --corpus python --vocab-size 32000 --out tokenizers/fit.json
python train.py --tokenizer tokenizers/fit.json --corpus python
```

## Files

| File | What it does |
|------|-------------|
| `data.py` | Named corpora (`python`, `wikitext`, `tinystories`) |
| `build_tokenizer.py` | Train a byte-level BPE at a fixed vocab size |
| `model.py` | `GlyphGPT`, a small tied-embedding decoder, shared by both runs |
| `train.py` | Train one model with one tokenizer, report bits per character |
| `run_experiment.py` | Build both tokenizers, run both trainings, write `results.json` |

## Status

Scaffold. No runs yet. Numbers go here once `run_experiment.py` has finished a real run.
