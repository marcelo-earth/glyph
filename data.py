"""Corpora for Glyph.

The experiment needs two text sources that come from clearly different
distributions:

- `train`  : what the model is trained and evaluated on. The tokenizer under
             test is fit to this.
- `general`: a broad, different corpus that stands in for an off-the-shelf
             tokenizer's training data. The baseline tokenizer is fit to this.

If both corpora were the same, "corpus-fit" and "off-the-shelf" would be the
same tokenizer and there would be nothing to measure. Keep them distinct.
"""

from datasets import load_dataset


# name -> (loader args). Each loader returns a list[str] of documents.
CORPORA = {
    # Python source. Strong distribution shift from prose: indentation,
    # identifiers, punctuation-heavy lines, long shared prefixes.
    "python": {
        "path": "bigcode/the-stack-smol",
        "data_dir": "data/python",
        "split": "train",
        "text_column": "content",
    },
    # Encyclopedic prose. Clean, general, and not code.
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "train",
        "text_column": "text",
    },
    # Same source, ~40x smaller. For smoke runs.
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "text_column": "text",
    },
    # Children's stories with a deliberately tiny natural vocabulary.
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "split": "train",
        "text_column": "text",
    },
}


def load_corpus(name, split=None, max_samples=None, min_chars=50, streaming=False):
    """Return a list of documents for a named corpus.

    `min_chars` drops near-empty lines (WikiText is full of blank lines and bare
    section headers) so the tokenizer trainer and the model both see real text.

    `streaming` pulls documents lazily and stops after `max_samples` kept ones,
    so a smoke run does not download a multi-GB dataset in full. It needs
    `max_samples` set. Full runs leave it off so the whole split is used.
    """
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")

    spec = dict(CORPORA[name])
    col = spec.pop("text_column")
    if split is not None:
        spec["split"] = split

    if streaming:
        if not max_samples:
            raise ValueError("streaming=True needs max_samples")
        ds = load_dataset(**spec, streaming=True)
        texts = []
        for row in ds:
            t = row[col]
            if t and len(t.strip()) >= min_chars:
                texts.append(t)
                if len(texts) >= max_samples:
                    break
        return texts

    ds = load_dataset(**spec)
    texts = [t for t in ds[col] if t and len(t.strip()) >= min_chars]
    if max_samples:
        texts = texts[:max_samples]
    return texts
