"""Train a BPE tokenizer at a fixed vocab size.

Both tokenizers in the experiment are built here with identical settings. The
only thing that differs between them is the corpus passed in: the training
corpus (the "fit" tokenizer) or a broad general corpus (the "general"
baseline). Same algorithm, same vocab budget, same pre-tokenizer, so any
difference downstream is about corpus fit and nothing else.
"""

import argparse
import os

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from data import load_corpus

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def build_bpe(texts, vocab_size, save_path, min_frequency=2):
    """Train a byte-level BPE on `texts` and save it to `save_path`.

    ByteLevel pre-tokenization means every byte is representable, so there is no
    real [UNK] path and the initial alphabet is a fixed 256 regardless of corpus.
    That keeps the "fit" and "general" tokenizers starting from the same place;
    a char-level BPE would hand each corpus a different starting alphabet and
    quietly spend a different share of the budget before the first merge.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)

    actual = tokenizer.get_vocab_size()
    if actual < vocab_size:
        # legitimate on a small or repetitive corpus: not enough distinct
        # merges exist. The caller needs to know because it makes the two
        # tokenizers different sizes, which the model config then has to absorb.
        print(f"  note: corpus only supports {actual} of {vocab_size} tokens")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"saved {save_path}  (vocab {actual})")
    return tokenizer


def main():
    p = argparse.ArgumentParser(description="Train a BPE tokenizer at a fixed vocab size")
    p.add_argument("--corpus", required=True, help="corpus name from data.CORPORA")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--out", required=True, help="path to write the tokenizer json")
    args = p.parse_args()

    texts = load_corpus(args.corpus, max_samples=args.max_samples)
    print(f"{args.corpus}: {len(texts)} docs")
    build_bpe(texts, args.vocab_size, args.out)


if __name__ == "__main__":
    main()
