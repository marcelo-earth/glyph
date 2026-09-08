"""The whole experiment: build both tokenizers, train the same model twice.

    python run_experiment.py

Steps:
  1. Build the `fit` tokenizer on the training corpus.
  2. Build the `general` tokenizer on a broad, different corpus, same vocab size
     and same settings.
  3. Train GlyphGPT on the training corpus with each tokenizer. Same seed, same
     data, same parameter budget. Only the tokenizer changes.
  4. Write results.json.

Read it as: if `general` and `fit` land at the same bits per character while
`fit` has the lower fertility, the corpus-fit tokenizer only bought shorter
sequences. If `fit` also has the lower bits per character, it did real work.
"""

import argparse
import json
import os

from build_tokenizer import build_bpe
from data import load_corpus
from train import train


def main():
    p = argparse.ArgumentParser(description="Run the full corpus-fit vs general experiment")
    p.add_argument("--train-corpus", default="python",
                   help="corpus the model trains on and the fit tokenizer learns")
    p.add_argument("--general-corpus", default="wikitext",
                   help="broad corpus the baseline tokenizer learns")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--tok-max-samples", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train-samples", type=int, default=40_000)
    p.add_argument("--target-params", type=int, default=40_000_000)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tok-dir", default="tokenizers")
    p.add_argument("--out", default="results.json")
    p.add_argument("--streaming", action="store_true",
                   help="pull data lazily (needs the sample caps set)")
    p.add_argument("--no-equal-tokens", action="store_true",
                   help="do not cap both runs to the same training token count")
    p.add_argument("--smoke", action="store_true",
                   help="tiny everything: checks the pipeline runs end to end, "
                        "numbers are not meaningful")
    args = p.parse_args()

    if args.train_corpus == args.general_corpus:
        raise SystemExit("train-corpus and general-corpus must differ, or there "
                         "is no contrast to measure")

    if args.smoke:
        args.streaming = True
        args.vocab_size = min(args.vocab_size, 4000)
        args.tok_max_samples = min(args.tok_max_samples, 4000)
        args.max_train_samples = min(args.max_train_samples, 3000)
        args.seq_len = min(args.seq_len, 128)
        args.epochs = 1
        args.target_params = min(args.target_params, 4_000_000)

    os.makedirs(args.tok_dir, exist_ok=True)
    fit_path = os.path.join(args.tok_dir, f"fit_{args.train_corpus}_v{args.vocab_size}.json")
    gen_path = os.path.join(args.tok_dir, f"general_{args.general_corpus}_v{args.vocab_size}.json")

    if not os.path.exists(fit_path):
        print(f"building fit tokenizer on {args.train_corpus}")
        build_bpe(load_corpus(args.train_corpus, max_samples=args.tok_max_samples,
                              streaming=args.streaming),
                  args.vocab_size, fit_path)
    if not os.path.exists(gen_path):
        print(f"building general tokenizer on {args.general_corpus}")
        build_bpe(load_corpus(args.general_corpus, max_samples=args.tok_max_samples,
                              streaming=args.streaming),
                  args.vocab_size, gen_path)

    # Equal token budget: encode the training docs with each tokenizer, cap both
    # runs to the smaller count. Otherwise the coarser tokenizer produces more
    # tokens from the same text and quietly trains for more steps.
    max_train_tokens = None
    if not args.no_equal_tokens:
        from tokenizers import Tokenizer
        from train import encode_corpus
        n_docs = args.max_train_samples
        docs = load_corpus(args.train_corpus, split="train", max_samples=n_docs,
                           streaming=args.streaming)
        counts = {p: len(encode_corpus(Tokenizer.from_file(p), docs))
                  for p in (fit_path, gen_path)}
        max_train_tokens = min(counts.values())
        print(f"token counts on train corpus: "
              f"fit={counts[fit_path]:,} general={counts[gen_path]:,}  "
              f"-> capping both at {max_train_tokens:,}")

    common = dict(
        corpus=args.train_corpus,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_train_tokens=max_train_tokens,
        target_params=args.target_params,
        device=args.device,
        seed=args.seed,
        streaming=args.streaming,
    )

    results = []
    for label, tok_path in [("fit", fit_path), ("general", gen_path)]:
        results.append(train(tokenizer_path=tok_path, label=label, **common))
        with open(args.out, "w") as f:  # write after each run so a late crash keeps the first
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nresults in {args.out}\n{'='*60}")
    print(f"{'run':>8} {'vocab':>7} {'fertility':>10} {'val ppl':>9} {'val bpc':>9}")
    for m in results:
        print(f"{m['label']:>8} {m['vocab_size']:>7} {m['val_fertility']:>10.2f} "
              f"{m['val_perplexities'][-1]:>9.1f} {m['val_bits_per_char'][-1]:>9.3f}")


if __name__ == "__main__":
    main()
