"""Train one GlyphGPT on a corpus with a given tokenizer.

Run this directly to train a single model, or let `run_experiment.py` call
`train()` twice with the two tokenizers.

The headline metric is bits per character on the validation set. Per-token loss
is denominated in a different unit for each tokenizer (a coarser tokenizer
predicts fewer, larger tokens), so it cannot be compared between the two runs.
Dividing the same loss by the raw character count of the validation text puts
both runs on one scale. Perplexity is recorded but not ranked on.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from data import load_corpus
from model import GlyphGPT, generate, get_model_config

SAMPLE_PROMPTS = [
    "def ",
    "import numpy as np\n",
    "The purpose of this",
]


def set_seed(seed):
    """Seed every RNG so the two runs differ only in their tokenizer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenWindows(Dataset):
    """Non-overlapping (x, y) windows over a flat token stream."""

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.tokens[s : s + self.seq_len]
        y = self.tokens[s + 1 : s + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def encode_corpus(tokenizer, texts):
    """Flatten a list of documents into one token list."""
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t).ids)
    return ids


def train(
    tokenizer_path,
    corpus="python",
    seq_len=512,
    batch_size=16,
    epochs=3,
    lr=3e-4,
    max_train_samples=40_000,
    max_val_samples=2_000,
    target_params=40_000_000,
    device=None,
    save_dir="checkpoints",
    seed=42,
    label=None,
):
    """Train a model and return a metrics dict."""
    set_seed(seed)
    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    label = label or os.path.splitext(os.path.basename(tokenizer_path))[0]

    print(f"\n{'='*60}\n{label}: tokenizer={tokenizer_path} corpus={corpus}")
    print(f"device={device} seed={seed}\n{'='*60}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    train_texts = load_corpus(corpus, split="train", max_samples=max_train_samples)
    try:
        val_texts = load_corpus(corpus, split="validation", max_samples=max_val_samples)
    except (ValueError, KeyError):
        # some corpora ship only a train split; carve a held-out tail off it
        cut = max(1, len(train_texts) - max_val_samples)
        train_texts, val_texts = train_texts[:cut], train_texts[cut:]
    print(f"train docs={len(train_texts)}  val docs={len(val_texts)}")

    train_tokens = encode_corpus(tokenizer, train_texts)
    val_tokens = encode_corpus(tokenizer, val_texts)

    # fertility: characters of raw text per token. Lower means the tokenizer
    # packs more text into each token on this corpus. This is where a
    # corpus-fit tokenizer is expected to win; the question is whether the
    # model also gets better, or only the sequences get shorter.
    train_chars = sum(len(t) for t in train_texts)
    val_chars = sum(len(t) for t in val_texts)
    train_fertility = train_chars / len(train_tokens)
    val_fertility = val_chars / len(val_tokens)
    print(f"train tokens={len(train_tokens):,}  fertility={train_fertility:.2f} chars/token")

    train_loader = DataLoader(TokenWindows(train_tokens, seq_len),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TokenWindows(val_tokens, seq_len),
                            batch_size=batch_size, shuffle=False, drop_last=True)

    cfg = get_model_config(vocab_size, target_params=target_params, seq_len=seq_len)
    model = GlyphGPT(vocab_size, cfg["dim"], cfg["n_heads"], cfg["n_layers"], seq_len).to(device)
    total_params = model.count_params()
    emb_pct = 100 * model.count_embedding_params() / total_params
    print(f"dim={cfg['dim']} heads={cfg['n_heads']} layers={cfg['n_layers']}  "
          f"params={total_params:,}  embedding={emb_pct:.1f}%")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))

    metrics = {
        "label": label,
        "tokenizer_path": tokenizer_path,
        "corpus": corpus,
        "seed": seed,
        "vocab_size": vocab_size,
        "dim": cfg["dim"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "total_params": total_params,
        "embedding_pct": emb_pct,
        "train_fertility": train_fertility,
        "val_fertility": val_fertility,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_losses": [],
        "val_losses": [],
        "val_perplexities": [],
        "val_bits_per_char": [],
        "epoch_seconds": [],
    }

    for epoch in range(epochs):
        model.train()
        running, nb, start = 0.0, 0, time.time()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            nb += 1
        train_loss = running / max(nb, 1)
        elapsed = time.time() - start

        model.eval()
        vloss, vb = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vloss += F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item()
                vb += 1
        val_loss = vloss / max(vb, 1)
        val_ppl = math.exp(min(val_loss, 20))
        # nats/token -> nats/char (divide by fertility) -> bits/char
        val_bpc = val_loss / (val_fertility * math.log(2))

        metrics["train_losses"].append(train_loss)
        metrics["val_losses"].append(val_loss)
        metrics["val_perplexities"].append(val_ppl)
        metrics["val_bits_per_char"].append(val_bpc)
        metrics["epoch_seconds"].append(elapsed)
        print(f"  epoch {epoch+1}: train={train_loss:.3f} val={val_loss:.3f} "
              f"ppl={val_ppl:.1f} bpc={val_bpc:.3f} ({elapsed:.0f}s)")

    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f"model_{label}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}")

    metrics["samples"] = []
    print("samples:")
    for prompt in SAMPLE_PROMPTS:
        text = generate(model, tokenizer, prompt, device=device)
        metrics["samples"].append({"prompt": prompt, "text": text})
        print(f"  [{prompt!r}] {text!r}")

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one GlyphGPT with a given tokenizer")
    p.add_argument("--tokenizer", required=True, help="path to a tokenizer json")
    p.add_argument("--corpus", default="python")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train-samples", type=int, default=40_000)
    p.add_argument("--target-params", type=int, default=40_000_000)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(
        tokenizer_path=args.tokenizer,
        corpus=args.corpus,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        target_params=args.target_params,
        device=args.device,
        seed=args.seed,
    )
