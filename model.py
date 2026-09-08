"""A small decoder-only transformer, shared verbatim by both runs.

Neither run is allowed to change anything in here. Given the same vocab size
they get byte-for-byte the same architecture; the only inputs that vary are the
token ids the tokenizer produces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphGPT(nn.Module):
    """Small GPT-style language model with tied input/output embeddings."""

    def __init__(self, vocab_size, dim, n_heads, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # norm_first makes the nested-tensor fast path inapplicable; disable it
        # explicitly so PyTorch does not warn on every construction.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying: at this scale the embedding table is a large share of the
        # model, so sharing it with the output projection is most of what keeps
        # the transformer itself from being starved of parameters.
        self.head.weight = self.tok_emb.weight

        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        _, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_embedding_params(self):
        return self.tok_emb.weight.numel()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8,
             top_k=40, device=None):
    """Sample a continuation of `prompt`. Used only to eyeball the two runs."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt tokenized to zero tokens")

    for _ in range(max_new_tokens):
        context = ids[-model.seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(temperature, 1e-6)

        if top_k:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(probs, num_samples=1)))

    return tokenizer.decode(ids)


def estimate_params(vocab_size, dim, n_layers, seq_len):
    """Exact GlyphGPT parameter count without building the model.

    Weight tying means the embedding table is counted once. Per encoder layer:
    attention in_proj (3*dim^2 + 3*dim) and out_proj (dim^2 + dim), two FFN
    projections (8*dim^2 + 5*dim), two LayerNorms (4*dim) => 12*dim^2 + 13*dim.
    """
    emb = vocab_size * dim + seq_len * dim
    layers = n_layers * (12 * dim * dim + 13 * dim)
    return emb + layers + 2 * dim  # + final LayerNorm


def get_model_config(vocab_size, target_params=40_000_000, seq_len=512,
                     n_heads=8, n_layers=8):
    """Pick `dim` so total params land near `target_params` for this vocab size.

    In the primary experiment both tokenizers are the same size, so both runs
    get the same config and this just sizes the model once. It still solves for
    `dim` numerically so that if the two tokenizers end up slightly different
    sizes (a small corpus can fall short of the vocab budget) the param counts
    are matched instead of drifting apart and confounding the comparison.
    """
    best = None
    for dim in range(n_heads, 2048 + 1, n_heads):  # dim must be divisible by n_heads
        diff = abs(estimate_params(vocab_size, dim, n_layers, seq_len) - target_params)
        if best is None or diff < best[1]:
            best = (dim, diff)
    return {"dim": best[0], "n_heads": n_heads, "n_layers": n_layers}
