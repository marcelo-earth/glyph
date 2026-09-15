"""Exact accounting for a document-separated canonical-token evaluation.

This measures the NLL of the tokenizer's one deterministic encoding, *not* the
marginal probability of a string over all possible token sequences. Every text
token is a target once, including the first (conditioned on BOS). EOS is omitted
by default; when enabled it is an additional target and contributes to NLL.
Denominators always include all original Unicode code points and UTF-8 bytes.

``chunks`` resets attention/positions every ``context_length`` targets. The
first input of later chunks is the preceding true token, not a new BOS. Thus
chunk boundaries depend on tokenization. ``rolling`` scores each target with
the longest available preceding context up to ``context_length`` input tokens;
it removes the sawtooth in context size but needs many more forward passes.
Neither policy matches the raw-text span of context across tokenizers.
"""

from dataclasses import dataclass
from collections import Counter
import hashlib
import math
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvaluationDocument:
    index: int
    sha256: str
    token_ids: tuple[int, ...]
    text_tokens: int
    characters: int
    utf8_bytes: int


@dataclass(frozen=True)
class PreparedEvaluation:
    documents: tuple[EvaluationDocument, ...]
    bos_id: int
    pad_id: int
    include_eos: bool


def prepare_evaluation(tokenizer, texts: Iterable[str], *, include_eos=False):
    """Encode once, check lossless round trips, and preserve document identity.

    Automatic post-processing is disabled; this function alone handles special
    tokens. Literal special-token spellings in source text are encoded as normal
    text, so a quoted "[EOS]" cannot become a document-control token. A nonempty
    corpus with nonzero character count is required. Empty
    documents are retained with zero text loss (or an EOS loss when requested).
    """
    if hasattr(tokenizer, "encode_special_tokens"):
        # Hugging Face's counterintuitive name: True means split literal special
        # token strings using the normal tokenizer instead of emitting controls.
        tokenizer.encode_special_tokens = True
    special_ids = {}
    names = ["[BOS]", "[PAD]"] + (["[EOS]"] if include_eos else [])
    for name in names:
        token_id = tokenizer.token_to_id(name)
        if token_id is None:
            raise ValueError(f"evaluation requires tokenizer token {name}")
        special_ids[name] = token_id
    documents = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError("evaluation documents must be strings")
        encoded = tuple(tokenizer.encode(text, add_special_tokens=False).ids)
        if text and not encoded:
            raise ValueError(f"nonempty document {index} encoded to no tokens")
        if tokenizer.decode(list(encoded), skip_special_tokens=False) != text:
            raise ValueError(f"tokenizer is not lossless on document {index}")
        targets = encoded + ((special_ids["[EOS]"],) if include_eos else ())
        raw_bytes = text.encode("utf-8")
        documents.append(EvaluationDocument(
            index=index, sha256=hashlib.sha256(raw_bytes).hexdigest(),
            token_ids=targets, text_tokens=len(encoded),
            characters=len(text), utf8_bytes=len(raw_bytes),
        ))
    if not documents or sum(d.characters for d in documents) == 0:
        raise ValueError("evaluation requires at least one nonempty document")
    return PreparedEvaluation(tuple(documents), special_ids["[BOS]"],
                              special_ids["[PAD]"], bool(include_eos))


def _windows(prepared, context_length, context_policy):
    """Yield (document index, inputs, labels); -100 marks unscored labels."""
    for document in prepared.documents:
        targets = document.token_ids
        if not targets:
            continue
        inputs = (prepared.bos_id,) + targets[:-1]
        if context_policy == "chunks":
            for start in range(0, len(targets), context_length):
                yield (document.index, inputs[start:start + context_length],
                       targets[start:start + context_length])
        else:
            for index, target in enumerate(targets):
                start = max(0, index + 1 - context_length)
                context = inputs[start:index + 1]
                yield (document.index, context,
                       (-100,) * (len(context) - 1) + (target,))


def _default_device(model):
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


@torch.no_grad()
def evaluate_documents(model, tokenizer=None, texts=None, *, prepared=None,
                       context_length=None, batch_size=16, device=None,
                       include_eos=False, context_policy="chunks"):
    """Return JSON-ready corpus metrics and per-document sufficient statistics.

    Supply either ``tokenizer``/``texts`` or a cached ``prepared`` object. When
    using cached data its EOS policy is authoritative. Right padding is masked
    out of the loss; causal attention ensures padding cannot affect valid
    tokens, so GlyphGPT does not need a padding-attention-mask interface.
    The model's previous training/evaluation mode is restored even on failure.

    Character-normalized losses are ratios of summed NLL to summed raw length,
    never means of batch losses or per-document ratios. Document records allow
    paired document bootstrap; independent training seeds remain necessary.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if context_policy not in {"chunks", "rolling"}:
        raise ValueError("context_policy must be 'chunks' or 'rolling'")
    if context_length is None:
        context_length = getattr(model, "seq_len", None)
    if context_length is None or context_length < 1:
        raise ValueError("context_length must be positive")
    if context_length > getattr(model, "seq_len", context_length):
        raise ValueError("context_length exceeds the model position capacity")
    if prepared is None:
        if tokenizer is None or texts is None:
            raise ValueError("supply tokenizer and texts, or prepared evaluation")
        prepared = prepare_evaluation(tokenizer, texts, include_eos=include_eos)
    elif tokenizer is not None or texts is not None:
        raise ValueError("supply prepared alone, not alongside tokenizer/texts")
    device = device or _default_device(model)
    nlls = [0.0] * len(prepared.documents)
    scored_counts = [0] * len(prepared.documents)
    window_count = 0
    was_training = model.training

    def score_batch(batch):
        width = max(len(inputs) for _, inputs, _ in batch)
        x = torch.full((len(batch), width), prepared.pad_id,
                       dtype=torch.long, device=device)
        y = torch.full_like(x, -100)
        for row, (_, inputs, labels) in enumerate(batch):
            x[row, :len(inputs)] = torch.tensor(inputs, device=device)
            y[row, :len(labels)] = torch.tensor(labels, device=device)
        logits = model(x)
        if logits.ndim != 3 or logits.shape[:2] != x.shape:
            raise ValueError("model must return [batch, sequence, vocabulary] logits")
        # FP32 CE works on MPS; accumulate on CPU in FP64 to avoid long-corpus
        # reduction error without requiring unsupported MPS float64 tensors.
        losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                 y.reshape(-1), reduction="none", ignore_index=-100)
        losses = losses.reshape(y.shape).cpu().double()
        if not torch.isfinite(losses).all():
            raise ValueError("nonfinite evaluation loss")
        counts = (y != -100).sum(dim=1).cpu().tolist()
        for row, (doc_index, _, _) in enumerate(batch):
            nlls[doc_index] += losses[row].sum().item()
            scored_counts[doc_index] += counts[row]

    try:
        model.eval()
        batch = []
        for window in _windows(prepared, context_length, context_policy):
            batch.append(window)
            window_count += 1
            if len(batch) == batch_size:
                score_batch(batch)
                batch = []
        if batch:
            score_batch(batch)
    finally:
        model.train(was_training)

    return _summarize(prepared, nlls, scored_counts,
                      context_policy=context_policy, context_length=context_length,
                      window_count=window_count)


def _summarize(prepared, nlls, scored_counts, **metadata):
    documents = []
    for document, nll, count in zip(prepared.documents, nlls, scored_counts):
        if count != len(document.token_ids):
            raise RuntimeError("evaluation failed to score every target exactly once")
        documents.append({
            "index": document.index, "sha256": document.sha256,
            "nll_nats": nll, "scored_tokens": count,
            "text_tokens": document.text_tokens, "characters": document.characters,
            "utf8_bytes": document.utf8_bytes,
        })
    nll = math.fsum(nlls)
    n_tokens = sum(scored_counts)
    characters = sum(d.characters for d in prepared.documents)
    utf8_bytes = sum(d.utf8_bytes for d in prepared.documents)
    text_tokens = sum(d.text_tokens for d in prepared.documents)
    nats_per_token = nll / n_tokens
    return {
        "metric": "canonical_token_sequence_nll",
        "include_eos": prepared.include_eos, "document_count": len(documents),
        "nll_nats": nll,
        "scored_tokens": n_tokens, "text_tokens": text_tokens,
        "characters": characters, "utf8_bytes": utf8_bytes,
        "nats_per_token": nats_per_token,
        "perplexity": math.exp(nats_per_token) if nats_per_token < 709 else None,
        "bits_per_character": nll / (characters * math.log(2)),
        "bits_per_utf8_byte": nll / (utf8_bytes * math.log(2)),
        "characters_per_token": characters / text_tokens,
        "documents": documents,
        **metadata,
    }


def evaluate_token_baselines(tokenizer, training_texts, texts=None, *, prepared=None,
                             alpha=1.0, include_eos=False):
    """Uniform and add-alpha training-unigram canonical coding baselines.

    Smoothing covers the *entire* model vocabulary including special tokens,
    just like its softmax. BOS is context-only; EOS contributes a training count
    per document exactly when it is a scored evaluation target. These models
    have no context dependence. Use exactly the LM training document selection
    for ``training_texts``; a token-budget-truncated LM run may see a different
    empirical unigram distribution, which callers should report explicitly.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    if prepared is None:
        if texts is None:
            raise ValueError("supply texts or prepared evaluation")
        prepared = prepare_evaluation(tokenizer, texts, include_eos=include_eos)
    elif texts is not None:
        raise ValueError("supply either texts or prepared evaluation")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size <= 0:
        raise ValueError("tokenizer vocabulary must be nonempty")
    training = prepare_evaluation(tokenizer, training_texts,
                                  include_eos=prepared.include_eos)
    counts = Counter(token for doc in training.documents for token in doc.token_ids)
    train_tokens = sum(counts.values())
    denominator = train_tokens + alpha * vocab_size
    log_probabilities = [math.log(counts[token] + alpha) - math.log(denominator)
                         for token in range(vocab_size)]
    for document in (*training.documents, *prepared.documents):
        if any(token < 0 or token >= vocab_size for token in document.token_ids):
            raise ValueError("token ID outside tokenizer vocabulary")
    target_counts = [len(document.token_ids) for document in prepared.documents]
    uniform_nlls = [count * math.log(vocab_size) for count in target_counts]
    unigram_nlls = [math.fsum(-log_probabilities[token] for token in doc.token_ids)
                    for doc in prepared.documents]
    common = {"context_policy": "independent", "vocab_size": vocab_size}
    return {
        "uniform": _summarize(prepared, uniform_nlls, target_counts,
                              baseline="uniform", **common),
        "unigram": _summarize(prepared, unigram_nlls, target_counts,
                              baseline="training_unigram", alpha=alpha,
                              training_scored_tokens=train_tokens, **common),
    }
