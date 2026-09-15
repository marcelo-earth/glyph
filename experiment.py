"""Auditable paired Glyph experiments over an immutable corpus snapshot.

Two estimands: packed equal-token training (practical fixed-token budget), and
shared raw-block training (same raw exposure, context boundaries, padded dense
shapes, updates, and byte-normalized loss). See research/PROTOCOL.md.
"""
import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from build_tokenizer import build_bpe
from model import GlyphGPT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def raw_blocks(texts, max_bytes):
    """Preserve all characters; each shared block fits a byte-BPE token window."""
    if max_bytes < 4:
        raise ValueError('max_bytes must be at least 4 for arbitrary UTF-8')
    blocks, parents = [], []
    for doc_id, text in enumerate(texts):
        start, used = 0, 0
        for i, ch in enumerate(text):
            width = len(ch.encode('utf-8'))
            if used + width > max_bytes:
                blocks.append(text[start:i]); parents.append(doc_id)
                start, used = i, 0
            used += width
        if start < len(text):
            blocks.append(text[start:]); parents.append(doc_id)
    return blocks, parents


def aggregate_documents(records, parents, texts):
    """Retain independent source units instead of bootstrapping correlated blocks."""
    grouped = [{'index': i, 'sha256': hashlib.sha256(t.encode('utf-8')).hexdigest(),
                'nll_nats': 0.0, 'scored_tokens': 0, 'text_tokens': 0,
                'characters': 0, 'utf8_bytes': 0, 'blocks': 0} for i,t in enumerate(texts)]
    if len(records) != len(parents):
        raise ValueError('Block parent mapping length mismatch')
    for record,parent in zip(records,parents):
        for key in ['nll_nats','scored_tokens','text_tokens','characters','utf8_bytes']:
            grouped[parent][key] += record[key]
        grouped[parent]['blocks'] += 1
    for row,text in zip(grouped,texts):
        if row['characters'] != len(text) or row['utf8_bytes'] != len(text.encode('utf-8')):
            raise ValueError('Source-document accounting mismatch')
    return grouped


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested=None):
    return requested or ('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')


def synchronize(device):
    if str(device).startswith('mps'):
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()


def build_tokenizers(snapshot_dir, vocab_size, out):
    from corpus_snapshot import load_snapshot
    snapshot = load_snapshot(snapshot_dir)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite tokenizer directory: {out}')
    out.mkdir(parents=True)
    manifest = {'snapshot_sha256': digest(Path(snapshot_dir) / 'manifest.json'),
                'vocab_size': vocab_size, 'settings': {'min_frequency': 2,
                'byte_level': True, 'add_prefix_space': False}}
    paths = {}
    for label, key in [('fit', 'fit_tokenizer'), ('general', 'general_tokenizer')]:
        texts = snapshot[key]
        path = out / f'{label}.json'
        tokenizer = build_bpe(texts, vocab_size, str(path))
        tokenizer.encode_special_tokens = True
        if tokenizer.get_vocab_size() != vocab_size:
            raise ValueError(f'{label} cannot support vocab {vocab_size}; choose a shared smaller vocabulary')
        # Special-token strings in raw data must not become atomic control tokens.
        for text in texts:
            encoded = tokenizer.encode(text)
            if tokenizer.decode(encoded.ids, skip_special_tokens=False) != text:
                raise ValueError(f'{label}: tokenizer failed exact round trip')
        paths[label] = str(path)
        manifest[label] = {'sha256': digest(path), 'training_chars': sum(map(len, texts)),
                           'training_bytes': sum(len(t.encode('utf-8')) for t in texts)}
    if manifest['fit']['training_chars'] != manifest['general']['training_chars']:
        raise ValueError('Tokenizer training character budgets differ')
    atomic_json(out / 'manifest.json', manifest)
    return paths


class TrainingData:
    def __init__(self, tokenizer, texts, seq_len, regime):
        self.regime, self.seq_len = regime, seq_len
        tokenizer.encode_special_tokens = True
        self.pad = tokenizer.token_to_id('[PAD]')
        self.bos = tokenizer.token_to_id('[BOS]')
        self.eos = tokenizer.token_to_id('[EOS]')
        if None in (self.pad, self.bos, self.eos):
            raise ValueError('Tokenizer lacks required special tokens')
        self.blocks = None
        if regime == 'raw':
            self.blocks, _ = raw_blocks(texts, seq_len)
            self.ids = [tokenizer.encode(t).ids for t in self.blocks]
            if not self.ids or any(not t or len(t) > seq_len for t in self.ids):
                raise ValueError('Invalid raw block encoding')
            self.size = len(self.ids)
        else:
            ids, chars, nbytes = [self.bos], [0], [0]
            for text in texts:
                enc = tokenizer.encode(text)
                if tokenizer.decode(enc.ids, skip_special_tokens=False) != text:
                    raise ValueError('Training text fails tokenizer roundtrip')
                # ByteLevel BPE symbols each stand for one source byte. Count
                # bytes exactly, even when a Unicode code point spans tokens.
                boundaries = np.cumsum([len(ch.encode('utf-8')) for ch in text]).tolist()
                consumed = completed = 0
                for token in enc.ids:
                    width = len(tokenizer.id_to_token(token))
                    consumed += width
                    now_completed = bisect.bisect_right(boundaries, consumed)
                    ids.append(token); chars.append(now_completed - completed)
                    nbytes.append(width); completed = now_completed
                if consumed != len(text.encode('utf-8')):
                    raise ValueError('ByteLevel token byte accounting mismatch')
                ids.append(self.eos); chars.append(0); nbytes.append(0)
            self.stream = torch.tensor(ids, dtype=torch.long)
            self.char_prefix = np.cumsum([0] + chars)
            self.byte_prefix = np.cumsum([0] + nbytes)
            self.size = (len(ids) - 1) // seq_len
        if self.size == 0:
            raise ValueError('No training examples')

    def batch(self, indices):
        x = torch.full((len(indices), self.seq_len), self.pad, dtype=torch.long)
        y = torch.full_like(x, -100)
        raw_chars = raw_bytes = 0
        for row, idx in enumerate(indices):
            if self.regime == 'raw':
                tokens = self.ids[idx]; n = len(tokens)
                x[row, :n] = torch.tensor([self.bos] + tokens[:-1])
                y[row, :n] = torch.tensor(tokens)
                raw_chars += len(self.blocks[idx]); raw_bytes += len(self.blocks[idx].encode('utf-8'))
            else:
                s = idx * self.seq_len
                x[row] = self.stream[s:s+self.seq_len]
                y[row] = self.stream[s+1:s+self.seq_len+1]
                raw_chars += int(self.char_prefix[s+self.seq_len+1]-self.char_prefix[s+1])
                raw_bytes += int(self.byte_prefix[s+self.seq_len+1]-self.byte_prefix[s+1])
        return x, y, {'raw_chars': raw_chars, 'raw_bytes': raw_bytes,
                      'target_tokens': int((y != -100).sum())}


def verify_tokenizer_manifest(snapshot_dir, tokenizer_path, label):
    path = Path(tokenizer_path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if manifest['snapshot_sha256'] != digest(Path(snapshot_dir)/'manifest.json'):
        raise ValueError('Tokenizer was fitted on another snapshot')
    if label not in ('fit','general') or manifest[label]['sha256'] != digest(path):
        raise ValueError('Tokenizer label or content differs from manifest')
    if manifest['fit']['training_chars'] != manifest['general']['training_chars']:
        raise ValueError('Tokenizer corpus budgets are unequal')
    for arm in ('fit','general'):
        pair_path = path.parent/f'{arm}.json'
        if digest(pair_path) != manifest[arm]['sha256']:
            raise ValueError('Tokenizer pair content differs from manifest')
        if Tokenizer.from_file(str(pair_path)).get_vocab_size() != manifest['vocab_size']:
            raise ValueError('Tokenizer pair vocabulary sizes are unequal')
    return manifest


def train_run(snapshot_dir, tokenizer_path, out_dir, *, label, regime, seed=42,
              dim=192, layers=4, heads=4, seq_len=128, batch_size=16,
              steps=1000, lr=3e-4, eval_every=250, val_docs=100,
              device=None, resume=False):
    from corpus_snapshot import load_snapshot
    from evaluation import evaluate_documents, prepare_evaluation
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / 'metrics.json'
    device = get_device(device)
    snapshot = load_snapshot(snapshot_dir)
    verify_tokenizer_manifest(snapshot_dir, tokenizer_path, label)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.encode_special_tokens = True
    config = dict(label=label, regime=regime, seed=seed, dim=dim, layers=layers,
                  heads=heads, seq_len=seq_len, batch_size=batch_size, steps=steps,
                  lr=lr, eval_every=eval_every, val_docs=val_docs,
                  vocab_size=tokenizer.get_vocab_size(), dropout=0.1,
                  snapshot_sha256=digest(Path(snapshot_dir)/'manifest.json'),
                  tokenizer_sha256=digest(tokenizer_path),
                  loss_normalization='raw_bytes' if regime == 'raw' else 'tokens',
                  device=device, torch_version=str(torch.__version__),
                  implementation_sha256={p: digest(Path(__file__).parent/p) for p in
                      ['experiment.py','model.py','evaluation.py','corpus_snapshot.py']})
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if steps < 1 or batch_size < 1 or eval_every < 1 or val_docs < 1:
        raise ValueError('Steps, batch size, evaluation interval and validation count must be positive')
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if existing['config_hash'] != config_hash:
            raise ValueError('Output directory belongs to another configuration')
        if existing.get('status') == 'complete':
            print(f'already complete: {out}', flush=True); return existing
        if not resume:
            raise ValueError('Partial run exists; pass --resume')
    seed_all(seed)
    model = GlyphGPT(config['vocab_size'], dim, heads, layers, seq_len).to(device)
    initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                         for p in model.parameters())).hexdigest()
    data = TrainingData(tokenizer, snapshot['train'], seq_len, regime)
    order_rng = np.random.default_rng(seed)
    order, cursor = order_rng.permutation(data.size), 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # Shared raw boundaries eliminate compression-dependent context windows.
    validation = snapshot['validation'][:val_docs]
    val_blocks, parents = raw_blocks(validation, seq_len)
    prepared = prepare_evaluation(tokenizer, val_blocks)
    result = {'config': config, 'config_hash': config_hash,
              'status': 'running', 'initial_weights_sha256': initial_hash,
              'parameters': model.count_params(), 'device': device,
              'torch_version': torch.__version__, 'python_version': platform.python_version(),
              'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
              'train_examples': data.size, 'history': [],
              'exposure': {'raw_chars': 0, 'raw_bytes': 0, 'target_tokens': 0,
                           'padded_positions': 0}, 'training_seconds': 0.0}
    start_step = 0
    ckpt_path = out / 'checkpoint.pt'
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['config_hash'] != config_hash:
            raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k != 'step': state[k]=v.to(device)
        sched.load_state_dict(ckpt['scheduler'])
        result=ckpt['result']; start_step=ckpt['step']
        order=ckpt['order']; cursor=ckpt['cursor']; order_rng.bit_generator.state=ckpt['order_rng']
        torch.set_rng_state(ckpt['torch_rng'])
        if device == 'mps' and ckpt.get('mps_rng') is not None: torch.mps.set_rng_state(ckpt['mps_rng'])
        if str(device).startswith('cuda') and ckpt.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    def evaluate(step):
        scores = evaluate_documents(model, prepared=prepared,
                                    batch_size=batch_size, device=device)
        scores['raw_block_bytes'] = seq_len
        scores['source_documents'] = aggregate_documents(scores['documents'], parents, validation)
        row = {'step': step, 'exposure': dict(result['exposure']),
               'training_seconds': result['training_seconds'], 'validation': scores}
        result['history'].append(row)
        atomic_json(metrics_path, result)
        print(json.dumps({'out':str(out),'step':step,'validation':{k:v for k,v in scores.items() if k not in ('documents','source_documents')}}),flush=True)
    if start_step == 0:
        evaluate(0)
    running_loss = 0.0
    for step in range(start_step + 1, steps + 1):
        model.train()
        indices=[]
        while len(indices)<batch_size:
            take=min(batch_size-len(indices),data.size-cursor)
            indices.extend(order[cursor:cursor+take].tolist());cursor+=take
            if cursor==data.size: order=order_rng.permutation(data.size);cursor=0
        x,y,exposure=data.batch(indices);x=x.to(device);y=y.to(device)
        synchronize(device); tick=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        nll=F.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum')
        denom=exposure['raw_bytes'] if regime=='raw' else exposure['target_tokens']
        loss=nll/denom
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step();synchronize(device)
        result['training_seconds']+=time.perf_counter()-tick
        for key,value in exposure.items(): result['exposure'][key]+=value
        result['exposure']['padded_positions']+=batch_size*seq_len
        running_loss+=float(loss.detach().cpu())
        if step % eval_every == 0 or step == steps:
            evaluate(step)
            result['history'][-1]['train_loss_since_last_eval']=running_loss/(step % eval_every or eval_every)
            running_loss=0.0
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                   'config_hash':config_hash,'config':config,'step':step,'result':result,
                   'order':order,'cursor':cursor,'order_rng':order_rng.bit_generator.state,
                   'torch_rng':torch.get_rng_state(),
                   'mps_rng':torch.mps.get_rng_state() if device=='mps' else None,
                   'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None}
            tmp=out/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(ckpt_path)
            atomic_json(metrics_path,result)
    result['status']='complete';atomic_json(metrics_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('tokenizers');build.add_argument('--snapshot',required=True)
    build.add_argument('--out',required=True);build.add_argument('--vocab-size',type=int,default=4096)
    run=sub.add_parser('train');run.add_argument('--snapshot',required=True)
    run.add_argument('--tokenizer',required=True);run.add_argument('--out',required=True)
    run.add_argument('--label',required=True);run.add_argument('--regime',choices=['raw','token'],required=True)
    for name,default in [('seed',42),('dim',192),('layers',4),('heads',4),('seq-len',128),
                         ('batch-size',16),('steps',1000),('eval-every',250),('val-docs',100)]:
        run.add_argument('--'+name,type=int,default=default)
    run.add_argument('--lr',type=float,default=3e-4);run.add_argument('--device')
    run.add_argument('--resume',action='store_true')
    args=vars(parser.parse_args());cmd=args.pop('command')
    if cmd=='tokenizers':build_tokenizers(args['snapshot'],args['vocab_size'],args['out'])
    else:
        args['snapshot_dir']=args.pop('snapshot');args['tokenizer_path']=args.pop('tokenizer');args['out_dir']=args.pop('out')
        train_run(**args)


if __name__=='__main__':main()
