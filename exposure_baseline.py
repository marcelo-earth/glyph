"""Fit a unigram on exactly the targets an experiment actually trained on.

The full-training-corpus unigram has access to more text than an early pilot.
This diagnostic replays the recorded sampler without training a model; it tests
whether conclusions about learned savings depend on that baseline data access.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from corpus_snapshot import load_snapshot
from evaluation import prepare_evaluation
from experiment import TrainingData, aggregate_documents, atomic_json, digest, raw_blocks


def replay_counts(data, steps, batch_size, seed, vocab_size):
    rng=np.random.default_rng(seed);order=rng.permutation(data.size);cursor=0
    counts=np.zeros(vocab_size,dtype=np.int64)
    exposure={'raw_chars':0,'raw_bytes':0,'target_tokens':0,'padded_positions':0}
    for _ in range(steps):
        indices=[]
        while len(indices)<batch_size:
            take=min(batch_size-len(indices),data.size-cursor)
            indices.extend(order[cursor:cursor+take].tolist());cursor+=take
            if cursor==data.size:order=rng.permutation(data.size);cursor=0
        _,targets,batch_exposure=data.batch(indices)
        ids=targets.numpy().ravel();ids=ids[ids>=0]
        counts+=np.bincount(ids,minlength=vocab_size)
        for key,value in batch_exposure.items():exposure[key]+=value
        exposure['padded_positions']+=batch_size*data.seq_len
    return counts,exposure


def score_counts(counts,prepared,alpha=1.0):
    if alpha<=0:raise ValueError('Positive smoothing required')
    logprobs=np.log(counts+alpha)-math.log(int(counts.sum())+alpha*len(counts))
    rows=[]
    for doc in prepared.documents:
        nll=float(-logprobs[list(doc.token_ids)].sum())
        rows.append({'index':doc.index,'sha256':doc.sha256,'nll_nats':nll,
                     'scored_tokens':len(doc.token_ids),'text_tokens':doc.text_tokens,
                     'characters':doc.characters,'utf8_bytes':doc.utf8_bytes})
    nll=math.fsum(x['nll_nats'] for x in rows);nbytes=sum(x['utf8_bytes'] for x in rows)
    return {'baseline':'exposure_matched_training_unigram','alpha':alpha,
            'nll_nats':nll,'utf8_bytes':nbytes,'bits_per_utf8_byte':nll/nbytes/math.log(2),
            'training_target_tokens':int(counts.sum()),'documents':rows}


def diagnose(snapshot_dir,tokenizer_path,metrics_path):
    metrics=json.loads(Path(metrics_path).read_text());cfg=metrics['config']
    if metrics['status']!='complete':raise ValueError('Requires completed run')
    if digest(Path(snapshot_dir)/'manifest.json')!=cfg['snapshot_sha256']:raise ValueError('Wrong snapshot')
    if digest(tokenizer_path)!=cfg['tokenizer_sha256']:raise ValueError('Wrong tokenizer')
    snapshot=load_snapshot(snapshot_dir);tok=Tokenizer.from_file(str(tokenizer_path));tok.encode_special_tokens=True
    data=TrainingData(tok,snapshot['train'],cfg['seq_len'],cfg['regime'])
    counts,exposure=replay_counts(data,cfg['steps'],cfg['batch_size'],cfg['seed'],cfg['vocab_size'])
    if exposure!=metrics['exposure']:raise ValueError('Sampler replay does not reproduce trained exposure')
    texts=snapshot['validation'][:cfg['val_docs']];blocks,parents=raw_blocks(texts,cfg['seq_len'])
    prepared=prepare_evaluation(tok,blocks);scores=score_counts(counts,prepared)
    scores['source_documents']=aggregate_documents(scores.pop('documents'),parents,texts)
    scores['training_exposure']=exposure;scores['token_counts']=counts.tolist()
    scores['metrics_sha256']=digest(metrics_path);scores['config_hash']=metrics['config_hash']
    scores['training_boundary_policy']='EOS included' if cfg['regime']=='token' else 'raw block content only'
    scores['model_gain_bits_per_utf8_byte']=scores['bits_per_utf8_byte']-metrics['history'][-1]['validation']['bits_per_utf8_byte']
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['snapshot','tokenizer','metrics','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();result=diagnose(a.snapshot,a.tokenizer,a.metrics);atomic_json(a.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_documents','token_counts']},indent=2))

if __name__=='__main__':main()
