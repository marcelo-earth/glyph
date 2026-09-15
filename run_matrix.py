"""Run a declared paired experiment matrix sequentially on one accelerator."""
import argparse
from pathlib import Path
import json

from experiment import atomic_json,train_run,digest


def check_pair(fit,general):
    if fit.get('status') != 'complete' or general.get('status') != 'complete':
        raise ValueError('Pair is not complete')
    if len(fit['history']) != len(general['history']):
        raise ValueError('Paired evaluation histories differ in length')
    differences={k:(fit['config'][k],general['config'][k]) for k in fit['config']
                 if k not in {'label','tokenizer_sha256'} and fit['config'][k]!=general['config'][k]}
    if differences:raise ValueError(f'Paired configurations differ: {differences}')
    for key in ['initial_weights_sha256','parameters']:
        if fit[key]!=general[key]:raise ValueError(f'Pair differs in {key}')
    required=['padded_positions']
    required += ['raw_bytes','raw_chars'] if fit['config']['regime']=='raw' else ['target_tokens']
    for key in required:
        if fit['exposure'][key]!=general['exposure'][key]:raise ValueError(f'Pair exposure differs: {key}')
    for a,b in zip(fit['history'],general['history']):
        for key in required:
            if a['exposure'][key] != b['exposure'][key]:raise ValueError(f'Intermediate exposure differs: {key}')
        if a['step']!=b['step']:raise ValueError('Evaluation steps differ')
        for key in ['utf8_bytes','characters']:
            if a['validation'][key]!=b['validation'][key]:raise ValueError('Evaluation denominator differs')
        if [d['sha256'] for d in a['validation']['source_documents']] != [d['sha256'] for d in b['validation']['source_documents']]:
            raise ValueError('Evaluation source documents differ')
    return {'passed':True,'matched_exposure_fields':required,
            'shared_initial_weights_sha256':fit['initial_weights_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['snapshot','tokenizers','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--regimes',nargs='+',choices=['raw','token'],default=['raw','token'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    for key,default in [('steps',500),('eval-every',100),('val-docs',100),('dim',192),('layers',4),
                        ('heads',4),('seq-len',128),('batch-size',16)]:p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--device');p.add_argument('--resume',action='store_true')
    a=vars(p.parse_args());root=Path(a.pop('out'));snapshot=a.pop('snapshot');tokenizers=Path(a.pop('tokenizers'))
    regimes=a.pop('regimes');seeds=a.pop('seeds');resume=a.pop('resume')
    plan={'snapshot':snapshot,'snapshot_sha256':digest(Path(snapshot)/'manifest.json'),
          'tokenizers':str(tokenizers),'tokenizer_manifest_sha256':digest(tokenizers/'manifest.json'),
          'regimes':regimes,'seeds':seeds,'training':a}
    path=root/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Existing matrix plan differs')
    atomic_json(path,plan)
    for regime in regimes:
        for seed in seeds:
            pair=[]
            for label in ['fit','general']:
                run=root/f'{regime}-{label}-s{seed}'
                print(f'START {run}',flush=True)
                result=train_run(snapshot,tokenizers/f'{label}.json',run,
                                 label=label,regime=regime,seed=seed,resume=resume,**a)
                pair.append(result)
            audit=check_pair(*pair)
            atomic_json(root/f'{regime}-s{seed}-pair-audit.json',audit)
            print(f'PAIR VERIFIED {regime} seed={seed}',flush=True)

if __name__=='__main__':main()
