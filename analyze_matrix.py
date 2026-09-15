"""Summarize completed paired development runs with source-group uncertainty."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from experiment import atomic_json,digest
from run_matrix import check_pair


def group_bootstrap(fit,general,groups,replicates=4000,seed=31415):
    if len(fit)!=len(general) or len(fit)!=len(groups):raise ValueError('Unpaired source records')
    totals=defaultdict(lambda:np.zeros(2))
    for a,b,group in zip(fit,general,groups):
        if a['sha256']!=b['sha256'] or a['utf8_bytes']!=b['utf8_bytes']:raise ValueError('Source mismatch')
        totals[group]+=np.array([a['nll_nats']-b['nll_nats'],a['utf8_bytes']])
    matrix=np.array(list(totals.values()));rng=np.random.default_rng(seed)
    values=[]
    for _ in range(replicates):
        summed=matrix[rng.integers(0,len(matrix),len(matrix))].sum(axis=0)
        values.append(summed[0]/summed[1]/math.log(2))
    return {'group_count':len(matrix),'replicates':replicates,'rng_seed':seed,
            'fit_minus_general_bpb_95_interval':np.percentile(values,[2.5,97.5]).tolist(),
            'scope':'source-group sampling only; training-seed variation is separate'}


def analyze(root,snapshot_dir):
    root=Path(root);snapshot_dir=Path(snapshot_dir)
    manifest=json.loads((snapshot_dir/'manifest.json').read_text())
    diagnostics_path=root/'tokenizer_diagnostics.json'
    diagnostics=json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else None
    runs={}
    paths={p.parent:p for p in root.glob('*/evidence.json')}
    paths.update({p.parent:p for p in root.glob('*/metrics.json')})
    for path in paths.values():
        result=json.loads(path.read_text())
        if result.get('status')=='complete':
            cfg=result['config'];runs[cfg['regime'],cfg['seed'],cfg['label']]=(result,path)
    pairs=[]
    for regime,seed,label in sorted(runs):
        if label!='fit' or (regime,seed,'general') not in runs:continue
        (fit,fit_path),(general,general_path)=runs[regime,seed,'fit'],runs[regime,seed,'general']
        audit=check_pair(fit,general)
        if fit['config']['snapshot_sha256']!=digest(snapshot_dir/'manifest.json'):raise ValueError('Snapshot identity differs')
        f=fit['history'][-1]['validation'];g=general['history'][-1]['validation'];n=len(f['source_documents'])
        if 'repository_metadata' in manifest:
            meta=(snapshot_dir/manifest['repository_metadata']['validation']['file']).read_bytes()
            if hashlib.sha256(meta).hexdigest()!=manifest['repository_metadata']['validation']['file_sha256']:
                raise ValueError('Metadata hash differs')
            records=[json.loads(line) for line in meta.decode().splitlines()][:n]
            if [r['sha256'] for r in records]!=[r['sha256'] for r in f['source_documents']]:raise ValueError('Metadata source order differs')
            groups=[r['repository_family'] for r in records];unit='repository_family'
        else:groups=[r['sha256'] for r in f['source_documents']];unit='document'
        row={'regime':regime,'seed':seed,'steps':fit['config']['steps'],'pair_audit':audit,
             'fit_bpb':f['bits_per_utf8_byte'],'general_bpb':g['bits_per_utf8_byte'],
             'fit_minus_general_bpb':f['bits_per_utf8_byte']-g['bits_per_utf8_byte'],
             'bootstrap_unit':unit,'bootstrap':group_bootstrap(f['source_documents'],g['source_documents'],groups),
             'training_exposure':{a:r['exposure'] for a,r in [('fit',fit),('general',general)]},
             'training_seconds':{a:r['training_seconds'] for a,r in [('fit',fit),('general',general)]},
             'metrics_sha256':{'fit':fit.get('evidence_export',{}).get('original_metrics_sha256',digest(fit_path)),
                               'general':general.get('evidence_export',{}).get('original_metrics_sha256',digest(general_path))}}
        if diagnostics:
            baselines={a:diagnostics['tokenizers'][a]['segmentations']['shared_raw_blocks']['baselines']['unigram']
                       for a in ['fit','general']}
            for arm,model in [('fit',f),('general',g)]:
                if [x['sha256'] for x in baselines[arm]['source_documents']]!=[x['sha256'] for x in model['source_documents']]:
                    raise ValueError('Baseline source evaluation differs')
            row['full_training_unigram_bpb']={a:b['bits_per_utf8_byte'] for a,b in baselines.items()}
            row['model_gain_over_full_training_unigram_bpb']={a:baselines[a]['bits_per_utf8_byte']-model['bits_per_utf8_byte']
                                                            for a,model in [('fit',f),('general',g)]}
        exposure_paths={a:path.parent/'exposure_unigram.json' for a,path in [('fit',fit_path),('general',general_path)]}
        if all(p.exists() for p in exposure_paths.values()):
            baselines={a:json.loads(p.read_text()) for a,p in exposure_paths.items()}
            for a,path in [('fit',fit_path),('general',general_path)]:
                if baselines[a]['metrics_sha256']!=row['metrics_sha256'][a]:raise ValueError('Stale exposure baseline')
            row['exposure_matched_unigram_bpb']={a:b['bits_per_utf8_byte'] for a,b in baselines.items()}
            row['model_gain_over_exposure_unigram_bpb']={a:b['model_gain_bits_per_utf8_byte'] for a,b in baselines.items()}
        pairs.append(row)
    return {'scope':'development only; incomplete pairs omitted, no final test inference',
            'snapshot_sha256':digest(snapshot_dir/'manifest.json'),'pairs':pairs,
            'training_seed_count_by_regime':{r:len({p['seed'] for p in pairs if p['regime']==r}) for r in ['raw','token']}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=analyze(a.root,a.snapshot);atomic_json(a.out,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
