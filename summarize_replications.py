"""Separate paired training-seed variation from held-out source uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_matrix import group_bootstrap
from experiment import atomic_json,digest
from run_matrix import check_pair


def seed_summary(pairs, groups):
    """Describe mean model NLLs, not the NLL of an ensemble distribution."""
    if not pairs:raise ValueError('At least one completed pair required')
    values=np.array([[f['bits_per_utf8_byte'],g['bits_per_utf8_byte']] for _,f,g in pairs])
    deltas=values[:,0]-values[:,1]
    def average_sources(arm):
        sources=[pair[arm]['source_documents'] for pair in pairs]
        expected=[d['sha256'] for d in sources[0]]
        for rows in sources:
            if [d['sha256'] for d in rows]!=expected:raise ValueError('Seed evaluation sources differ')
        averaged=[]
        for i,base in enumerate(sources[0]):
            if any(rows[i]['utf8_bytes']!=base['utf8_bytes'] for rows in sources):raise ValueError('Seed denominators differ')
            averaged.append(dict(base,nll_nats=float(np.mean([rows[i]['nll_nats'] for rows in sources]))))
        return averaged
    mean=values.mean(axis=0)
    return {'seed_count':len(pairs),'seeds':[s for s,_,_ in pairs],
            'fit_mean_bpb':float(mean[0]),'general_mean_bpb':float(mean[1]),
            'paired_mean_fit_minus_general_bpb':float(deltas.mean()),
            'paired_differences_bpb':{str(s):float(d) for (s,_,_),d in zip(pairs,deltas)},
            'paired_difference_sample_sd':float(deltas.std(ddof=1)) if len(pairs)>1 else None,
            'paired_difference_range':[float(deltas.min()),float(deltas.max())],
            'relative_mean_difference':float((mean[0]-mean[1])/mean[1]),
            'source_bootstrap_of_seed_mean':group_bootstrap(average_sources(1),average_sources(2),groups),
            'uncertainty_note':'Seed variation is shown separately; the bootstrap interval conditions on these trained seeds. Small seed counts cannot establish a precise population seed distribution.'}


def summarize(roots,snapshot,expected_seeds=(42,43,44)):
    snapshot=Path(snapshot);manifest=json.loads((snapshot/'manifest.json').read_text());runs={};identities={}
    for root in map(Path,roots):
        paths={p.parent:p for p in root.glob('*/evidence.json')}
        paths.update({p.parent:p for p in root.glob('*/metrics.json')})
        for path in paths.values():
            run=json.loads(path.read_text())
            if run.get('status')!='complete':continue
            cfg=run['config'];key=(cfg['regime'],cfg['seed'],cfg['label'])
            if cfg['seed'] not in expected_seeds:continue
            if cfg['snapshot_sha256']!=digest(snapshot/'manifest.json'):raise ValueError('Wrong snapshot')
            identity={k:v for k,v in cfg.items() if k not in ['seed','label','tokenizer_sha256']}
            basis=(cfg['regime'],cfg['label'])
            if basis in identities and identities[basis]!=(identity,cfg['tokenizer_sha256']):
                raise ValueError('Replication architecture, budget, tokenizer or implementation differs')
            identities[basis]=(identity,cfg['tokenizer_sha256'])
            if key in runs:raise ValueError(f'Duplicate seed arm: {key}')
            runs[key]=run
    results={};missing=[]
    for regime in ['raw','token']:
        pairs=[]
        for seed in expected_seeds:
            keys=[(regime,seed,a) for a in ['fit','general']]
            if not all(k in runs for k in keys):missing.append({'regime':regime,'seed':seed});continue
            fit,general=[runs[k] for k in keys];check_pair(fit,general)
            pairs.append((seed,fit['history'][-1]['validation'],general['history'][-1]['validation']))
        if pairs:
            docs=pairs[0][1]['source_documents']
            if 'repository_metadata' in manifest:
                spec=manifest['repository_metadata']['validation'];path=snapshot/spec['file']
                if digest(path)!=spec['file_sha256']:raise ValueError('Repository metadata changed')
                rows=[json.loads(line) for line in path.read_text().splitlines()][:len(docs)]
                if [r['sha256'] for r in rows]!=[d['sha256'] for d in docs]:raise ValueError('Repository alignment differs')
                groups=[r['repository_family'] for r in rows];unit='repository_family'
            else:groups=[d['sha256'] for d in docs];unit='document'
            results[regime]=dict(seed_summary(pairs,groups),source_unit=unit)
    return {'scope':'development seed replication; no final-test claim','expected_seeds':list(expected_seeds),
            'all_requested_pairs_complete':not missing,'missing_complete_pairs':missing,'regimes':results}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);a=p.parse_args()
    result=summarize(a.roots,a.snapshot,a.seeds);atomic_json(a.out,result);print(json.dumps(result,indent=2))
