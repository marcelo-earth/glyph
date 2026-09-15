"""Export inspectable source-level evidence while retaining local block diagnostics."""
import argparse
import json
import math
from pathlib import Path
from experiment import atomic_json,digest


def export(path):
    path=Path(path);result=json.loads(path.read_text())
    if result.get('status')!='complete':raise ValueError('Only completed runs can be exported')
    for row in result['history']:
        scores=row['validation'];docs=scores['source_documents']
        if sum(d['utf8_bytes'] for d in docs)!=scores['utf8_bytes']:raise ValueError('Source byte total mismatch')
        if not math.isclose(math.fsum(d['nll_nats'] for d in docs),scores['nll_nats'],rel_tol=1e-12):
            raise ValueError('Source NLL total mismatch')
        scores.pop('documents',None)
    result['evidence_export']={'original_metrics_sha256':digest(path),
                               'omitted':'per-block records; all per-source statistics retained'}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('metrics',nargs='+');a=p.parse_args()
    for file in a.metrics:
        dest=Path(file).with_name('evidence.json');atomic_json(dest,export(file));print(dest)
