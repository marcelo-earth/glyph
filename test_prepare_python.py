import random
import tempfile
from pathlib import Path
import unittest

from corpus_snapshot import load_snapshot,text_sha256
from prepare_python import repository_family,select_files,deduplicate_shingles,token_shingles,split_repositories,prepare_python_snapshot


def record(i,text):
    return {'text':text,'sha256':text_sha256(text),'repo_name':f'owner/project{i}',
            'repository_family':f'project{i}','path':'file.py','license':'mit','source_shard':'offline','source_line':i}


class PythonPreparationTests(unittest.TestCase):
    def test_grouped_split_keeps_same_name_forks_together(self):
        rows=[record(i,f'file {i}') for i in range(12)]
        rows[1]['repository_family']=rows[0]['repository_family']
        rows[1]['repo_name']='fork/PROJECT0'
        self.assertEqual(repository_family(rows[1]['repo_name']),'project0')
        groups=split_repositories(rows,2,2,17)
        seen=set()
        for items in groups.values():
            families={r['repository_family'] for r in items}
            self.assertFalse(seen & families);seen |= families
        self.assertEqual(sum(map(len,groups.values())),len(rows))

    def test_prefix_filter_matches_exhaustive_greedy_jaccard(self):
        rng=random.Random(123)
        texts=[]
        for i in range(20):
            base=[str(rng.randrange(30)) for _ in range(40)]
            texts.append(' '.join(base))
            for j in range(3):
                changed=base.copy();changed[j]=str(100+j)
                texts.append(' '.join(changed))
        records=[record(i,t) for i,t in enumerate(texts)]
        sets=[token_shingles(t) for t in texts]
        for threshold in [.5,.85,1.0]:
            kept=[]
            for i in sorted(range(len(sets)),key=lambda j:(len(sets[j]),j)):
                if not any(len(sets[i]&sets[j])/len(sets[i]|sets[j])>=threshold for j in kept):kept.append(i)
            actual,_=deduplicate_shingles(records,threshold)
            self.assertEqual([r['sha256'] for r in actual],[records[i]['sha256'] for i in sorted(kept)])

    def test_snapshot_preserves_metadata_and_hashes(self):
        rows=[{'content':f'def function_{i}():\n    return {i} + {i+2}\n',
               'repo_name':f'owner/project{i}','path':'module.py','license':'mit'} for i in range(15)]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'snapshot'
            snapshot=prepare_python_snapshot(out,max_docs=15,max_scanned=20,min_chars=10,
                max_chars=100,val_docs=2,test_docs=2,tokenizer_chars=100,
                rows=iter(rows),general_texts=['general mismatched prose with sufficient unique words '*5])
            self.assertEqual(snapshot['manifest']['repository_metadata']['validation']['repositories'],2)
            self.assertEqual(load_snapshot(out)['train'],snapshot['train'])
            with (out/'train.metadata.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'metadata checksum'):load_snapshot(out)

if __name__=='__main__':unittest.main()
