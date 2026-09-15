"""Reproducible scientific figures from completed, source-backed Glyph evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from run_matrix import check_pair
from experiment import atomic_json,digest


def plot_pilot(root,output):
    root=Path(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    diagnostics=json.loads((root/'tokenizer_diagnostics.json').read_text())
    colors={'fit':'#245CB3','general':'#BC641D'}
    names={'fit':'Corpus-fit','general':'Mismatched prose'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#9CA3AF','text.color':'#17202B',
                         'axes.labelcolor':'#17202B','xtick.color':'#475569','ytick.color':'#475569'})
    fig,axes=plt.subplots(2,2,figsize=(11.5,8),sharex='col',sharey='row')
    source_rows=[];source_hashes={}
    for col,regime in enumerate(['raw','token']):
        paths={a:root/f'{regime}-{a}-s42'/'evidence.json' for a in ['fit','general']}
        runs={a:json.loads(p.read_text()) for a,p in paths.items()}
        check_pair(runs['fit'],runs['general'])
        for run in runs.values():
            if (run['parameters'],run['config']['steps'],run['config']['seq_len'],run['config']['vocab_size'],run['config']['val_docs']) != (2590848,500,128,4096,50):
                raise ValueError('Figure labels are specific to the registered Python pilot')
            if run['history'][-1]['validation']['utf8_bytes'] != 273747:
                raise ValueError('Figure validation population differs')
        for arm,run in runs.items():
            baseline=diagnostics['tokenizers'][arm]['segmentations']['shared_raw_blocks']['baselines']['unigram']
            source_hashes[str(paths[arm])]=digest(paths[arm])
            steps=[r['step'] for r in run['history']]
            values=[r['validation']['bits_per_utf8_byte'] for r in run['history']]
            for row in run['history']:
                scores=row['validation']
                assert scores['utf8_bytes']==baseline['utf8_bytes']
                assert [d['sha256'] for d in scores['source_documents']]==[d['sha256'] for d in baseline['source_documents']]
                source_rows.append({'regime':regime,'arm':arm,'seed':42,'step':row['step'],
                    'bits_per_utf8_byte':scores['bits_per_utf8_byte'],
                    'full_train_unigram_bpb':baseline['bits_per_utf8_byte'],
                    'model_gain_over_unigram_bpb':baseline['bits_per_utf8_byte']-scores['bits_per_utf8_byte'],
                    'validation_bytes':scores['utf8_bytes']})
            marker='o' if arm=='fit' else 's'
            kwargs={'color':colors[arm],'linewidth':2,'marker':marker,'markersize':4,
                    'markerfacecolor':colors[arm] if arm=='fit' else 'white'}
            axes[0,col].plot(steps,values,**kwargs)
            axes[0,col].axhline(baseline['bits_per_utf8_byte'],color=colors[arm],linestyle='--',linewidth=1.2,alpha=.75)
            axes[1,col].plot(steps,[baseline['bits_per_utf8_byte']-v for v in values],**kwargs)
            axes[0,col].annotate(f'{values[-1]:.3f}',(steps[-1],values[-1]),xytext=(7,0),textcoords='offset points',
                                 color=colors[arm],va='center',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','pad':1.2})
            axes[1,col].annotate(f'{baseline["bits_per_utf8_byte"]-values[-1]:+.3f}',
                                 (steps[-1],baseline['bits_per_utf8_byte']-values[-1]),xytext=(7,0),
                                 textcoords='offset points',color=colors[arm],va='center',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','pad':1.2})
        axes[0,col].set_title('Equal raw text + shared context' if regime=='raw' else 'Equal packed-token budget',loc='left',fontweight='bold',pad=14)
        for row in range(2):
            ax=axes[row,col];ax.grid(axis='y',color='#E5E7EB',linewidth=.7);ax.set_axisbelow(True)
            ax.set_xlim(0,570);ax.set_xticks([0,100,200,300,400,500])
        axes[0,col].set_ylim(0,8)
        axes[1,col].set_ylim(-3.4,.85);axes[1,col].axhline(0,color='#475569',linewidth=1)
        axes[1,col].set_xlabel('Optimizer updates')
    axes[0,0].set_ylabel('Canonical bits per UTF-8 byte ↓')
    axes[1,0].set_ylabel('Improvement over unigram (bits/byte) ↑')
    fig.suptitle('Glyph · Python development pilot',x=.08,y=.985,ha='left',fontsize=18,fontweight='bold')
    fig.text(.08,.935,'2.59M parameters · 4,096-token vocabularies · seed 42 · identical 128-byte evaluation blocks',fontsize=10,color='#475569')
    handles=[Line2D([0],[0],color=colors[a],marker='o' if a=='fit' else 's',markerfacecolor=colors[a] if a=='fit' else 'white',lw=2,label=names[a]) for a in colors]
    handles.append(Line2D([0],[0],color='#475569',linestyle='--',lw=1.2,label='Full-training unigram (arm color)'))
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.072,.067),ncol=3,frameon=False,fontsize=10)
    fig.text(.08,.047,'50 validation files · 43 repository families · 273,747 bytes. One training seed; no seed-uncertainty bands.',fontsize=9,color='#475569')
    fig.text(.08,.024,'Early development result, not a final test conclusion. Baselines use the full training split; gains are descriptive.',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.08,right=.95,top=.87,bottom=.16,hspace=.22,wspace=.15)
    for suffix in ['svg','png','pdf']:fig.savefig(output.with_suffix('.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)
    atomic_json(output.with_suffix('.sources.json'),{'source_sha256':source_hashes,
                'diagnostics_sha256':digest(root/'tokenizer_diagnostics.json'),'rows':source_rows})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();plot_pilot(a.root,a.out)
