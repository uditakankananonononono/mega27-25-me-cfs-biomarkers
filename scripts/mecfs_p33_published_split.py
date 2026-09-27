"""P33 registered comparator on authors' public seed-22 ME/CFS cfRNA split."""
import csv,hashlib,json,re,urllib.request
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score
root=Path(__file__).resolve().parents[1];folder=root/'data/geo/p33';folder.mkdir(parents=True,exist_ok=True)
base='https://raw.githubusercontent.com/aegardella/cfRNA_MECFS/main/Scripts/Manuscript_files/GLMNETLasso/'
files=['seed_22_all.metadata.test.tsv','seed_22_all.metadata.train.tsv','seed_22_all.counts.test.VST.tsv','seed_22_all.counts.train.VST.tsv','seed_22_all.GLMNETLasso.test.txt']
sha={}
for name in files:
 path=folder/name
 if not path.exists():urllib.request.urlretrieve(base+name,path)
 sha[name]=hashlib.sha256(path.read_bytes()).hexdigest()
meta={k:pd.read_csv(folder/f'seed_22_all.metadata.{k}.tsv',sep='\t') for k in ['train','test']}
x={k:pd.read_csv(folder/f'seed_22_all.counts.{k}.VST.tsv',sep='\t',index_col=0) for k in ['train','test']}
assert len(meta['train'])==108 and len(meta['test'])==60 and x['train'].shape==(108,31) and x['test'].shape==(60,31)
assert x['train'].columns.equals(x['test'].columns) and np.isfinite(x['train'].to_numpy()).all() and np.isfinite(x['test'].to_numpy()).all()
canon=lambda s:'cfs_cfrna_'+str(int(s.rsplit('_',1)[-1]))
geo=list(csv.DictReader((root/'results/mecfs_p29_gsm_audit.csv').open()));d={canon(r['column']):r for r in geo};assert len(d)==168
assert not set(meta['train'].sample_id).intersection(meta['test'].sample_id)
for k in ('train','test'):
 m=meta[k];assert m.sample_id.is_unique and x[k].index.is_unique and set(m.sample_id)==set(x[k].index) and set(m.phenotype)=={'MECFS','HC'}
 assert m.exclude.eq(0).all() and m.pre_post_cpet.isin(['D1-PRE-A','D1-PRE-B','D1-PRE-C']).all() and {canon(i) for i in m.sample_id}.issubset(d)
 for _,r in m.iterrows():
  a=d[canon(r.sample_id)];assert (r.phenotype=='MECFS')==(a['phenotype']=='phenotype: case')
  assert {'NYC':'New York City, NY','ITH':'Ithaca, NY','LA':'Los Angeles, CA'}[r.test_site]==a['site'].removeprefix('test site: ')
  assert {'F':'female','M':'male'}[r.sex]==a['sex'].removeprefix('Sex: ')
assert {canon(i) for i in set(meta['train'].sample_id)|set(meta['test'].sample_id)}==set(d)
y={k:meta[k].set_index('sample_id').loc[x[k].index,'phenotype'].eq('MECFS').astype(int).to_numpy() for k in ['train','test']}
model=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',solver='liblinear',max_iter=5000,random_state=20260925))
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260925)
fit=GridSearchCV(model,{'logisticregression__C':[.001,.01,.1,1,10,100]},scoring='roc_auc',cv=cv,n_jobs=1,refit=True)
fit.fit(x['train'],y['train']);score=fit.predict_proba(x['test'])[:,1];auc=roc_auc_score(y['test'],score)
rng=np.random.default_rng(20260925);pos=np.flatnonzero(y['test']);neg=np.flatnonzero(~y['test'].astype(bool));boot=[]
for _ in range(10000):
 ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
 boot.append(roc_auc_score(y['test'][ix],score[ix]))
out=pd.DataFrame({'sample_id':x['test'].index,'gsm':[d[canon(i)]['accession'] for i in x['test'].index],'phenotype':meta['test'].set_index('sample_id').loc[x['test'].index,'phenotype'].values,'p_MECFS':score})
out.to_csv(root/'results/mecfs_p33_test_predictions.csv',index=False)
results=fit.cv_results_;rec=dict(published_source='https://github.com/aegardella/cfRNA_MECFS',published_article='https://pmc.ncbi.nlm.nih.gov/articles/PMC12377778/',source_sha256=sha,n_train=108,n_test=60,n_features=31,case_train=int(y['train'].sum()),case_test=int(y['test'].sum()),best_c=fit.best_params_['logisticregression__C'],cv_mean_auc=float(fit.best_score_),cv_auc_by_c={str(c):float(results['mean_test_score'][i]) for i,c in enumerate([.001,.01,.1,1,10,100])},test_auc=float(auc),bootstrap_ci_95=[float(v) for v in np.quantile(boot,[.025,.975])],published_test_auc=.810267857142857,delta_vs_published_point=float(auc-.810267857142857),published_artifact_test_text=(folder/files[-1]).read_text())
(root/'results/mecfs_p33_result.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k not in ['source_sha256','published_artifact_test_text']},indent=2))
