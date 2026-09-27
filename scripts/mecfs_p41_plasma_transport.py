"""P41 fixed leukocyte sign-panel transport to pre-exercise plasma particles."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ttest_ind
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ubiomark import geo,stats
root=Path(__file__).resolve().parents[1];folder=root/'data/geo/p41';p=folder/'GSE236402_raw_counts.txt.gz';expected='0ccb55030d0af76c5d35e45670fbf0c8838d7ce7d43a590f0c4b09e9cdf85240';assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
rows=list(csv.DictReader((root/'results/mecfs_p41_samples.csv').open()));assert len(rows)==12 and len(set(z['gsm'] for z in rows))==len(set(z['column'] for z in rows))==12
for z in rows:assert hashlib.sha256((folder/'gsm'/(z['gsm']+'.txt')).read_bytes()).hexdigest()==z['sha256']
x=pd.read_csv(p,sep='\t',index_col=0);assert x.index.is_unique and x.index.str.fullmatch(r'ENSG\d+').all() and set(x.columns)=={z['column'] for z in rows} and (x.to_numpy()>=0).all() and np.isfinite(x.to_numpy()).all() and np.equal(x.to_numpy(),np.floor(x.to_numpy())).all()
d1=[z for z in rows if z['timepoint']=='D1-preCPET'];assert len(d1)==6 and len({z['subject'] for z in d1})==6
h=pd.read_csv(geo.HGNC_PATH,sep='\t',dtype=str,usecols=['symbol','ensembl_gene_id']).dropna();amb=set(h.loc[h.ensembl_gene_id.duplicated(keep=False),'ensembl_gene_id']);h=h[~h.ensembl_gene_id.isin(amb)];mapping=dict(zip(h.ensembl_gene_id,h.symbol));y=x[[z['column'] for z in d1]].copy();y['gene']=y.index.map(mapping);expr=y.dropna(subset=['gene']).groupby('gene').sum();assert expr.index.is_unique
lib=expr.sum(axis=0);assert (lib>0).all();cpm=expr.div(lib,axis=1)*1e6;log=np.log2(cpm.loc[(cpm>1).mean(axis=1)>=.2]+1)
case=[z['column'] for z in d1 if z['group']=='case'];ctrl=[z['column'] for z in d1 if z['group']=='control'];assert len(case)==len(ctrl)==3
g,v=stats.hedges_g(log[case].to_numpy(),log[ctrl].to_numpy());E=pd.DataFrame({'g':g,'v':v},index=log.index).replace([np.inf,-np.inf],np.nan).dropna()
D=pd.read_csv(root/'results/meta_discovery/me_cfs.csv.gz',index_col=0);C=pd.read_csv(root/'results/mecfs_p29_fixed_panel.csv').set_index('gene');assert len(C)==16 and (C.k>=2).all()
obs=C.index.intersection(E.index);upcount=int((C.loc[obs,'mu']>0).sum());downcount=len(obs)-upcount;agree=np.sign(C.loc[obs,'mu'])==np.sign(E.loc[obs,'g'])
pool=D.index.intersection(E.index);pool=pool[(D.loc[pool,'k']>=2)&~pool.isin(C.index)];up=pool[D.loc[pool,'mu']>0].to_numpy();down=pool[D.loc[pool,'mu']<0].to_numpy();assert len(up)>=upcount and len(down)>=downcount
rng=np.random.default_rng(20260925);null=np.zeros(10000,dtype=int)
for i in range(10000):
 genes=list(rng.choice(up,upcount,replace=False))+list(rng.choice(down,downcount,replace=False));null[i]=int((np.sign(D.loc[genes,'mu'])==np.sign(E.loc[genes,'g'])).sum())
p=ttest_ind(log.loc[obs,case].to_numpy(),log.loc[obs,ctrl].to_numpy(),axis=1,equal_var=False).pvalue
with (root/'results/mecfs_p41_genes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['gene','discovery_mu','target_g','target_v','welch_p','agrees','status'],lineterminator='\n');w.writeheader()
 for gene in C.index:
  if gene in obs:
   i=obs.get_loc(gene);w.writerow(dict(gene=gene,discovery_mu=float(C.loc[gene,'mu']),target_g=float(E.loc[gene,'g']),target_v=float(E.loc[gene,'v']),welch_p=float(p[i]),agrees=int(agree.loc[gene]),status='measured'))
  else:w.writerow(dict(gene=gene,discovery_mu=float(C.loc[gene,'mu']),target_g='',target_v='',welch_p='',agrees='',status='not uniquely mapped or expression filtered'))
pd.DataFrame({'matched_random_agreements':null}).to_csv(root/'results/mecfs_p41_null.csv.gz',index=False)
count=int(agree.sum());pv=float((1+(null>=count).sum())/10001);corrected=int(sum((agree.to_numpy())&(p<.05/16)))
rec=dict(source='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236402',matrix_sha256=expected,n_case=3,n_control=3,n_repeated_D2_excluded=6,n_unique_mapped_genes=len(expr),n_effect_genes=len(E),n_fixed_measured=len(obs),missing_fixed=list(C.index.difference(obs)),n_positive=upcount,n_negative=downcount,n_sign_match=count,null_mean=float(null.mean()),empirical_p=pv,n_corrected_same_direction=corrected,registered_descriptive_support=bool(len(obs)>=12 and count>=12 and pv<.025 and corrected>=1),ambiguous_ensembl_ids=len(amb),limitation='Three people per group, uncertain participant linkage across studies, different assay; repeated D2 excluded.')
(root/'results/mecfs_p41_result.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
