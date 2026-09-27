"""P32 fixed ME/CFS leukocyte discovery signs in NIH baseline muscle counts."""
import csv,hashlib,json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from scipy.stats import ttest_ind
from ubiomark import geo,stats
root=Path(__file__).resolve().parents[1];folder=root/'data/geo/p32'
rows=list(csv.DictReader((root/'results/mecfs_p32_samples.csv').open()))
assert len(rows)==25 and len({r['gsm'] for r in rows})==25 and len({re.search(r'MECFS_(\d+)',r['title']).group(1) for r in rows})==25
prior={r['accession'] for r in csv.DictReader((root/'results/dataset_manifest.csv').open())};assert not prior.intersection(r['gsm'] for r in rows)
source={};frames=[];index=None
for r in rows:
 assert 'Vastus Lateralis Muscle' in r['title'] and 'Baseline' in r['title']
 assert r['status'] in ('subject status: healthy volunteer (HV)','subject status: post-infectious Myalgic encephalomyelitis/chronic fatigue syndrome (PI-ME/CFS) volunteer')
 assert ('PI-ME/CFS' in r['title'])==(r['status'].startswith('subject status: post-infectious'))
 p=folder/r['column'];source[r['gsm']]=hashlib.sha256(p.read_bytes()).hexdigest()
 x=pd.read_csv(p,sep='\t',dtype={'ensembl_gene_id':str,'external_gene_name':str});assert x.shape[1]==3 and x.columns[0:2].tolist()==['ensembl_gene_id','external_gene_name']
 assert x.ensembl_gene_id.str.fullmatch(r'ENSG\d+').all() and x.ensembl_gene_id.is_unique
 if index is None:index=x.ensembl_gene_id.tolist();symbols=x.external_gene_name.tolist()
 else:assert index==x.ensembl_gene_id.tolist() and symbols==x.external_gene_name.tolist()
 col=x.columns[2];assert col.split('_')[-1] == re.search(r'MECFS_(\d+)',r['title']).group(1)
 val=x.iloc[:,2].to_numpy(dtype=float);assert np.isfinite(val).all() and (val>=0).all() and np.equal(val,np.floor(val)).all()
 frames.append(pd.Series(val,index=index,name=r['gsm']))
counts=pd.concat(frames,axis=1)
h=pd.read_csv(geo.HGNC_PATH,sep='\t',dtype=str,usecols=['symbol','ensembl_gene_id']).dropna();amb=set(h.loc[h.ensembl_gene_id.duplicated(keep=False),'ensembl_gene_id']);h=h[~h.ensembl_gene_id.isin(amb)];mapping=dict(zip(h.ensembl_gene_id,h.symbol))
counts['gene']=counts.index.map(mapping);expr=counts.dropna(subset=['gene']).groupby('gene').sum();assert expr.index.is_unique
lib=expr.sum(axis=0);assert (lib>0).all();cpm=expr.div(lib,axis=1)*1e6;log=np.log2(cpm.loc[(cpm>1).mean(axis=1)>=.2]+1)
case=[r['gsm'] for r in rows if 'PI-ME/CFS' in r['title']];ctrl=[r['gsm'] for r in rows if 'HV' in r['title']];assert len(case)==13 and len(ctrl)==12
g,v=stats.hedges_g(log[case].to_numpy(),log[ctrl].to_numpy());E=pd.DataFrame({'g':g,'v':v},index=log.index).replace([np.inf,-np.inf],np.nan).dropna()
D=pd.read_csv(root/'results/meta_discovery/me_cfs.csv.gz',index_col=0);C=pd.read_csv(root/'results/mecfs_p29_fixed_panel.csv').set_index('gene');assert len(C)==16 and (C.k>=2).all()
obs=C.index.intersection(E.index);cp=int((C.loc[obs,'mu']>0).sum());cn=len(obs)-cp
agree=np.sign(C.loc[obs,'mu'])==np.sign(E.loc[obs,'g']);pool=D.index.intersection(E.index);pool=pool[(D.loc[pool,'k']>=2)&~pool.isin(C.index)]
up=np.array(pool[D.loc[pool,'mu']>0]);down=np.array(pool[D.loc[pool,'mu']<0]);assert len(up)>=cp and len(down)>=cn
rng=np.random.default_rng(20260925);null=np.zeros(10000,dtype=int)
for i in range(10000):
 picks=list(rng.choice(up,cp,replace=False))+list(rng.choice(down,cn,replace=False))
 null[i]=int((np.sign(D.loc[picks,'mu'])==np.sign(E.loc[picks,'g'])).sum())
p=ttest_ind(log.loc[obs,case].to_numpy(),log.loc[obs,ctrl].to_numpy(),axis=1,equal_var=False).pvalue
with (root/'results/mecfs_p32_genes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['gene','discovery_mu','muscle_g','muscle_v','welch_p','agrees','status'],lineterminator='\n');w.writeheader()
 for gene in C.index:
  if gene in obs:
   i=obs.get_loc(gene);w.writerow(dict(gene=gene,discovery_mu=float(C.loc[gene,'mu']),muscle_g=float(E.loc[gene,'g']),muscle_v=float(E.loc[gene,'v']),welch_p=float(p[i]),agrees=int(agree.loc[gene]),status='measured'))
  else:w.writerow(dict(gene=gene,discovery_mu=float(C.loc[gene,'mu']),muscle_g='',muscle_v='',welch_p='',agrees='',status='not uniquely mapped or filtered'))
pd.DataFrame({'matched_random_agreements':null}).to_csv(root/'results/mecfs_p32_null.csv.gz',index=False)
count=int(agree.sum());pv=float((1+(null>=count).sum())/10001)
rec=dict(source='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245661',source_sha256=source,n_case=13,n_control=12,n_unique_mapped_genes=len(expr),n_effect_genes=len(E),n_fixed_measured=len(obs),missing_fixed=list(C.index.difference(obs)),n_positive=cp,n_negative=cn,n_sign_match=count,null_mean=float(null.mean()),empirical_p=pv,registered_descriptive_support=bool(len(obs)>=12 and count>=12 and pv<.025),ambiguous_ensembl_ids=len(amb))
(root/'results/mecfs_p32_result.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k!='source_sha256'},indent=2))
