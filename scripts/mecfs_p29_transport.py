"""P29 frozen leukocyte ME/CFS sign-panel transport to plasma cfRNA."""
import csv,hashlib,json,re,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from scipy.stats import ttest_ind
from ubiomark import geo,stats
root=Path(__file__).resolve().parents[1];folder=root/'data/geo/p29';folder.mkdir(parents=True,exist_ok=True)
sources=[('GSE293840_series_matrix.txt.gz','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293840/matrix/GSE293840_series_matrix.txt.gz','0d3026d48729fc0570610e1f5bc70a32709425471b2c3476956e240327c6ec3a'),('GSE293840_raw_counts_all.csv.gz','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293840/suppl/GSE293840_raw_counts_all.csv.gz','97ca1aa82b1098e75042a7aac5aa003bd3411d2bbf5cdde2da329f6e0f4965d8')]
for n,url,sha in sources:
 p=folder/n
 if not p.exists():urllib.request.urlretrieve(url,p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
_,ann,_=geo.parse_series_matrix(str(folder/sources[0][0]));assert len(ann)==168 and ann.index.is_unique and ann.Sample_title.is_unique
m={gsm:ann.loc[gsm,'Sample_description_1'].split(': ')[-1] for gsm in ann.index};assert len(set(m.values()))==168
for gsm,col in m.items():
 a=ann.loc[gsm];token=re.search(r'subject: (cfs_cfrna_\d+),',a.Sample_title).group(1)
 assert col==re.sub(r'_0+(\d+)$',r'_\1',token) and a.Sample_characteristics_ch1_2 in ('phenotype: case','phenotype: control')
 assert ('ME/CFS patient' in a.Sample_title)==(a.Sample_characteristics_ch1_2=='phenotype: case')
prior={r['accession'] for r in csv.DictReader((root/'results/dataset_manifest.csv').open())};assert not prior.intersection(ann.index)
x=pd.read_csv(folder/sources[1][0],index_col=0,low_memory=False);assert len(x.columns)==168 and set(x.columns)==set(m.values()) and x.index.is_unique
assert x.index.str.fullmatch(r'ENSG\d+\.\d+(?:_PAR_Y)?').all()
values=x.to_numpy(dtype=float);assert np.isfinite(values).all() and (values>=0).all() and np.equal(values,np.floor(values)).all()
h=pd.read_csv(geo.HGNC_PATH,sep='\t',dtype=str,usecols=['symbol','ensembl_gene_id']).dropna();amb=set(h.loc[h.ensembl_gene_id.duplicated(keep=False),'ensembl_gene_id']);h=h[~h.ensembl_gene_id.isin(amb)];mapping=dict(zip(h.ensembl_gene_id,h.symbol))
expr=pd.DataFrame(values,columns=x.columns);expr['gene']=x.index.str.split('.').str[0].map(mapping);expr=expr.dropna(subset=['gene']).groupby('gene').sum();assert expr.index.is_unique
lib=expr.sum(axis=0);assert (lib>0).all();cpm=expr.div(lib,axis=1)*1e6
ordered=list(ann.index);case=[m[g] for g in ordered if ann.loc[g,'Sample_characteristics_ch1_2']=='phenotype: case'];ctrl=[m[g] for g in ordered if ann.loc[g,'Sample_characteristics_ch1_2']=='phenotype: control'];assert (len(case),len(ctrl))==(93,75)
log=np.log2(cpm.loc[(cpm>1).mean(axis=1)>=.2]+1)
g,v=stats.hedges_g(log[case].to_numpy(),log[ctrl].to_numpy());effect=pd.DataFrame({'g':g,'v':v},index=log.index).replace([np.inf,-np.inf],np.nan).dropna()
D=pd.read_csv(root/'results/meta_discovery/me_cfs.csv.gz',index_col=0);C=pd.read_csv(root/'results/mecfs_p29_fixed_panel.csv').set_index('gene');assert len(C)==16 and (C.k>=2).all()
obs=C.index.intersection(effect.index);cp=int((C.loc[obs,'mu']>0).sum());cn=len(obs)-cp
agreement=np.sign(C.loc[obs,'mu'])==np.sign(effect.loc[obs,'g'])
pool=D.index.intersection(effect.index);pool=pool[(D.loc[pool,'k']>=2)&~pool.isin(C.index)];up=np.array(pool[D.loc[pool,'mu']>0]);down=np.array(pool[D.loc[pool,'mu']<0]);assert len(up)>=cp and len(down)>=cn
rng=np.random.default_rng(20260925);null=np.zeros(10000,dtype=int)
for i in range(10000):
 picks=list(rng.choice(up,cp,replace=False))+list(rng.choice(down,cn,replace=False))
 null[i]=int((np.sign(D.loc[picks,'mu'])==np.sign(effect.loc[picks,'g'])).sum())
rawp=ttest_ind(log.loc[obs,case].to_numpy(),log.loc[obs,ctrl].to_numpy(),axis=1,equal_var=False).pvalue;adjusted=stats.bh_fdr(rawp)
with (root/'results/mecfs_p29_genes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['gene','discovery_mu','cohort_g','cohort_v','agrees','welch_p','panel_bh_q','status'],lineterminator='\n');w.writeheader()
 for gene in C.index:
  if gene in obs:
   i=obs.get_loc(gene);w.writerow(dict(gene=gene,discovery_mu=float(C.loc[gene,'mu']),cohort_g=float(effect.loc[gene,'g']),cohort_v=float(effect.loc[gene,'v']),agrees=int(agreement.loc[gene]),welch_p=float(rawp[i]),panel_bh_q=float(adjusted[i]),status='measured'))
  else:w.writerow(dict(gene=gene,discovery_mu=float(C.loc[gene,'mu']),cohort_g='',cohort_v='',agrees='',welch_p='',panel_bh_q='',status='not uniquely mapped or filtered'))
with (root/'results/mecfs_p29_samples.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['gsm','column','title','phenotype','batch','site','sex'],lineterminator='\n');w.writeheader()
 for gsm in ann.index:
  r=ann.loc[gsm];w.writerow(dict(gsm=gsm,column=m[gsm],title=r.Sample_title,phenotype=r.Sample_characteristics_ch1_2,batch=r.Sample_characteristics_ch1_1,site=r.Sample_characteristics_ch1_4,sex=r.Sample_characteristics_ch1_3))
strata=[]
for field in ['Sample_characteristics_ch1_1','Sample_characteristics_ch1_4','Sample_characteristics_ch1_3']:
 for level in sorted(ann[field].unique()):
  z=ann[ann[field]==level];c=[m[g] for g in z.index if z.loc[g,'Sample_characteristics_ch1_2']=='phenotype: case'];h=[m[g] for g in z.index if z.loc[g,'Sample_characteristics_ch1_2']=='phenotype: control']
  if min(len(c),len(h))<3:continue
  ga,_=stats.hedges_g(log.loc[obs,c].to_numpy(),log.loc[obs,h].to_numpy())
  for j,gene in enumerate(obs):strata.append(dict(stratum_field=field,stratum=level,gene=gene,n_case=len(c),n_control=len(h),g=float(ga[j]) if np.isfinite(ga[j]) else '',agrees_discovery=int(np.sign(ga[j])==np.sign(C.loc[gene,'mu'])) if np.isfinite(ga[j]) else ''))
pd.DataFrame(strata).to_csv(root/'results/mecfs_p29_strata.csv',index=False)
pd.DataFrame({'matched_random_agreements':null}).to_csv(root/'results/mecfs_p29_null.csv.gz',index=False)
count=int(agreement.sum());p=float((1+(null>=count).sum())/10001)
rec=dict(source='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293840',sha256={n:s for n,_,s in sources},n_case=93,n_control=75,n_unique_mapped_genes=len(expr),n_effect_genes=len(effect),n_fixed_measured=len(obs),missing_fixed=list(C.index.difference(obs)),n_positive=cp,n_negative=cn,n_sign_match=count,null_mean=float(null.mean()),empirical_p=p,registered_descriptive_support=bool(len(obs)>=12 and count>=12 and p<.025),panel_bh_hits=int((adjusted<.05).sum()),n_stratum_gene_results=len(strata),ambiguous_ensembl_ids=len(amb))
(root/'results/mecfs_p29_result.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
