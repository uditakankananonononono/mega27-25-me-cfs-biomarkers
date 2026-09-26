"""ubiomark command-line tool.

  ubiomark series GSE12345 --disease fibromyalgia     # download, label, per-gene Hedges g
  ubiomark meta results/series/fibromyalgia__*.csv.gz  # DerSimonian-Laird random effects
  ubiomark novelty GENE --disease fibromyalgia         # PubMed co-mention count
  ubiomark rank META.csv.gz --disease fibromyalgia     # GraphSAGE+RWR ranking of novel candidates
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd


def cmd_series(a):
    from . import geo, labels, stats
    for path in geo.download_matrices(a.gse):
        expr, ann, _ = geo.parse_series_matrix(path)
        lab = labels.label_samples(ann, a.disease, a.gse)
        print(f"{path}: case={int((lab=='case').sum())} control={int((lab=='control').sum())}", file=sys.stderr)
        if (lab == "case").sum() < 3 or (lab == "control").sum() < 3:
            continue
        gx = geo.to_gene_level(expr, geo.probe_to_symbol(ann["Sample_platform_id"].iloc[0]))
        g, v = stats.hedges_g(gx[lab[lab == "case"].index].to_numpy(float), gx[lab[lab == "control"].index].to_numpy(float))
        pd.DataFrame({"gene": gx.index, "g": g, "v": v}).dropna().to_csv(a.out or sys.stdout, index=False)


def cmd_meta(a):
    from . import stats
    fr = [pd.read_csv(f).set_index("gene").add_suffix(f"_{i}") for i, f in enumerate(a.files)]
    X = pd.concat(fr, axis=1)
    r = stats.dersimonian_laird(X.filter(like="g_").to_numpy(float), X.filter(like="v_").to_numpy(float))
    o = pd.DataFrame(r, index=X.index)
    o["q"] = stats.bh_fdr(o.p.to_numpy())
    o.sort_values("p").to_csv(a.out or sys.stdout)


def cmd_novelty(a):
    from . import novelty
    print(novelty.comention(a.gene, a.disease))


def cmd_rank(a):
    import torch, scipy.sparse as sp
    from . import network, models
    meta = pd.read_csv(a.meta, index_col=0)
    genes, A = network.load_string(400)
    idx = {g: i for i, g in enumerate(genes)}
    ot = pd.read_csv(a.known)
    y = np.zeros(len(genes))
    for s in ot.iloc[:, 0 if "symbol" not in ot else list(ot.columns).index("symbol")]:
        if s in idx:
            y[idx[s]] = 1
    F = np.zeros((len(genes), 4), np.float32)
    for g, r in meta.iterrows():
        if g in idx:
            F[idx[g]] = [r.mu, r.z, -np.log10(max(r.p, 1e-300)), 1]
    rw = network.rwr(A, y)
    F = np.column_stack([F, np.log(rw + 1e-12)]).astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    deg = np.asarray(A.sum(1)).ravel()
    Am = models.to_torch_sparse(sp.diags(1 / np.maximum(deg, 1)) @ A)
    s = models.train_node_model(models.seeded_node_model(models.SAGE, F.shape[1], seed=0), torch.tensor(F), Am, y, np.ones(len(y), bool))
    o = pd.DataFrame({"gene": genes, "score": s, "known": y.astype(bool)}).join(meta[["mu", "p", "q"]], on="gene")
    o[~o.known].sort_values("score", ascending=False).head(a.top).to_csv(a.out or sys.stdout, index=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ubiomark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("series"); s.add_argument("gse"); s.add_argument("--disease", required=True); s.add_argument("--out"); s.set_defaults(f=cmd_series)
    s = sub.add_parser("meta"); s.add_argument("files", nargs="+"); s.add_argument("--out"); s.set_defaults(f=cmd_meta)
    s = sub.add_parser("novelty"); s.add_argument("gene"); s.add_argument("--disease", required=True); s.set_defaults(f=cmd_novelty)
    s = sub.add_parser("rank"); s.add_argument("meta"); s.add_argument("--known", required=True, help="CSV with a 'symbol' column of known disease genes")
    s.add_argument("--disease"); s.add_argument("--top", type=int, default=30); s.add_argument("--out"); s.set_defaults(f=cmd_rank)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
