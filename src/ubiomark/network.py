"""STRING graph loading and classical network-propagation baselines.

Random walk with restart (RWR): p_{t+1} = (1-r) W p_t + r p_0, W column-normalised adjacency;
fixed point p* = r (I - (1-r) W)^{-1} p_0, reached by power iteration."""
from __future__ import annotations
import gzip, os
import numpy as np
import pandas as pd
import scipy.sparse as sp

STRING_DIR = os.path.expanduser("~/mega27-25/data/raw/string")


def load_string(min_score: int = 400, links=None, info=None) -> tuple[list[str], sp.csr_matrix]:
    links = links or os.path.join(STRING_DIR, "9606.protein.physical.links.v12.0.txt.gz")
    info = info or os.path.join(STRING_DIR, "9606.protein.info.v12.0.txt.gz")
    inf = pd.read_csv(info, sep="\t", usecols=[0, 1])
    inf.columns = ["pid", "sym"]
    m = dict(zip(inf.pid, inf.sym))
    df = pd.read_csv(links, sep=" ")
    df = df[df.combined_score >= min_score]
    a, b = df.protein1.map(m), df.protein2.map(m)
    genes = sorted(set(a) | set(b))
    idx = {g: i for i, g in enumerate(genes)}
    r, c = a.map(idx).to_numpy(), b.map(idx).to_numpy()
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(genes), len(genes))).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    return genes, A.tocsr()


def rwr(A: sp.csr_matrix, seeds: np.ndarray, restart: float = 0.5, tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    deg = np.asarray(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1
    W = A @ sp.diags(1.0 / deg)
    p0 = seeds.astype(float)
    p0 = p0 / p0.sum() if p0.sum() > 0 else np.full(len(p0), 1.0 / len(p0))
    p = p0.copy()
    for _ in range(max_iter):
        pn = (1 - restart) * (W @ p) + restart * p0
        if np.abs(pn - p).sum() < tol:
            return pn
        p = pn
    return p


def sym_norm(A: sp.csr_matrix) -> sp.csr_matrix:
    """GCN normalisation  Â = D^{-1/2}(A+I)D^{-1/2}."""
    A = A + sp.eye(A.shape[0], dtype=np.float32)
    d = np.asarray(A.sum(axis=1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    return (Dm @ A @ Dm).tocsr().astype(np.float32)


def fiedler_order(A: sp.csr_matrix, genes: list[str], keep: list[str]) -> list[str]:
    """Order `keep` genes by the Fiedler vector of the induced subgraph Laplacian (spectral seriation),
    so that PPI neighbours sit next to each other for the 1-D CNN."""
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    idx = {g: i for i, g in enumerate(genes)}
    ks = [g for g in keep if g in idx]
    sub = A[[idx[g] for g in ks]][:, [idx[g] for g in ks]] + sp.eye(len(ks)) * 1e-3
    L = laplacian(sub.astype(float), normed=True)
    vals, vecs = eigsh(L, k=2, sigma=-1e-3, which="LM")
    order = np.argsort(vecs[:, np.argsort(vals)[1]])
    rest = [g for g in keep if g not in idx]
    return [ks[i] for i in order] + rest
