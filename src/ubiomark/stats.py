"""Effect sizes and random-effects meta-analysis.

Hedges' g (bias-corrected standardised mean difference):
    d = (m1 - m0) / s_p,  s_p^2 = ((n1-1)s1^2 + (n0-1)s0^2)/(n1+n0-2)
    J = 1 - 3/(4(n1+n0) - 9),  g = J d
    Var(g) ~= (n1+n0)/(n1 n0) + g^2 / (2 (n1+n0))
DerSimonian-Laird random effects:
    w_i = 1/v_i, Q = sum w_i (g_i - g_FE)^2,  tau^2 = max(0, (Q-(k-1)) / (sum w - sum w^2/sum w))
    w*_i = 1/(v_i + tau^2), mu = sum w* g / sum w*, SE = 1/sqrt(sum w*), I^2 = max(0,(Q-(k-1))/Q)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def hedges_g(x1: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x1, x0: genes x samples arrays (NaN allowed). Returns (g, var) per gene."""
    n1 = np.sum(np.isfinite(x1), axis=1).astype(float)
    n0 = np.sum(np.isfinite(x0), axis=1).astype(float)
    m1, m0 = np.nanmean(x1, axis=1), np.nanmean(x0, axis=1)
    s1, s0 = np.nanvar(x1, axis=1, ddof=1), np.nanvar(x0, axis=1, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (m1 - m0) / sp
        J = 1 - 3 / (4 * (n1 + n0) - 9)
        g = J * d
        v = (n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0))
    bad = (n1 < 2) | (n0 < 2) | ~np.isfinite(g) | (sp <= 0)
    g[bad] = np.nan
    v[bad] = np.nan
    return g, v


def dersimonian_laird(g: np.ndarray, v: np.ndarray) -> dict:
    """g, v: genes x studies (NaN = missing). Returns dict of arrays per gene."""
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0)
    w = np.where(ok, 1.0 / np.where(ok, v, 1.0), 0.0)
    gg = np.where(ok, g, 0.0)
    k = ok.sum(axis=1)
    sw = w.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fe = (w * gg).sum(axis=1) / sw
        Q = (w * (gg - fe[:, None]) ** 2).sum(axis=1)
        c = sw - (w ** 2).sum(axis=1) / sw
        tau2 = np.maximum(0.0, (Q - (k - 1)) / c)
        tau2 = np.where(k > 1, tau2, 0.0)
        ws = np.where(ok, 1.0 / (np.where(ok, v, 1.0) + tau2[:, None]), 0.0)
        mu = (ws * gg).sum(axis=1) / ws.sum(axis=1)
        se = 1.0 / np.sqrt(ws.sum(axis=1))
        z = mu / se
        I2 = np.where(Q > 0, np.maximum(0.0, (Q - (k - 1)) / Q), 0.0)
    p = 2 * stats.norm.sf(np.abs(z))
    return {"mu": mu, "se": se, "z": z, "p": p, "tau2": tau2, "Q": Q, "I2": I2, "k": k}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg: q_(i) = min_{j>=i} p_(j) m / j."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    ps = p[ok]
    m = len(ps)
    if m == 0:
        return out
    o = np.argsort(ps)
    q = ps[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    r = np.empty(m)
    r[o] = np.minimum(q, 1.0)
    out[ok] = r
    return out
