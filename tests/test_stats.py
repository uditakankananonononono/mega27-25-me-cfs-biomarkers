import numpy as np, pytest
from ubiomark import stats


def test_hedges_g_matches_closed_form():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, (1, 20)), rng.normal(0, 1, (1, 15))
    g, v = stats.hedges_g(a, b)
    n1, n0 = 20, 15
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2))
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n0) - 9)
    assert g[0] == pytest.approx(J * d)
    assert v[0] == pytest.approx((n1 + n0) / (n1 * n0) + (J * d) ** 2 / (2 * (n1 + n0)))


def test_hedges_g_nan_for_tiny_groups():
    g, v = stats.hedges_g(np.array([[1.0]]), np.array([[0.0, 1.0]]))
    assert np.isnan(g[0]) and np.isnan(v[0])


def test_dl_reduces_to_fixed_effect_when_homogeneous():
    g = np.array([[0.5, 0.5, 0.5]]); v = np.array([[0.1, 0.2, 0.4]])
    r = stats.dersimonian_laird(g, v)
    assert r["tau2"][0] == 0 and r["mu"][0] == pytest.approx(0.5)
    assert r["se"][0] == pytest.approx(1 / np.sqrt(10 + 5 + 2.5))


def test_dl_known_heterogeneous_example():
    g = np.array([[0.0, 1.0]]); v = np.array([[0.1, 0.1]])
    r = stats.dersimonian_laird(g, v)
    # Q = 10*(0.25+0.25)=5, c = 20-200/20=10, tau2=(5-1)/10=0.4
    assert r["Q"][0] == pytest.approx(5.0) and r["tau2"][0] == pytest.approx(0.4)
    assert r["I2"][0] == pytest.approx(0.8) and r["mu"][0] == pytest.approx(0.5)


def test_bh_fdr_monotone_and_known():
    q = stats.bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
