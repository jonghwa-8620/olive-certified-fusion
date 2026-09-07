import numpy as np
import pandas as pd
import pytest

from olive_if_v18 import claim_rewrite as cr
from olive_if_v18 import lacf_v2 as L
from olive_if_v18.theorem9_lacf import empirical_rademacher


@pytest.fixture(scope="module")
def data():
    pv, keys = L.synthetic_views(J=3, n=600, folds=1, seeds=(42,))
    P = np.stack([v[("cal", 42, 0)][0] for v in pv]); y = pv[0][("cal", 42, 0)][1]
    return P, y, L.default_cost_matrix(5)


def test_gradient_matches_finite_difference(data):
    P, y, C = data
    m = L.LACFv2(C, maxiter=1); m.fit(P, y)
    th = m.theta.copy(); f = lambda t: m._loss(t, P, y)
    if L.HAVE_AUTOGRAD:
        g = L._grad(f)(th); i = int(np.argmax(np.abs(g))); h = 1e-5
        e = np.zeros_like(th); e[i] = h
        assert abs((f(th + e) - f(th - e)) / (2 * h) - g[i]) < 1e-3 * max(1.0, abs(g[i]))


def test_exact_identity_off_sliver_and_valid_distribution(data):
    P, y, C = data
    m = L.LACFv2(C, maxiter=5).fit(P, y)
    p, lam, g, pb = m.forward(P, return_parts=True)
    assert np.allclose(p[lam == 0], pb[lam == 0], atol=1e-9) and np.allclose(p.sum(1), 1, atol=1e-6)
    assert (lam == 1).any() and (lam == 0).any()


def test_global_ablation_changes_everything(data):
    P, y, C = data
    m = L.LACFv2(C, maxiter=5, use_gate=False).fit(P, y)
    p, lam, g, pb = m.forward(P, return_parts=True)
    assert lam.all()


def test_rademacher_restricted_le_global():
    rng = np.random.default_rng(0); n, m = 400, 20
    G = rng.normal(0, 1, (n, m)); S = G * (rng.random(n) < 0.3)[:, None]
    assert empirical_rademacher(S) <= empirical_rademacher(G)


def test_claim_scan_finds_old_thesis():
    txt = "We show the combination operator is not a lever but a residual. Nine of eleven rules tie once certified."
    df = cr.scan_conflicts(txt, True)
    assert len(df) >= 2
    new, n = cr.apply_conflicts(txt)
    assert n >= 2 and "not a lever but a residual" not in new


def test_claim_blocks_refuse_unsupported(tmp_path):
    tab = pd.DataFrame([{"provenance": "SMOKE_TEST_NOT_RESULTS", "method": "LACF-v2 (ours)", "is_proposed": True,
                         "beats_all_sota": True, "cost": 0.4}])
    tab.to_csv(tmp_path / "lacf_main_v18.csv", index=False)
    blocks, ok, prov = cr.build_blocks(tmp_path)
    assert not ok and "NOT SUPPORTED" in blocks and "matches the strongest" in blocks
