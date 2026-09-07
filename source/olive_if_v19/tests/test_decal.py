import numpy as np
from olive_if_v19.decal import DECAL, decision_gap, proj_simplex, plug_in_action
from olive_if_v17.sota_dl_comparison import default_cost_matrix

def _data(n=4000, K=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    lg = rng.normal(0, 1, (n, K)); lg[np.arange(n), y] += 1.5
    lg[:, 0] += 0.8                                   # systematic over-confidence in 'healthy' -> costly misses
    e = np.exp(lg); p = e / e.sum(1, keepdims=True)
    return p, y

def test_projection_is_on_simplex():
    v = np.random.default_rng(1).normal(size=(50, 5))
    q = proj_simplex(v); assert np.allclose(q.sum(1), 1) and (q >= 0).all()

def test_gap_falls_on_held_out():
    C = default_cost_matrix(5); p, y = _data()
    h = len(y) // 2
    dc = DECAL(C).fit(p[:h], y[:h])
    assert dc.gap_after <= dc.gap_before
    q = dc(p[h:])
    assert decision_gap(q, y[h:], C) < decision_gap(p[h:], y[h:], C)
    # Proposition D1 bounds the *regret* by 2*Gap; it does not promise a lower realised cost on
    # every sample, so cost is reported by the harness, not asserted here.

def test_identity_when_already_decision_calibrated():
    C = default_cost_matrix(5); rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(5) * 2, 6000); y = np.array([rng.choice(5, p=r) for r in p])   # exactly calibrated
    dc = DECAL(C, tol=0.05).fit(p, y)
    assert np.abs(dc(p) - p).max() < 0.06
