import numpy as np
from olive_if_v19.certified_multisource import conformal_pvalues, merge_mean, merge_bonf
from olive_if_v17.sota_dl_comparison import CertifiedScorer, default_cost_matrix, product_rule

def _views(J=4, n=3000, K=5, seed=0):
    rng = np.random.default_rng(seed); y = rng.integers(0, K, n); shared = rng.normal(0, 1, (n, K)); shared[np.arange(n), y] += 1.2
    P = []
    for j in range(J):
        lg = shared + rng.normal(0, 0.3, (n, K)); e = np.exp(lg); P.append(e / e.sum(1, keepdims=True))
    return np.stack(P), y

def test_E1_exactness_vs_conservative_merging():
    """On comonotone (strongly dependent) sources the merged sets are conservative; the fused-posterior
    set conformalised directly is at the nominal level and never larger on average."""
    P, y = _views(); K = 5; C = default_cost_matrix(K); a = 0.10
    fit, conf, tst = slice(0, 1000), slice(1000, 2000), slice(2000, 3000)
    Pv = np.stack([conformal_pvalues(P[j, fit], y[fit], P[j, tst]) for j in range(P.shape[0])])
    for mg in (merge_mean, merge_bonf):
        merged_set = mg(Pv) > a
        cov_m = merged_set[np.arange(1000), y[tst]].mean(); size_m = merged_set.sum(1).mean()
        sc = CertifiedScorer(C, a).fit(product_rule(P[:, conf]), y[conf])
        S = sc.sets(product_rule(P[:, tst])); cov_c = S[np.arange(1000), y[tst]].mean(); size_c = S.sum(1).mean()
        assert cov_c >= 1 - a - 0.03            # nominal (finite-sample slack)
        assert size_c <= size_m + 1e-9          # never larger than the conservative merge
        assert cov_m >= cov_c - 0.02            # the merge is (over-)covering, i.e. conservative

def test_E2_certificate_inherited_for_any_J():
    P, y = _views(J=6, seed=1); C = default_cost_matrix(5); a = 0.10
    viol = []
    for J in (1, 2, 4, 6):
        p = product_rule(P[:J]); sc = CertifiedScorer(C, a).fit(p[1000:2000], y[1000:2000])
        viol.append(sc.evaluate(p[2000:], y[2000:])["violation"])
    assert max(viol) <= a + 0.03
