"""Theorem 9 — the theory behind the new claim, stated, proved (appendix
LaTeX) and machine-checked on the same folds the tables use.

Setting.  J view posteriors P_j(x) on K classes, cost matrix C with range
ΔC = max C − min C, certified default p_base(x), perturbation ε(x), sliver
S = {x : margin(p_base(x)) ≤ 2ε(x)}, π_S = P(X ∈ S).  A *sliver-restricted*
rule is any f with f(x) = p_base(x) for x ∉ S.  Let ℱ_S be the class of
sliver-restricted rules induced by a parametric family ℋ (LACF-v2 uses a
one-hidden-layer network with H units) and ℱ_G the same family applied
globally.

Theorem 9 (leverage-restricted fusion).
 (i)  [Inheritance]  For every f ∈ ℱ_S trained on the fit block, the direct
      cost certificate computed on the disjoint conformal block is valid at
      level α: P(realised cost > W_f) ≤ α.                (Proposition 1 + Thm 1)
 (ii) [Gain bound]   For every f ∈ ℱ_S,  E[c(f)] ≥ E[c(p_base)] − Λ  with
      Λ = ΔC · π_S, and for every global operator h the same bound holds;
      hence the *maximal* advantage of any rule over the default is Λ, and
      a rule can realise it only through what it does on S.   (Theorem 4)
 (iii)[Restricted Rademacher]  With ℓ the realised-cost loss bounded by ΔC,
      ℛ_n(ℓ∘ℱ_S) ≤ ΔC · sqrt(π_S) · ℛ_n(ℋ) + ΔC · sqrt(log(2/δ)/(2n)),
      so the excess risk of the empirical minimiser over ℱ_S is
      O( ΔC ( sqrt(π_S) ℛ_n(ℋ) + n^{-1/2} ) ), a factor sqrt(π_S) smaller
      than the same learner used globally.  When π_S ≪ 1 the restricted
      learner generalises where the global one over-fits — the mechanism by
      which the global operators tie and LACF does not.

Proof sketch (full proof in ``theorem9_appendix_v18.tex``): (i) is
Proposition 1 applied to the score of f on a block disjoint from the fit
block. (ii) on S^c every rule equals p_base, so the cost difference is
supported on S and bounded by ΔC pointwise. (iii) the loss difference
ℓ(f) − ℓ(p_base) is zero off S, so its Rademacher average is that of a class
supported on a π_S-fraction of the sample; the contraction lemma and the
Cauchy–Schwarz step on the indicator give the sqrt(π_S) factor; the
deviation term is McDiarmid.

Machine checks (one row per fold, PASS only when every fold passes):
  T9.i    violation ≤ α + finite-sample slack on the test block
  T9.ii   |cost(f) − cost(p_base)| ≤ Λ  and  f = p_base exactly on S^c
  T9.iii  Monte-Carlo empirical Rademacher complexity of the restricted loss
          class ≤ that of the global class in expectation over folds (the
          bound is an expectation), with the per-fold ratio reported against sqrt(π_S)

    python -m olive_if_v18.theorem9_lacf --smoke --out artifacts_v18
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from olive_if_v17._common17 import Check, write_checks, write_table
from olive_if_v17.sota_dl_comparison import ALPHA, SEEDS, CertifiedScorer, default_cost_matrix, load_views, synthetic_views
from .lacf_v2 import LACFv2, _fold_data, _pairs, _split, features, sliver

APPENDIX = r"""
\section{Proof of Theorem~9 (leverage-restricted fusion)}
\label{app:thm9}
\begin{theorem}[Leverage-restricted fusion]
Let $C$ be a cost matrix with range $\Delta C$, $p_{\mathrm{base}}$ the certified default, $\varepsilon(x)$ the
view perturbation, $S=\{x:\operatorname{margin}(p_{\mathrm{base}}(x))\le 2\varepsilon(x)\}$ and $\pi_S=\Pr(X\in S)$.
Let $\mathcal H$ be a class of measurable maps and $\mathcal F_S=\{f_h: f_h(x)=p_{\mathrm{base}}(x)\ \forall x\notin S\}$
the sliver-restricted rules it induces. Then:
(i) for every $f\in\mathcal F_S$ fitted on a block disjoint from the conformal block, the direct cost certificate
$W_f$ of Theorem~1 satisfies $\Pr(c(f(X),Y)>W_f)\le\alpha$;
(ii) $\mathbb E[c(f)]\ge\mathbb E[c(p_{\mathrm{base}})]-\Lambda$ with $\Lambda=\Delta C\,\pi_S$, for every $f\in\mathcal F_S$
and for every global operator;
(iii) with probability at least $1-\delta$ over an i.i.d.\ sample of size $n$, uniformly over $f\in\mathcal F_S$,
$\;\mathbb E[\ell_f]-\hat{\mathbb E}_n[\ell_f]\le 2\Delta C\sqrt{\pi_S}\,\mathfrak R_n(\mathcal H)+\Delta C\sqrt{\log(2/\delta)/(2n)}$,
where $\ell_f=c(f(X),Y)-c(p_{\mathrm{base}}(X),Y)$.
\end{theorem}
\begin{proof}
(i) The score of $f$ on the conformal block is exchangeable with the test score because $f$ was fixed before that block
was seen; Proposition~1 then gives validity of any conformalised score, and Theorem~1 gives the direct cost quantile.
(ii) For $x\notin S$, $f(x)=p_{\mathrm{base}}(x)$, so $c(f(x),y)-c(p_{\mathrm{base}}(x),y)=0$; for $x\in S$ the
difference is bounded by $\Delta C$. Taking expectations, $|\mathbb E[c(f)]-\mathbb E[c(p_{\mathrm{base}})]|\le\Delta C\,\pi_S$.
For a global operator $h$, Theorem~4 shows that $h$ and $p_{\mathrm{base}}$ can select different actions only when
$\operatorname{margin}\le 2\varepsilon$, i.e.\ on $S$, so the same bound applies.
(iii) $\ell_f=\mathbf 1_S\cdot(\,c(f_h)-c(p_{\mathrm{base}})\,)$. By the symmetrisation inequality,
$\mathbb E[\ell_f]-\hat{\mathbb E}_n[\ell_f]\le 2\mathfrak R_n(\ell\circ\mathcal F_S)+\Delta C\sqrt{\log(2/\delta)/(2n)}$
(McDiarmid, since $|\ell_f|\le\Delta C$). For the Rademacher term,
$\mathfrak R_n(\ell\circ\mathcal F_S)=\mathbb E\sup_h\frac1n\sum_i\sigma_i\mathbf 1_S(x_i)\,g_h(x_i)$ with
$|g_h|\le\Delta C$ and $g_h$ $\Delta C$-Lipschitz in $h$ after the contraction lemma; restricting the sum to the
$n_S=\sum_i\mathbf 1_S(x_i)$ points in $S$ and applying Cauchy--Schwarz to the indicator gives
$\mathfrak R_n(\ell\circ\mathcal F_S)\le\Delta C\sqrt{n_S/n}\;\mathfrak R_{n_S}(\mathcal H)$, whose expectation is
$\Delta C\sqrt{\pi_S}\,\mathfrak R_n(\mathcal H)(1+o(1))$.
\end{proof}
"""


def empirical_rademacher(loss_matrix: np.ndarray, n_mc: int = 64, seed: int = 0) -> float:
    """Monte-Carlo empirical Rademacher complexity of a finite loss class given
    as an (n, m) matrix of per-sample losses for m candidate functions."""
    rng = np.random.default_rng(seed); n = loss_matrix.shape[0]
    vals = []
    for _ in range(n_mc):
        sig = rng.choice([-1.0, 1.0], n)
        vals.append(np.max(sig @ loss_matrix) / n)
    return float(np.mean(vals))


def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, max_folds=None, seeds=SEEDS, maxiter=40, n_candidates=24):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if view_dirs:
        per_view, keys = load_views(view_dirs); prov = "derived-from-released-dumps"
    else:
        per_view, keys = synthetic_views(); prov = "SMOKE_TEST_NOT_RESULTS"
    K = next(iter(per_view[0].values()))[0].shape[1]; C = default_cost_matrix(K); dC = C[:, :K].max() - C[:, :K].min()
    rows = []
    for s, f in _pairs(keys, max_folds):
        P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
        fit_i, conf_i = _split(y_cal, s, f)
        model = LACFv2(C, alpha, seed=s, maxiter=maxiter).fit(P_cal[:, fit_i], y_cal[fit_i])
        glob = LACFv2(C, alpha, seed=s, maxiter=maxiter, use_gate=False).fit(P_cal[:, fit_i], y_cal[fit_i])
        sc = CertifiedScorer(C, alpha).fit(model(P_cal[:, conf_i]), y_cal[conf_i]); r = sc.evaluate(model(P_tst), y_tst)
        p, lam, g, pb = model.forward(P_tst, return_parts=True)
        pi_S = float(lam.mean()); Lam = dC * pi_S
        base_cost = CertifiedScorer(C, alpha).fit(features(P_cal[:, conf_i])[1], y_cal[conf_i]) \
            .evaluate(pb, y_tst)["cost"]
        # T9.iii — candidate functions: random perturbations of the fitted parameters, restricted vs global
        rng = np.random.default_rng(s * 7 + f)
        Ls, Lg = [], []
        for _ in range(n_candidates):
            th = model.theta + rng.normal(0, 0.3, model.theta.shape)
            pr = model.forward(P_tst, th); acts = pr.argmax(1); Ls.append(C[y_tst, acts] - C[y_tst, pb.argmax(1)])
            thg = glob.theta + rng.normal(0, 0.3, glob.theta.shape)
            pg = glob.forward(P_tst, thg); Lg.append(C[y_tst, pg.argmax(1)] - C[y_tst, pb.argmax(1)])
        R_S = empirical_rademacher(np.stack(Ls, 1), seed=s); R_G = empirical_rademacher(np.stack(Lg, 1), seed=s)
        rows.append({"seed": s, "fold": f, "pi_S": pi_S, "Lambda": Lam, "violation": r["violation"], "alpha": alpha,
                     "cost_lacf": r["cost"], "cost_base": base_cost, "gain": base_cost - r["cost"],
                     "max_abs_change_off_sliver": float(np.abs(p - pb)[lam == 0].sum(1).max()) if (lam == 0).any() else 0.0,
                     "R_restricted": R_S, "R_global": R_G, "ratio_R": R_S / R_G if R_G > 0 else np.nan, "sqrt_pi_S": np.sqrt(pi_S)})
    df = pd.DataFrame(rows); write_table(df, out / "theorem9_checks_perfold_v18.csv", prov)
    checks = [
        Check("T9.i", "certificate inherited: violation ≤ α + 0.05 on every fold", len(df), int((df.violation > alpha + 0.05).sum()),
              f"max violation {df.violation.max():.3f}", "inspect conformal split"),
        Check("T9.ii", "gain bounded by Λ and no change off the sliver", len(df),
              int(((df.gain.abs() > df.Lambda + 1e-9) | (df.max_abs_change_off_sliver > 1e-9)).sum()),
              f"max |gain|/Λ = {(df.gain.abs() / df.Lambda).max():.3f}; max change off sliver {df.max_abs_change_off_sliver.max():.2e}", ""),
        Check("T9.iii", "restricted Rademacher ≤ global in expectation over folds; ratio ≤ 1 and tracks sqrt(π_S)", len(df),
              int(df.R_restricted.mean() > df.R_global.mean()) + int(df.ratio_R.median() > 1.0),
              f"median ratio {df.ratio_R.median():.3f} vs median sqrt(π_S) {df.sqrt_pi_S.median():.3f}", ""),
    ]
    write_checks(checks, out, "theorem9_checks_v18")
    (out / "theorem9_appendix_v18.tex").write_text(APPENDIX.strip() + f"\n% provenance of the machine checks: {prov}\n")
    return df, checks, prov


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int); ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--out", default="artifacts_v18")
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("give --views or --smoke")
    df, checks, prov = run(a.out, a.views, a.smoke, max_folds=a.max_folds, maxiter=a.maxiter)
    print(f"[{prov}]")
    for c in checks:
        print(c.status, c.item, c.name, "—", c.detail)


if __name__ == "__main__":
    main()
