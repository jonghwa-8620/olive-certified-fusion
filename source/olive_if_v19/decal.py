"""DECAL -- Decision-Calibrated Evidence fusion for a priced, certified action.

Why this is the lever the leverage bound leaves open
----------------------------------------------------
Theorem 4 bounds what a change of *combination operator* can be worth by
Lambda = DeltaC * P(margin(X) <= 2 eps(X)); on the olive corpus that budget is
spent (Sec. 5.2.1, 7.1.2).  Every rule in the comparison, however, consumes the
same *temperature-scaled* posterior and takes the plug-in action
a(x) = argmin_a J_hat(a|x), J_hat(a|x) = sum_y p_hat(y|x) C[y,a].  Temperature
scaling minimises a proper score; it does not make the expected costs J_hat
*decision-calibrated* [zhao2021decision]: E[C[Y,a] | a(X) = a] can differ from
E[J_hat(a|X) | a(X) = a], and under an asymmetric matrix (miss 1.6/7.2/21.2
against spray 1.0) that residual is what the plug-in action and the VOI
referral ranking are both wrong by.  The lever is therefore not the operator
but the *scale on which the operator's output is read into the cost matrix*.

The method
----------
Inputs: any fused posterior p_hat(x) (an operator's output), the cost matrix
C (K x (K+1), last column = referral price), the fit half of the calibration
block (never the conformal half, so Proposition 1 applies unchanged).

  1. Action-cell recalibration (decision calibration, Alg. 1 of
     [zhao2021decision], restricted to the cost class L_C = {C[., a]}):
        repeat until max_cell ||pi_cell - mean_cell p||_1 <= tol or T rounds
          cell(x) = plug-in action under the current p_tilde  (K cells: the referral action is not a cell, so the referral ranking is not recalibrated)
          for each cell c with n_c >= n_min:
              delta_c = pi_c - mean_{x in c} p_tilde(x)   (pi_c = empirical
              class frequency in the cell)
          p_tilde(x) <- proj_simplex( p_tilde(x) + delta_cell(x) )
     The map has at most T*K*K parameters, is piecewise-constant in the
     action cell, and is fitted on the fit half only.
  2. Certified action: J_tilde(a|x) = sum_y p_tilde(y|x) C[y,a]; the
     automated action is argmin over the K label actions; VOI(x) =
     min_a J_tilde(a|x) - C[.,K] ranks referrals under the same budget as
     every other arm (matched automation).
  3. Certificate: LAC quantile and the direct cost quantile of Theorem 1 are
     taken on the disjoint conformal half from p_tilde, so validity is
     inherited (Proposition 1) and the bound is comparable row-for-row.

Two quantities are stated *before* the comparison, in the style of the
leverage index:
  * the decision-calibration gap of the base posterior,
        Gap_dc = sum_a P(a(X)=a) * | E[C[Y,a] | a(X)=a] - E[J_hat(a|X) | a(X)=a] |,
    estimated on the fit half; the inheritance argument of Sec. 7.1.3 bounds the plug-in regret of
    any rule by 2 * Gap_dc, so a rule whose Gap_dc is below the design's MDE
    cannot be improved by DECAL and is predicted to tie;
  * the post-recalibration gap, which that same argument bounds by
    O((K+1) K sqrt(log(K/delta) / n_fit)) w.p. 1-delta, i.e. below the MDE at
    n_fit = 167 only for coarse cells -- which is why the cell partition is by
    action (K cells: the referral action is not a cell, so the referral ranking is not recalibrated) and not by (action x VOI-decile).

Everything is reported with the per-fold paired design, Holm-corrected
Wilcoxon tests against the base arm, the MDE, and a provenance stamp.  A
smoke run (synthetic views) is stamped SMOKE_TEST_NOT_RESULTS and is never a
result.

    python -m olive_if_v19.decal --views runs/b0 ... --out artifacts_v19
    python -m olive_if_v19.decal --smoke --out artifacts_v19
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from olive_if_v17._common17 import fmt3, write_table
from olive_if_v17.sota_dl_comparison import (ALPHA, SEEDS, OPERATORS, CertifiedScorer, default_cost_matrix,
                                             load_views, synthetic_views, _norm)

EPS = 1e-12
BASES = ("sum_rule", "product_rule", "entropy_weighted", "tmc", "etmc", "edl", "pdf", "cml", "dynmm")


def proj_simplex(v):
    """Euclidean projection of each row of v onto the probability simplex."""
    n, K = v.shape
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, K + 1)
    cond = u - css / ind > 0
    rho = cond.sum(1)
    theta = css[np.arange(n), rho - 1] / rho
    return np.clip(v - theta[:, None], EPS, None) / np.clip(v - theta[:, None], EPS, None).sum(1, keepdims=True)


def plug_in_action(p, C):
    K = C.shape[0]
    return (p @ C[:, :K]).argmin(1)


def decision_gap(p, y, C):
    """Gap_dc of a posterior on labelled data: action-cell weighted |realised - predicted| expected cost."""
    K = C.shape[0]
    a = plug_in_action(p, C)
    J = p @ C[:, :K]
    gap = 0.0
    for c in range(K):
        m = a == c
        if m.any():
            gap += m.mean() * abs(C[y[m], c].mean() - J[m, c].mean())
    return float(gap)


class DECAL:
    """Action-cell decision calibration wrapped around any base posterior."""

    def __init__(self, cost, rounds=5, tol=1e-3, n_min=8):
        self.C, self.rounds, self.tol, self.n_min = cost, rounds, tol, n_min
        self.deltas = []                                     # one (K+1, K) table per round

    def fit(self, p_fit, y_fit):
        K = self.C.shape[0]
        p = np.asarray(p_fit, float).copy()
        Y = np.eye(K)[y_fit]
        self.gap_before = decision_gap(p, y_fit, self.C)
        for _ in range(self.rounds):
            a = plug_in_action(p, self.C)
            delta = np.zeros((K, K))
            worst = 0.0
            for c in range(K):
                m = a == c
                if m.sum() >= self.n_min:
                    d = Y[m].mean(0) - p[m].mean(0)
                    delta[c] = d
                    worst = max(worst, np.abs(d).sum())
            self.deltas.append(delta)
            p = proj_simplex(p + delta[a])
            if worst <= self.tol:
                break
        self.gap_after = decision_gap(p, y_fit, self.C)
        return self

    def __call__(self, p):
        p = np.asarray(p, float).copy()
        for delta in self.deltas:
            a = plug_in_action(p, self.C)
            p = proj_simplex(p + delta[a])
        return _norm(p)


def _mde(diff):
    diff = np.asarray(diff, float)
    return float(2.802 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan


def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, seeds=SEEDS, max_folds=None, bases=BASES):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if view_dirs:
        per_view, keys = load_views(view_dirs); prov = "derived-from-released-dumps"
    else:
        per_view, keys = synthetic_views(); prov = "SMOKE_TEST_NOT_RESULTS"
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    rows = []
    pairs = sorted({(s, f) for (_, s, f) in keys if (("cal", s, f) in keys and ("test", s, f) in keys)})
    if max_folds:
        pairs = [q for q in pairs if q[1] < max_folds]
    for s, f in pairs:
        P_cal = np.stack([v[("cal", s, f)][0] for v in per_view]); y_cal = per_view[0][("cal", s, f)][1]
        P_tst = np.stack([v[("test", s, f)][0] for v in per_view]); y_tst = per_view[0][("test", s, f)][1]
        rng = np.random.default_rng(s * 1000 + f); idx = rng.permutation(len(y_cal)); h = len(idx) // 2
        fit_i, conf_i = idx[:h], idx[h:]
        for base in bases:
            op = OPERATORS[base]
            p_fit, p_conf, p_tst = op(P_cal[:, fit_i]), op(P_cal[:, conf_i]), op(P_tst)
            # base arm
            sc = CertifiedScorer(C, alpha).fit(p_conf, y_cal[conf_i])
            r = sc.evaluate(p_tst, y_tst)
            r.update({"method": base, "base": base, "decal": False, "seed": s, "fold": f,
                      "gap_dc_fit": decision_gap(p_fit, y_cal[fit_i], C),
                      "gap_dc_test": decision_gap(p_tst, y_tst, C)})
            rows.append(r)
            # DECAL arm: recalibrate on the fit half, certify on the conformal half
            dc = DECAL(C).fit(p_fit, y_cal[fit_i])
            sc2 = CertifiedScorer(C, alpha).fit(dc(p_conf), y_cal[conf_i])
            r2 = sc2.evaluate(dc(p_tst), y_tst)
            r2.update({"method": f"{base}+DECAL", "base": base, "decal": True, "seed": s, "fold": f,
                       "gap_dc_fit": dc.gap_after, "gap_dc_test": decision_gap(dc(p_tst), y_tst, C),
                       "rounds": len(dc.deltas),
                       "action_change_rate": float((plug_in_action(dc(p_tst), C) != plug_in_action(p_tst, C)).mean())})
            rows.append(r2)
    per = pd.DataFrame(rows)
    write_table(per, out / "decal_perfold_v19.csv", prov)
    return summarise(per, out, prov)


def summarise(per, out, prov):
    metrics = ["cost", "bound_W", "violation", "missed_disease", "set_size", "coverage", "automation",
               "gap_dc_fit", "gap_dc_test"]
    g = per.groupby("method")
    tab = pd.DataFrame({m: g[m].mean() for m in metrics}).join(pd.DataFrame({m + "_sd": g[m].std() for m in metrics}))
    tab["base"] = g["base"].first(); tab["decal"] = g["decal"].first()
    # paired test DECAL vs its own base, per (seed, fold)
    praw, mde, gain = {}, {}, {}
    for m in tab.index:
        if not tab.loc[m, "decal"]:
            praw[m] = np.nan; mde[m] = np.nan; gain[m] = np.nan; continue
        b = tab.loc[m, "base"]
        A = per[per.method == m].set_index(["seed", "fold"])["cost"]
        B = per[per.method == b].set_index(["seed", "fold"])["cost"].loc[A.index]
        d = B - A                                              # positive: DECAL cheaper
        gain[m] = float(d.mean()); mde[m] = _mde(d)
        praw[m] = stats.wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    p = pd.Series(praw).dropna().sort_values()
    holm = pd.Series({m: min(1.0, pv * (len(p) - i)) for i, (m, pv) in enumerate(p.items())}).cummax()
    tab["gain_vs_base"] = pd.Series(gain); tab["mde"] = pd.Series(mde)
    tab["p_holm_vs_base"] = holm.reindex(tab.index)
    tab["resolvable"] = tab["gain_vs_base"].abs() >= tab["mde"]
    tab["predicted_to_matter"] = tab["gap_dc_fit"] > tab["mde"]      # Prop. D1 pre-check on the base arm
    for m in tab.index:
        if tab.loc[m, "decal"]:
            tab.loc[m, "predicted_to_matter"] = tab.loc[tab.loc[m, "base"], "gap_dc_fit"] > tab.loc[m, "mde"]
    tab = tab.sort_values(["base", "decal"]).reset_index().rename(columns={"index": "method"})
    write_table(tab, out / "decal_summary_v19.csv", prov)
    _latex(tab, out / "decal_table_v19.tex", prov)
    return tab


def _latex(tab, path, prov):
    L = ["\\begin{table*}[t]\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
         "Base operator & Arm & Cost (EUR/tree) & Bound $W$ & Violation & Missed disease & $\\mathrm{Gap}_{dc}$ (fit) & "
         "Gain vs.\\ base [MDE] & $p_{\\mathrm{Holm}}$\\\\", "\\midrule"]
    for _, r in tab.iterrows():
        arm = "\\textbf{+DECAL}" if r["decal"] else "base"
        gainc = "---" if not r["decal"] else f"{fmt3(r['gain_vs_base'])} [{fmt3(r['mde'])}]"
        pc = "---" if not r["decal"] else fmt3(r["p_holm_vs_base"])
        L.append(f"{r['base'].replace('_', ' ')} & {arm} & {fmt3(r['cost'])} $\\pm$ {fmt3(r['cost_sd'])} & {fmt3(r['bound_W'])} & "
                 f"{fmt3(r['violation'])} & {fmt3(r['missed_disease'])} & {fmt3(r['gap_dc_fit'])} & {gainc} & {pc}\\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Decision calibration (DECAL) wrapped around each fusion operator, at matched automation and "
          "$\\alpha=0.10$; every arm is certified on the same disjoint conformal half. $\\mathrm{Gap}_{dc}$ is the "
          "decision-calibration gap of the arm measured on the fit half (the inheritance argument of Sec. 7.1.3: plug-in regret $\\le 2\\,\\mathrm{Gap}_{dc}$); "
          "a base whose gap is below the MDE is predicted to tie with its DECAL arm. Gain is the paired per-fold cost "
          f"difference (positive: DECAL cheaper), Holm-corrected one-sided Wilcoxon. Provenance: {prov}.}}",
          "\\label{tab:decal}", "\\end{table*}"]
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="artifacts_v19"); ap.add_argument("--max-folds", type=int)
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("pass --views <dump dirs> or --smoke")
    tab = run(a.out, a.views, a.smoke, max_folds=a.max_folds)
    print(tab[["method", "cost", "bound_W", "violation", "gap_dc_fit", "gain_vs_base", "mde", "p_holm_vs_base"]].to_string())


if __name__ == "__main__":
    main()
