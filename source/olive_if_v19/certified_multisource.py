"""CMS -- Certified Multi-Source decision: the proposed decision layer fed by
all released sources at once, with the certificate and the priced referral it
already carries.

The lever, and why the leverage bound does not forbid it
-------------------------------------------------------
Theorem 4 bounds what a change of *combination operator* is worth for a fixed
evidence set.  It says nothing about *adding evidence*.  Every EUR/tree number
in Sections 5-6 feeds Equation (1) a single posterior (the deployed B0 run),
while the cheapest arms of Section 7.1.1 are p-value merges over ten sources.
The gap between them (EUR 0.607 against 0.577) is therefore not an operator
effect but a *source-count* effect, and it is the one lever on the released
dumps that is (i) label-free beyond the existing blocks, (ii) larger than the
design's MDE, and (iii) fully compatible with the certificate.

The method
----------
  sources        J released view posteriors P_j(x) (the eight backbone dumps,
                 or the ten training-method variants)
  selection      sources are ordered by single-source plug-in cost on the FIT
                 half and the count J* is the forward-selection minimiser of
                 fit-half cost (a discrete choice, no continuous parameter);
                 the conformal half and the test fold never inform it
  fusion         p_J(x) = product rule (geometric mean, normalised) -- the
                 independent-evidence Bayes combination; sum rule as control
  optional       DECAL action-cell decision calibration of p_J on the fit half
  decision       Equation (1) at lambda = 0: a(x) = argmin_a sum_y p_J(y|x) C[y,a]
  referral       VOI ranking under the matched automation budget
  certificate    LAC quantile + direct cost quantile (Theorem 1) on the
                 disjoint conformal half, from p_J -- inherited (Prop. 1)

Comparators, all through the identical CertifiedScorer at matched automation:
  * single-source Eq. (1) on each view alone (the paper's deployed arm is B0)
  * p-value merging over the J sources (Vovk-Wang mean, Bonferroni,
    Fisher-as-invalid-reference), read as a decision via the normalised
    merged p-vector -- the Section 7.1.1 arms
  * every belief-function / deep-fusion operator of olive_if_v17 on the same
    J posteriors (sum, TMC, ETMC, EDL, PDF, CML, DynMM, gated mixture)

Two statements checked in tests/test_cms.py:
  (E1) Exactness. Conformalising the fused posterior's own LAC score on the
       disjoint half gives coverage exactly at the order statistic
       ceil((n+1)(1-alpha))/(n+1), whereas the mean / Bonferroni merges are
       valid only up to a dependence factor and are conservative when the
       source p-values are comonotone; on comonotone sources the CMS set is
       never larger than the merged set at equal alpha.
  (E2) Inheritance. The CMS certificate is a direct cost quantile on a block
       disjoint from every fitted quantity, so its violation rate on the
       automated set is <= alpha in expectation for any J.

A source-count curve (J = 1..J_max, views added in order of single-source
clean cost) is written so the reader can see where the gain saturates, with
the paired per-fold MDE beside every point.  Smoke runs are stamped
SMOKE_TEST_NOT_RESULTS and are never a result.

    python -m olive_if_v19.certified_multisource --views runs/b0 runs/b1 ... --out artifacts_v19
    python -m olive_if_v19.certified_multisource --smoke --out artifacts_v19
"""
from __future__ import annotations

import argparse
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from olive_if_v17._common17 import fmt3, write_table
from olive_if_v17.sota_dl_comparison import (ALPHA, SEEDS, OPERATORS, GatedMixture, CertifiedScorer,
                                             default_cost_matrix, load_views, synthetic_views, _norm,
                                             sum_rule, product_rule)
from olive_if_v19.decal import DECAL, _mde

EPS = 1e-12


# ------------------------------------------------------------------ p-value merging comparators
def conformal_pvalues(p_cal, y_cal, p):
    """Per-class LAC conformal p-values for every row of p, from one source's calibration half."""
    s_cal = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    s = 1 - p                                                        # (n, K)
    return (1 + (s_cal[None, None, :] >= s[..., None]).sum(-1)) / (len(s_cal) + 1)


def merge_mean(Pv):    return np.minimum(1.0, 2.0 * Pv.mean(0))     # Vovk-Wang arithmetic mean, factor 2
def merge_bonf(Pv):    return np.minimum(1.0, Pv.shape[0] * Pv.min(0))
def merge_fisher(Pv):  # valid only under independence: kept as the paper's "invalid reference"
    stat = -2 * np.log(np.clip(Pv, EPS, None)).sum(0)
    return stats.chi2.sf(stat, 2 * Pv.shape[0])
def merge_mean_raw(Pv):  return Pv.mean(0)                       # same rule, validity factor removed
def merge_min_raw(Pv):   return Pv.min(0)                        # same rule, validity factor removed
MERGES = {"merge: Vovk-Wang mean": merge_mean, "merge: Bonferroni": merge_bonf,
          "merge: Fisher (invalid ref.)": merge_fisher,
          # Controls. The merged p-vector is renormalised into a pseudo-posterior, under which a
          # constant multiplier is a no-op; the validity factors therefore reach the decision only
          # where they saturate at 1. These two arms are the same rules with the factor removed, so
          # the difference between each pair isolates what the saturation costs.
          "merge: mean, factor removed (control)": merge_mean_raw,
          "merge: min, factor removed (control)": merge_min_raw}


def merged_pseudo_posterior(Pv_merged):
    return _norm(Pv_merged)


# ------------------------------------------------------------------ harness
def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, seeds=SEEDS, max_folds=None, with_decal=True):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if view_dirs:
        per_view, keys = load_views(view_dirs); prov = "derived-from-released-dumps"
        names = [pathlib.Path(d).name for d in view_dirs]
    else:
        per_view, keys = synthetic_views(J=6); prov = "SMOKE_TEST_NOT_RESULTS"; names = [f"view{j}" for j in range(len(per_view))]
    J = len(per_view); K = next(iter(per_view[0].values()))[0].shape[1]; C = default_cost_matrix(K)
    rows = []
    pairs = sorted({(s, f) for (_, s, f) in keys if (("cal", s, f) in keys and ("test", s, f) in keys)})
    if max_folds:
        pairs = [q for q in pairs if q[1] < max_folds]
    for s, f in pairs:
        P_cal = np.stack([v[("cal", s, f)][0] for v in per_view]); y_cal = per_view[0][("cal", s, f)][1]
        P_tst = np.stack([v[("test", s, f)][0] for v in per_view]); y_tst = per_view[0][("test", s, f)][1]
        rng = np.random.default_rng(s * 1000 + f); idx = rng.permutation(len(y_cal)); h = len(idx) // 2
        fi, ci = idx[:h], idx[h:]

        def arm(name, p_fit, p_conf, p_tst, group, J_used, decal=False):
            if decal:
                dc = DECAL(C).fit(p_fit, y_cal[fi]); p_conf, p_tst = dc(p_conf), dc(p_tst)
            sc = CertifiedScorer(C, alpha).fit(p_conf, y_cal[ci]); r = sc.evaluate(p_tst, y_tst)
            r.update({"method": name, "group": group, "J": J_used, "seed": s, "fold": f}); rows.append(r)

        # single-source arms (the deployed system is view 0 = B0)
        for j in range(J):
            arm(f"single: {names[j]}", P_cal[j, fi], P_cal[j, ci], P_tst[j], "single", 1)
        # source order and source count are chosen on the FIT half only (plug-in realised cost, no referral),
        # so the conformal half and the test fold never inform the selection
        # Select on the SAME objective the table reports: the certified decision path with the
        # referral budget on. The earlier version minimised a plug-in cost with no referral, so
        # selection optimised a quantity that omits the layer which dominates the reported cost.
        _sel_scorer = CertifiedScorer(C, alpha)
        def fit_cost(p):
            sc = _sel_scorer.fit(p, y_cal[fi])
            return float(sc.evaluate(p, y_cal[fi])["cost"])
        order = np.argsort([fit_cost(P_cal[j, fi]) for j in range(J)])
        curve_fit = {1: fit_cost(P_cal[order[0], fi])}
        for m in range(2, J + 1):
            sel = order[:m]
            curve_fit[m] = fit_cost(product_rule(P_cal[sel][:, fi]))
            for fname, op in (("CMS (product)", product_rule), ("CMS (sum)", sum_rule)):
                arm(f"{fname} J={m}", op(P_cal[sel][:, fi]), op(P_cal[sel][:, ci]), op(P_tst[sel]), fname, m)
        J_star = min(curve_fit, key=curve_fit.get)                   # forward-selected source count
        sel = order[:J_star]
        arm("CMS (product) J*", product_rule(P_cal[sel][:, fi]) if J_star > 1 else P_cal[sel[0], fi],
            product_rule(P_cal[sel][:, ci]) if J_star > 1 else P_cal[sel[0], ci],
            product_rule(P_tst[sel]) if J_star > 1 else P_tst[sel[0]], "CMS (selected)", J_star)
        # full set: every operator, merges, and DECAL on top of the proposed
        for oname, op in OPERATORS.items():
            # product_rule and sum_rule are already run as the CMS arms above.
            # qmf is skipped because on posterior-only evidence its free-energy weight is
            # identically zero, making it bit-identical to sum_rule (Supp. S24); running it
            # would print the same numbers twice under a second citation.
            if oname in ("product_rule", "sum_rule", "qmf"):
                continue
            arm(f"op: {oname}", op(P_cal[:, fi]), op(P_cal[:, ci]), op(P_tst), "operator", J)
        gm = GatedMixture().fit(P_cal[:, fi], y_cal[fi])
        arm("op: gated_mixture", gm(P_cal[:, fi]), gm(P_cal[:, ci]), gm(P_tst), "operator", J)
        # p-value merges: per-source p-values are calibrated on the fit half so the conformal half stays disjoint
        Pv_conf = np.stack([conformal_pvalues(P_cal[j, fi], y_cal[fi], P_cal[j, ci]) for j in range(J)])
        Pv_tst = np.stack([conformal_pvalues(P_cal[j, fi], y_cal[fi], P_tst[j]) for j in range(J)])
        Pv_fit = np.stack([conformal_pvalues(P_cal[j, fi], y_cal[fi], P_cal[j, fi]) for j in range(J)])
        for mname, mg in MERGES.items():
            arm(mname, merged_pseudo_posterior(mg(Pv_fit)), merged_pseudo_posterior(mg(Pv_conf)),
                merged_pseudo_posterior(mg(Pv_tst)), "merge", J)
        if with_decal:
            arm(f"CMS (product) J={J} + DECAL", product_rule(P_cal[:, fi]), product_rule(P_cal[:, ci]), product_rule(P_tst),
                "CMS+DECAL", J, decal=True)
    per = pd.DataFrame(rows)
    write_table(per, out / "cms_perfold_v19.csv", prov)
    _write_manifest_v19(out, view_dirs, names, alpha, prov)
    return summarise(per, out, prov, names, J)


def _write_manifest_v19(out_dir, view_dirs, names, alpha, prov):
    """Record what this run consumed. The VIEW ORDER is load-bearing: it fixes the
    forward-selection sequence, so it is written out explicitly."""
    import datetime, hashlib, json, os, platform, socket
    def digest(run):
        p = pathlib.Path(run); preds = p / "preds" if (p / "preds").is_dir() else p
        h = hashlib.sha256(); files = sorted(preds.glob("*.npz"))
        for f in files:
            st = f.stat(); h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}|".encode())
        return h.hexdigest()[:16], len(files)
    def ver(m):
        try: return __import__(m).__version__
        except Exception: return None
    rows = []
    for v in (view_dirs or []):
        d, n = digest(v)
        rows.append({"path": os.path.abspath(v), "view_digest": d, "n_dumps": n})
    man = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "host": socket.gethostname(), "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": ver("numpy"), "scipy": ver("scipy"),
           "pandas": ver("pandas"), "argv": sys.argv, "alpha": alpha, "provenance": prov,
           "view_order": list(names), "views": rows, "n_views": len(rows),
           "note": ("Costs are in the harness cost matrix's own units, not EUR/tree. "
                    "view_order fixes the forward-selection sequence and therefore J*.")}
    try:
        p = pathlib.Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "run_manifest_v19.json").write_text(json.dumps(man, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
        print(f"[manifest] {p / 'run_manifest_v19.json'}  views {len(rows)}")
    except Exception as e:                                            # noqa: BLE001
        print(f"[manifest] not written (ignored): {type(e).__name__}: {e}")


# --- screening patch v19 ---
def summarise(per, out, prov, names, J):
    # The reference is the CHEAPEST single source, which is what the manuscript compares
    # against. It is an in-sample argmin over the same folds, so it is an oracle: a
    # practitioner cannot know which source is cheapest without labels. Two further
    # baselines are emitted beside it because they need no selection at all --
    # `gain_vs_mean_single` (the source you get by picking one arbitrarily) and
    # `gain_vs_loo_single` (choosing the cheapest source on the other nine folds).
    _sing = per[per.group == "single"].groupby("method")["cost"].mean()
    ref_name = _sing.idxmin()
    prop_name = "CMS (product) J*"
    metrics = ["cost", "bound_W", "violation", "missed_disease", "set_size", "coverage", "automation"]
    g = per.groupby("method")
    tab = pd.DataFrame({m: g[m].mean() for m in metrics}).join(pd.DataFrame({m + "_sd": g[m].std() for m in metrics}))
    tab["group"] = g["group"].first(); tab["J"] = g["J"].first()
    ref = per[per.method == ref_name].set_index(["seed", "fold"])["cost"]
    prop = per[per.method == prop_name].set_index(["seed", "fold"])["cost"]
    praw_ref, praw_prop, gain_ref, gain_prop, mde = {}, {}, {}, {}, {}
    for m in tab.index:
        A = per[per.method == m].set_index(["seed", "fold"])["cost"].reindex(ref.index)
        d = ref - A; gain_ref[m] = float(d.mean()); mde[m] = _mde(d)
        praw_ref[m] = np.nan if m == ref_name else (stats.wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0)
        d2 = A - prop; gain_prop[m] = float(d2.mean())                    # positive: proposed cheaper than m
        praw_prop[m] = np.nan if m == prop_name else (stats.wilcoxon(d2, alternative="greater").pvalue if np.any(d2 != 0) else 1.0)
    def holm(pr):
        p = pd.Series(pr).dropna().sort_values()
        return pd.Series({k: min(1.0, v * (len(p) - i)) for i, (k, v) in enumerate(p.items())}).cummax()
    tab["gain_vs_single"] = pd.Series(gain_ref); tab["mde"] = pd.Series(mde)
    tab["p_holm_vs_single"] = holm(praw_ref).reindex(tab.index)
    tab["proposed_minus_this"] = pd.Series(gain_prop); tab["p_holm_proposed_cheaper"] = holm(praw_prop).reindex(tab.index)
    tab["resolvable_vs_single"] = tab["gain_vs_single"].abs() >= tab["mde"]
    tab.loc[ref_name, ["gain_vs_single", "mde"]] = np.nan              # a row is not resolvable against itself
    tab.loc[ref_name, "resolvable_vs_single"] = False
    # --- selection-free baselines -------------------------------------------------
    wide = per[per.group == "single"].pivot_table(index=["seed", "fold"], columns="method", values="cost")
    mean_single = wide.mean(axis=1).reindex(ref.index)
    loo = []
    for k in range(len(wide)):
        rest = wide.drop(wide.index[k])
        loo.append(wide.iloc[k][rest.mean().idxmin()])
    loo_single = pd.Series(loo, index=wide.index).reindex(ref.index)
    g_mean, g_loo, m_mean, m_loo = {}, {}, {}, {}
    for m in tab.index:
        A = per[per.method == m].set_index(["seed", "fold"])["cost"].reindex(ref.index)
        d1 = mean_single - A; g_mean[m] = float(d1.mean()); m_mean[m] = _mde(d1)
        d2 = loo_single - A; g_loo[m] = float(d2.mean()); m_loo[m] = _mde(d2)
    tab["gain_vs_mean_single"] = pd.Series(g_mean); tab["mde_vs_mean_single"] = pd.Series(m_mean)
    tab["gain_vs_loo_single"] = pd.Series(g_loo); tab["mde_vs_loo_single"] = pd.Series(m_loo)
    tab["ref_arm"] = ref_name
    tab["mean_single_cost"] = float(mean_single.mean())
    tab["loo_single_cost"] = float(loo_single.mean())
    tab["is_proposed"] = tab.index == prop_name
    others = tab.drop(prop_name)
    tab["beats_all_comparators"] = bool(((others["proposed_minus_this"] > 0) & (others["p_holm_proposed_cheaper"] < 0.05)).all())
    tab = tab.reset_index().rename(columns={"index": "method"}).sort_values(["group", "J", "cost"])
    write_table(tab, out / "cms_summary_v19.csv", prov)
    curve = tab[tab.group.isin(["single", "CMS (product)", "CMS (sum)", "CMS (selected)"])][["method", "group", "J", "cost", "cost_sd", "bound_W", "violation", "gain_vs_single", "mde"]]
    write_table(curve, out / "cms_source_curve_v19.csv", prov)
    _latex(tab, out / "cms_table_v19.tex", prov, ref_name, prop_name)
    return tab


def _latex(tab, path, prov, ref_name, prop_name):
    L = ["\\begin{table*}[t]\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         "Arm & $J$ & Cost (harness units) & Bound $W$ & Violation & Missed disease & Gain vs.\\ single [MDE] & $p_{\\mathrm{Holm}}$\\\\", "\\midrule"]
    show = tab[~tab.method.str.startswith("single:") | (tab.method == ref_name)]
    for _, r in show.iterrows():
        name = r["method"].replace("_", "\\_")
        if r["is_proposed"]: name = "\\textbf{" + name + " (proposed)}"
        gainc = "---" if r["method"] == ref_name else f"{fmt3(r['gain_vs_single'])} [{fmt3(r['mde'])}]"
        pc = "---" if r["method"] == ref_name else fmt3(r["p_holm_vs_single"])
        L.append(f"{name} & {int(r['J'])} & {fmt3(r['cost'])} $\\pm$ {fmt3(r['cost_sd'])} & {fmt3(r['bound_W'])} & {fmt3(r['violation'])} & "
                 f"{fmt3(r['missed_disease'])} & {gainc} & {pc}\\\\")
    beats = bool(tab["beats_all_comparators"].iloc[0])
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{The proposed decision layer fed by $J$ released sources (CMS) against the deployed single-source "
          "system, every fusion operator on the same sources, and p-value merging, all through the identical certified "
          "decision path at matched automation and $\\alpha=0.10$. Gain is the paired per-fold cost difference against "
          "the single-source arm (positive: cheaper), with the design's MDE; Holm-corrected one-sided Wilcoxon. "
          f"Improvement gate (proposed strictly cheaper than every comparator at Holm $p<0.05$): {'met' if beats else 'NOT met -- reported as a tie'}. Provenance: {prov}.}}",
          "\\label{tab:cms}", "\\end{table*}"]
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="artifacts_v19"); ap.add_argument("--max-folds", type=int)
    ap.add_argument("--no-decal", action="store_true")
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("pass --views <dump dirs> or --smoke")
    tab = run(a.out, a.views, a.smoke, max_folds=a.max_folds, with_decal=not a.no_decal)
    print(tab[["method", "J", "cost", "bound_W", "violation", "set_size", "gain_vs_single", "mde", "p_holm_vs_single"]].to_string())


if __name__ == "__main__":
    main()
