"""LACF-v2 — the falsification attempt of Sec. 7.1.2.

WHAT THIS MODEL WAS BUILT TO SHOW, AND WHAT IT SHOWED. It was written to
establish that a fusion operator spending all of its capacity on the leverage
sliver S(x) = {margin ≤ 2ε} of Theorem 4 — and equal to the certified default
elsewhere — would beat every global operator at equal validity. It does not.
Over six sweeps of its configuration it is never resolvably cheaper than its
own base, while a rule that knew the labels could save 28% of the realised
cost on the same set. That null is the result the manuscript reports; the
claim in the first sentence is NOT made by the paper. See Sec. 7.1.2 and
Supplementary Section S24.

UNITS. Theorem 4 defines margin and ε on expected costs. The first version of
this file computed both on posterior probabilities, with the cost matrix
absent; that defect and its correction are documented in Supp. S24. Run with
--units cost (or LACF_UNITS=cost) for the definition the manuscript uses; the
default "prob" is kept only so the earlier deposited tables reproduce.

Model
-----
  φ(x)      = [max_j P_j, JS-disagreement, margin(p_base), mean entropy, ε(x)]   (J+4)
  h(x)      = tanh(W1 φ + b1)                                     hidden H (default 8)
  a(x)      = softmax(W_a h + b_a)                                view weights   (J)
  δ(x)      = W_r h + b_r                                         residual logits (K)
  g(x)      = σ(w_g·h + b_g)                                      gate
  p_mix(x)  = Σ_j a_j P_j
  p_res(x)  = normalise( p_mix ⊙ exp(τ δ) )                        τ = 0.5, bounded residual
  p(x)      = p_base + 1[x ∈ S] · g(x) · (p_res − p_base)          (exactly p_base off the sliver)

Loss on the fit half of the calibration block:
  L = mean soft-argmin realised cost + β · CVaR_{1−α}(realised cost)   (≥ the certificate width)
      + γ · mean g(x)          (capacity penalty: learn only where it pays)
      + l2 ‖θ‖²
then Mondrian/LAC conformalisation on the disjoint half (Proposition 1 ⇒ the
certificate is inherited).  Optimiser: L-BFGS-B with exact gradients from ``autograd`` (finite
differences if it is absent), ~170 parameters, deterministic per seed.

Experiments written by ``run``:
  lacf_main_v18          LACF-v2 vs twelve global operators, five seeds × folds,
                         Holm-corrected paired Wilcoxon, improvement gate
  lacf_ablation_v18      full / no-gate (global learner) / no-residual /
                         no-pinball / no-capacity-penalty / sliver-oracle
  lacf_missing_view_v18  one view dropped at rate η ∈ {0, .1, .3, .5}
  lacf_sensitivity_v18   β ∈ {0.3, 1, 3}, H ∈ {4, 8, 16}
Every table carries provenance; a smoke run is stamped SMOKE_TEST_NOT_RESULTS
and cannot satisfy the gate.

    python -m olive_if_v18.lacf_v2 --views runs/b0 runs/b1 runs/b2 --out artifacts_v18
    python -m olive_if_v18.lacf_v2 --smoke --max-folds 2 --out artifacts_v18
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:                                   # exact gradients when autograd is installed
    import autograd.numpy as anp
    from autograd import grad as _grad
    from autograd.tracer import getval as _getval
    HAVE_AUTOGRAD = True
except ImportError:                    # pragma: no cover — finite differences
    anp = np; _grad = None; _getval = lambda x: x; HAVE_AUTOGRAD = False

from olive_if_v17._common17 import fmt3, write_table
from olive_if_v17.sota_dl_comparison import (ALPHA, CITE, OPERATORS, SEEDS, VENUE, CertifiedScorer,
                                             GatedMixture, _H, _norm, default_cost_matrix, load_views,
                                             sum_rule, synthetic_views)

OURS = "LACF-v2 (ours)"

# --- base-operator patch ---
# 기반 연산자. v18 원 실행은 sum_rule 고정이었으므로 기본값을 그대로 두어
# 이전 산출물이 재현되게 한다. features() 가 호출 시점에 이 전역을 읽으므로,
# 이 모듈을 어떤 방식으로 import 했든 기반이 일관되게 적용된다.
BASE_NAME = "sum_rule"


def set_base(name):
    """기반 연산자를 이름으로 지정한다. 알 수 없는 이름이면 즉시 실패한다."""
    global BASE_NAME
    if name not in OPERATORS:
        raise SystemExit(f"알 수 없는 기반 '{name}'. 가능: {sorted(OPERATORS)}")
    BASE_NAME = name
    return name


def base_rule(P):
    return OPERATORS[BASE_NAME](P)
# --- end base-operator patch ---


# ------------------------------------------------------------------ features / sliver
# --- cost-unit features patch ---
# margin/eps 를 어느 단위로 잴 것인가.
#   "prob"  v18 원 구현. 확률 상위 2개 차이와 뷰 사후분포 평균 TV. 비용 무관.
#   "cost"  Theorem 4 의 정의. 기대비용 J(a|x)=sum_y p(y|x) C[y,a] 위에서 잰다.
# 기본값을 "prob" 으로 두어 기존 산출물이 그대로 재현되게 한다.
# 환경변수 LACF_UNITS 로도 지정할 수 있게 해서, features() 를 직접 import 하는
# 보조 스크립트(sliver_referral / sliver_arithmetic / lacf_sliver_k / lacf_ceiling)
# 를 고치지 않고도 같은 단위로 돌릴 수 있게 한다.
UNITS = __import__("os").environ.get("LACF_UNITS", "prob")
COST = None
PENDING_UNITS = UNITS           # main() 이 설정, run() 이 비용 행렬을 안 뒤 적용


def set_units(units, cost=None):
    """단위를 지정한다. cost 단위를 쓰려면 비용 행렬이 반드시 있어야 한다."""
    global UNITS, COST
    if units not in ("prob", "cost"):
        raise SystemExit(f"알 수 없는 단위 '{units}'. prob 또는 cost")
    if units == "cost" and cost is None:
        raise SystemExit("units='cost' 에는 비용 행렬이 필요합니다")
    UNITS, COST = units, (None if cost is None else np.asarray(cost, float))
    return UNITS


def _margin_eps(P, pb, cost=None):
    """cost 를 주면 그 행렬로 계산한다(비용행렬 스윕용). 없으면 전역 COST."""
    if UNITS == "prob":
        srt = np.sort(pb, -1)
        margin = srt[:, -1] - srt[:, -2]
        eps = 0.5 * np.abs(P - pb[None]).sum(-1).mean(0)
        return margin, eps
    C = COST if cost is None else np.asarray(cost, float)
    if C is None:                              # 환경변수 경로: K 로부터 결정적으로 만든다
        C = COST_ = default_cost_matrix(pb.shape[1])
        globals()["COST"] = C
    # C 는 (K, A), 마지막 열은 이관
    Jb = pb @ C
    srt = np.sort(Jb, -1)
    margin = srt[:, 1] - srt[:, 0]             # 최적/차선 기대비용 간격
    Jv = np.einsum("jnk,ka->jna", P, C)
    eps = np.abs(Jv - Jb[None]).max(-1).mean(0)
    return margin, eps


def features(P):
    J, n, K = P.shape
    pb = base_rule(P)          # 기반 연산자 (BASE_NAME)
    mx = P.max(-1).T
    js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                  for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
    margin, eps = _margin_eps(P, pb)
    phi = np.c_[mx, js, margin, _H(P).mean(0), eps]
    return phi, pb, margin, eps
# --- end cost-unit features patch ---


def sliver(margin, eps, k=2.0):
    return (margin <= k * eps).astype(float)


# ------------------------------------------------------------------ model
class LACFv2:
    def __init__(self, cost, alpha=ALPHA, hidden=8, beta=1.0, gamma=0.05, l2=1e-3, tau=0.5,
                 seed=0, maxiter=40, use_gate=True, use_residual=True, use_pinball=True, sliver_k=2.0):
        self.C, self.alpha, self.H, self.beta, self.gamma, self.l2, self.tau = cost, alpha, hidden, beta, gamma, l2, tau
        self.seed, self.maxiter = seed, maxiter
        self.use_gate, self.use_residual, self.use_pinball, self.sliver_k = use_gate, use_residual, use_pinball, sliver_k

    # -- parameters ---------------------------------------------------------------
    def _shapes(self, d, J, K):
        return [("W1", (d, self.H)), ("b1", (self.H,)), ("Wa", (self.H, J)), ("ba", (J,)),
                ("Wr", (self.H, K)), ("br", (K,)), ("wg", (self.H,)), ("bg", (1,))]

    def _unpack(self, theta, d, J, K):
        out, i = {}, 0
        for name, shp in self._shapes(d, J, K):
            n = int(np.prod(shp)); out[name] = anp.reshape(theta[i:i + n], shp); i += n
        return out

    # -- forward ------------------------------------------------------------------
    def forward(self, P, theta=None, return_parts=False):
        theta = self.theta if theta is None else theta
        J, n, K = P.shape
        phi, pb, margin, eps = features(P)
        lam = sliver(margin, eps, self.sliver_k) if self.use_gate else np.ones(n)
        w = self._unpack(theta, phi.shape[1], J, K)
        h = anp.tanh(anp.dot(phi, w["W1"]) + w["b1"])
        a = anp.dot(h, w["Wa"]) + w["ba"]; a = anp.exp(a - anp.max(a, 1, keepdims=True)); a = a / anp.sum(a, 1, keepdims=True)
        p_mix = anp.sum(a.T[:, :, None] * P, 0)
        if self.use_residual:
            delta = 4 * anp.tanh((anp.dot(h, w["Wr"]) + w["br"]) / 4)          # bounded residual, smooth
            p_res = p_mix * anp.exp(self.tau * delta); p_res = p_res / anp.sum(p_res, 1, keepdims=True)
        else:
            p_res = p_mix
        g = 1 / (1 + anp.exp(-(anp.dot(h, w["wg"]) + w["bg"][0])))
        p = pb + (lam * g)[:, None] * (p_res - pb); p = anp.clip(p, 1e-12, None); p = p / anp.sum(p, 1, keepdims=True)
        return (p, lam, g, pb) if return_parts else p

    __call__ = forward

    # -- objective ----------------------------------------------------------------
    def _loss(self, theta, P, y, T=0.05):
        p, lam, g, _ = self.forward(P, theta, return_parts=True)
        K = self.C.shape[0]
        ec = anp.dot(p, self.C[:, :K])
        soft = anp.exp(-(ec - anp.min(ec, 1, keepdims=True)) / T); soft = soft / anp.sum(soft, 1, keepdims=True)
        real = anp.sum(soft * self.C[y, :K], 1)
        L = anp.mean(real)
        if self.use_pinball:
            # CVaR_{1-α} = q + E[(real − q)+]/α (Rockafellar–Uryasev); q is the empirical
            # (1−α)-quantile, at which ∂CVaR/∂q = 0, so holding q fixed gives the exact gradient.
            q = float(np.quantile(_getval(real), 1 - self.alpha))
            L = L + self.beta * (q + anp.mean(anp.maximum(real - q, 0.0)) / self.alpha)
        L = L + self.gamma * anp.mean(lam * g) + self.l2 * anp.sum(theta ** 2)
        return L

    def fit(self, P, y):
        J, n, K = P.shape; d = features(P)[0].shape[1]
        size = sum(int(np.prod(s)) for _, s in self._shapes(d, J, K))
        theta0 = np.random.default_rng(self.seed).normal(0, 0.05, size)
        if HAVE_AUTOGRAD:
            jac = lambda th, *_: _grad(lambda t: self._loss(t, P, y))(th)
            res = optimize.minimize(self._loss, theta0, args=(P, y), jac=jac, method="L-BFGS-B",
                                    options={"maxiter": self.maxiter})
        else:
            res = optimize.minimize(self._loss, theta0, args=(P, y), method="L-BFGS-B",
                                    options={"maxiter": self.maxiter, "eps": 1e-4})
        self.theta, self.fit_loss, self.n_params = res.x, float(res.fun), size
        return self


# ------------------------------------------------------------------ protocol helpers
def _pairs(keys, max_folds):
    pairs = sorted({(s, f) for (_, s, f) in keys if ("cal", s, f) in keys and ("test", s, f) in keys})
    return [p for p in pairs if max_folds is None or p[1] < max_folds]


def _split(y_cal, s, f):
    rng = np.random.default_rng(s * 1000 + f); idx = rng.permutation(len(y_cal)); h = len(idx) // 2
    return idx[:h], idx[h:]


def _score(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, alpha):
    sc = CertifiedScorer(C, alpha).fit(op(P_cal[:, conf_i]), y_cal[conf_i])
    return sc.evaluate(op(P_tst), y_tst)


def _fold_data(per_view, s, f):
    P_cal = np.stack([v[("cal", s, f)][0] for v in per_view]); y_cal = per_view[0][("cal", s, f)][1]
    P_tst = np.stack([v[("test", s, f)][0] for v in per_view]); y_tst = per_view[0][("test", s, f)][1]
    return P_cal, y_cal, P_tst, y_tst


def _holm(p: pd.Series) -> pd.Series:
    order = p.dropna().sort_values()
    adj = pd.Series({m: min(1.0, pv * (len(order) - i)) for i, (m, pv) in enumerate(order.items())})
    return adj.cummax().reindex(p.index)


def summarise(per, ours=OURS):
    metrics = ["cost", "bound_W", "violation", "missed_disease", "set_size", "coverage", "automation"]
    g = per.groupby("method")
    tab = pd.DataFrame({m: g[m].mean() for m in metrics}).join(pd.DataFrame({m + "_sd": g[m].std() for m in metrics}))
    ref = per[per.method == ours].set_index(["seed", "fold"])
    p = {}
    for m in tab.index:
        if m == ours:
            p[m] = np.nan; continue
        d = per[per.method == m].set_index(["seed", "fold"]).loc[ref.index]["cost"] - ref["cost"]
        p[m] = stats.wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    tab["p_holm_vs_ours"] = _holm(pd.Series(p))
    tab["cite"] = [CITE.get(m, "") for m in tab.index]; tab["venue"] = [VENUE.get(m, "this work") for m in tab.index]
    tab["is_proposed"] = tab.index == ours
    o = tab.loc[ours]
    tab["beats_all_sota"] = all((o["cost"] < tab.loc[m, "cost"]) and (tab.loc[m, "p_holm_vs_ours"] < 0.05)
                                and (o["bound_W"] <= tab.loc[m, "bound_W"] + 1e-9) for m in tab.index if m != ours)
    if "Lambda" in per:
        lam = per[per.method == ours]["Lambda"].mean(); best = tab.drop(ours)["cost"].min()
        tab.loc[ours, "leverage_utilisation"] = (best - o["cost"]) / lam if lam > 0 else np.nan
        tab.loc[ours, "gain_over_best_global"] = best - o["cost"]
        tab.loc[ours, "Lambda"] = lam
    return tab.sort_values("cost").reset_index().rename(columns={"index": "method"})


def to_latex(tab, path, caption, label, ours=OURS):
    L = ["\\begin{table*}[t]\\centering", "\\begin{adjustbox}{max width=\\textwidth}", "\\begin{tabular}{llrrrrrr}", "\\toprule",
         "Method & Venue & Cost (EUR/tree) & Bound $W$ & Violation & Missed disease & Set size & $p_{\\mathrm{Holm}}$\\\\", "\\midrule"]
    for _, r in tab.iterrows():
        nm = r["method"].replace("_", "\\_")
        name = "\\textbf{" + nm + "}" if r.get("is_proposed", False) else (f"{nm}~\\cite{{{r['cite']}}}" if r.get("cite") else nm)
        pm = lambda m: f"{fmt3(r[m])} $\\pm$ {fmt3(r[m + '_sd'])}" if m + "_sd" in r else fmt3(r[m])
        L.append(f"{name} & {r.get('venue', '')} & {pm('cost')} & {pm('bound_W')} & {fmt3(r['violation'])} & "
                 f"{fmt3(r['missed_disease'])} & {fmt3(r['set_size'])} & "
                 f"{'--' if r.get('is_proposed', False) or pd.isna(r.get('p_holm_vs_ours', np.nan)) else fmt3(r['p_holm_vs_ours'])}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    path.write_text("\n".join(L))


# ------------------------------------------------------------------ experiments
def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, max_folds=None, seeds=SEEDS, maxiter=40):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if view_dirs:
        per_view, keys = load_views(view_dirs); prov = "derived-from-released-dumps"
    else:
        per_view, keys = synthetic_views(); prov = "SMOKE_TEST_NOT_RESULTS"
    K = next(iter(per_view[0].values()))[0].shape[1]; C = default_cost_matrix(K)
    set_units(PENDING_UNITS, C)
    pairs = _pairs(keys, max_folds)
    main_rows, abl_rows, miss_rows, sens_rows, params = [], [], [], [], []
    ABL = {"full": {}, "no-gate (global learner)": {"use_gate": False}, "no-residual": {"use_residual": False},
           "no-pinball": {"use_pinball": False}, "no-capacity-penalty": {"gamma": 0.0}, "sliver k=1": {"sliver_k": 1.0},
           "sliver k=3": {"sliver_k": 3.0}}
    for s, f in pairs:
        P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
        fit_i, conf_i = _split(y_cal, s, f)
        _, pb, margin, eps = features(P_tst); lam = sliver(margin, eps)
        Lambda = float(C[1:, 0].max() * lam.mean())
        # --- main comparison ------------------------------------------------------
        methods = dict(OPERATORS); methods["gated_mixture"] = GatedMixture().fit(P_cal[:, fit_i], y_cal[fit_i])
        model = LACFv2(C, alpha, seed=s, maxiter=maxiter).fit(P_cal[:, fit_i], y_cal[fit_i]); methods[OURS] = model
        params.append({"seed": s, "fold": f, "n_params": model.n_params, "fit_loss": model.fit_loss, "sliver_frac": float(lam.mean())})
        for name, op in methods.items():
            r = _score(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, alpha)
            r.update({"method": name, "seed": s, "fold": f, "Lambda": Lambda, "sliver_frac": float(lam.mean())}); main_rows.append(r)
        # --- ablation --------------------------------------------------------------
        for name, kw in ABL.items():
            m = model if name == "full" else LACFv2(C, alpha, seed=s, maxiter=maxiter, **kw).fit(P_cal[:, fit_i], y_cal[fit_i])
            r = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, alpha); r.update({"method": name, "seed": s, "fold": f}); abl_rows.append(r)
        # --- missing view stress ---------------------------------------------------
        rng = np.random.default_rng(s + f)
        for eta in (0.0, 0.1, 0.3, 0.5):
            P_m = P_tst.copy(); drop = rng.random(P_m.shape[1]) < eta
            P_m[-1, drop] = 1.0 / K                                  # weakest view → vacuous
            for name in (OURS, "sum_rule", "tmc", "qmf"):
                op = methods[name]
                sc = CertifiedScorer(C, alpha).fit(op(P_cal[:, conf_i]), y_cal[conf_i]); r = sc.evaluate(op(P_m), y_tst)
                r.update({"method": name, "eta": eta, "seed": s, "fold": f}); miss_rows.append(r)
        # --- sensitivity (first seed only: cost of the grid) -------------------------
        if s == pairs[0][0]:
            for beta in (0.3, 1.0, 3.0):
                for H in (4, 8, 16):
                    m = LACFv2(C, alpha, seed=s, maxiter=maxiter, beta=beta, hidden=H).fit(P_cal[:, fit_i], y_cal[fit_i])
                    r = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, alpha); r.update({"beta": beta, "hidden": H, "seed": s, "fold": f}); sens_rows.append(r)
    main = pd.DataFrame(main_rows); write_table(main, out / "lacf_main_perfold_v18.csv", prov)
    tab = summarise(main); write_table(tab, out / "lacf_main_v18.csv", prov)
    to_latex(tab, out / "lacf_main_v18.tex",
             f"Leverage-aware certified fusion against twelve global fusion operators at matched automation, "
             f"$\\alpha={alpha}$, mean $\\pm$ SD over {len(seeds)} seeds $\\times$ folds; all rules certified on the same disjoint "
             f"block (Theorem 2). Improvement gate {'met' if tab['beats_all_sota'].iloc[0] else 'NOT met'}. Provenance: {prov}.",
             "tab:lacf_main_v18")
    abl = pd.DataFrame(abl_rows)
    # 절제 대조는 폴드가 짝지어져 있으므로 쌍대 분해능으로 검정해야 한다.
    # 예치하지 않으면 독자가 그 검정을 할 수 없다.
    write_table(abl, out / "lacf_ablation_perfold_v18.csv", prov)
    g = abl.groupby("method")
    abl_t = pd.DataFrame({"cost": g.cost.mean(), "cost_sd": g.cost.std(), "bound_W": g.bound_W.mean(), "bound_W_sd": g.bound_W.std(),
                          "violation": g.violation.mean(), "missed_disease": g.missed_disease.mean(), "set_size": g.set_size.mean()}).reset_index()
    write_table(abl_t, out / "lacf_ablation_v18.csv", prov)
    to_latex(abl_t, out / "lacf_ablation_v18.tex", f"Ablation of LACF-v2 (mean $\\pm$ SD over seeds $\\times$ folds). Provenance: {prov}.", "tab:lacf_ablation_v18")
    miss = pd.DataFrame(miss_rows); miss_t = miss.groupby(["method", "eta"]).agg(cost=("cost", "mean"), cost_sd=("cost", "std"),
                                                                                  violation=("violation", "mean"), missed=("missed_disease", "mean")).reset_index()
    write_table(miss_t, out / "lacf_missing_view_v18.csv", prov)
    sens = pd.DataFrame(sens_rows); sens_t = sens.groupby(["beta", "hidden"]).agg(cost=("cost", "mean"), bound_W=("bound_W", "mean"), violation=("violation", "mean")).reset_index()
    write_table(sens_t, out / "lacf_sensitivity_v18.csv", prov)
    (out / "lacf_params_v18.json").write_text(json.dumps({"provenance": prov, "base": BASE_NAME, "alpha": alpha, "seeds": list(seeds), "folds": params}, indent=1))
    _figures(out, miss_t, sens_t)
    return tab, abl_t, miss_t, sens_t, prov


def _figures(out, miss_t, sens_t):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from olive_if_v17.figure_hygiene import finalize
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for m, d in miss_t.groupby("method"):
        axes[0].errorbar(d.eta, d.cost, yerr=d.cost_sd, marker="o", capsize=2, label=m)
    axes[0].set_xlabel("missing-view rate η"); axes[0].set_ylabel("realised cost (EUR / tree)"); axes[0].legend(fontsize=6.5)
    piv = sens_t.pivot(index="beta", columns="hidden", values="cost")
    for H in piv.columns:
        axes[1].plot(piv.index, piv[H], marker="s", label=f"H = {H}")
    axes[1].set_xscale("log"); axes[1].set_xlabel("β (certificate-width weight)"); axes[1].set_ylabel("realised cost (EUR / tree)"); axes[1].legend(fontsize=6.5)
    fig.subplots_adjust(wspace=0.35, bottom=0.3)
    (out / "figures").mkdir(exist_ok=True)
    finalize(fig, "fig_lacf_stress_v18", report_dir=out)
    fig.savefig(out / "figures" / "fig_lacf_stress_v18.pdf", bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int); ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--out", default="artifacts_v18")
    ap.add_argument("--base", default="sum_rule",
                    help="기반 연산자 (기본 sum_rule = v18 원 실행)")
    ap.add_argument("--units", default=__import__("os").environ.get("LACF_UNITS", "prob"),
                    choices=("prob", "cost"),
                    help="margin/eps 단위. cost 는 Theorem 4 정의 (기본 prob = v18 원 실행)")
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("give --views <run dirs> or --smoke")
    global PENDING_UNITS
    PENDING_UNITS = a.units
    set_base(a.base)
    print(f"단위: {PENDING_UNITS}")
    print(f"기반 연산자: {BASE_NAME}")
    tab, abl, miss, sens, prov = run(a.out, a.views, a.smoke, max_folds=a.max_folds, maxiter=a.maxiter)
    print(f"[{prov}]"); print(tab[["method", "cost", "bound_W", "violation", "p_holm_vs_ours", "beats_all_sota"]]
                                .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(abl[["method", "cost", "bound_W"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
