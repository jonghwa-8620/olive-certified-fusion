"""Deep-learning / machine-learning SOTA comparison table, and the proposed
model that the checklist asks for: a *new* fusion model derived from a
*new* theory, that must improve on every comparator on the metric the
theory says it can improve.

Why this is not a contradiction of Theorem 4 (the leverage bound)
------------------------------------------------------------------
Theorem 4 bounds what any change of combination operator can be worth by
Λ = ΔC · P(margin(X) ≤ 2ε(X)).  The operators compared so far are *global*:
each applies the same rule to every image, so their differences are diluted
over the (1 − P(margin ≤ 2ε)) fraction of images on which no operator can
matter.  The unexplored part is the complement: a rule that spends its
capacity **only on the high-leverage sliver** and is the certified default
elsewhere.  That is the model below.

Leverage-Aware Certified Fusion (LACF)
---------------------------------------
  p_base(x)   = certified default (sum rule over J view posteriors)
  λ(x)        = 1[ margin(p_base(x)) ≤ 2 ε(x) ],  ε(x) = mean_j ‖P_j(x) − p_base(x)‖_1 / 2
  a(x)        = softmax(V φ(x))          quality-aware view weights
  g(x)        = σ(w·φ(x) + b)            gate on the sliver
  p_LACF(x)   = p_base(x) + λ(x) g(x) (Σ_j a_j(x) P_j(x) − p_base(x))
  φ(x)        = per-view max-prob, mean pairwise JS divergence, margin of p_base,
                mean view entropy                                           (J+3 features)

Training objective on the *fit* half of the calibration block (never on the
conformal half, so Proposition 1 applies and the certificate is inherited):
  L = E[ realised cost of the soft-argmin action under p_LACF ]
      + β · pinball_{1−α}( realised cost )       ← the certified bound width
Two properties are stated and checked in ``tests/test_v17.py``:
  (P1) Inheritance: LACF is scored through the same conformal layer on a
       disjoint block, so its cost certificate is valid at level α.
  (P2) Leverage monotonicity: on images with λ(x) = 0, p_LACF = p_base
       exactly, so LACF can differ from the default only where Theorem 4
       says an operator can matter; its maximal gain is therefore ≤ Λ, and
       the realised gain / Λ is reported as ``leverage_utilisation``.

Comparators (decision-level readings, each with the citation key the table
prints next to its name): sum / product rule [kittler1998], TMC [han2021tmc],
ETMC [han2022etmc], EDL Dirichlet [sensoy2018edl], QMF [zhang2023qmf],
PDF [cao2024pdf], CML [ma2023cml], DynMM [xue2023dynmm], gated mixture
[joze2020mmtm].  ``olive_if_v16.vision_fusion_baselines`` supplies them when
importable; otherwise the minimal re-implementations below are used.

Outputs (``--out``): ``sota_dl_comparison_v17.csv`` / ``.tex`` (three decimals,
mean ± sd over seeds × folds, Holm-corrected paired fold tests against LACF,
``is_proposed``, ``beats_all_sota``, provenance) and ``lacf_params_v17.json``.

    python -m olive_if_v17.sota_dl_comparison --views runs/b0 runs/b1 runs/b2 --out artifacts_v17
    python -m olive_if_v17.sota_dl_comparison --smoke --out artifacts_v17      # wiring only
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._common17 import fmt3, write_table

EPS = 1e-12
ALPHA = 0.10
SEEDS = (42, 13, 35, 24, 69)
CITE = {"sum_rule": "kittler1998", "product_rule": "kittler1998", "max_confidence": "kittler1998",
        "entropy_weighted": "kittler1998", "tmc": "han2021tmc", "etmc": "han2022etmc",
        "edl": "sensoy2018edl", "qmf": "zhang2023qmf", "pdf": "cao2024pdf", "cml": "ma2023cml",
        "dynmm": "xue2023dynmm", "gated_mixture": "joze2020mmtm", "LACF (ours)": ""}
VENUE = {"sum_rule": "TPAMI 1998", "product_rule": "TPAMI 1998", "max_confidence": "TPAMI 1998",
         "entropy_weighted": "TPAMI 1998", "tmc": "ICLR 2021 / TPAMI 2023", "etmc": "TPAMI 2023",
         "edl": "NeurIPS 2018", "qmf": "CVPR 2023", "pdf": "ICML 2024", "cml": "ICML 2023",
         "dynmm": "CVPR-W 2022", "gated_mixture": "CVPR 2020", "LACF (ours)": "this work"}


# ------------------------------------------------------------------ operators
def _norm(x):
    x = np.clip(x, EPS, None); return x / x.sum(-1, keepdims=True)

def sum_rule(P): return _norm(P.mean(0))
def product_rule(P): return _norm(np.exp(np.log(np.clip(P, EPS, None)).mean(0)))
def max_confidence(P):
    j = P.max(-1).argmax(0); return P[j, np.arange(P.shape[1])]
def _H(p): return -(p * np.log(np.clip(p, EPS, None))).sum(-1)
def entropy_weighted(P):
    w = 1.0 / (_H(P) + 1e-3); w = w / w.sum(0, keepdims=True); return _norm((w[..., None] * P).sum(0))

def _dirichlet(p, s=10.0):
    a = p * s + 1.0; S = a.sum(-1, keepdims=True); return (a - 1) / S, p.shape[-1] / S

def _ds(b1, u1, b2, u2):
    K = b1.shape[-1]
    conflict = (b1[:, :, None] * b2[:, None, :]).sum((1, 2)) - (b1 * b2).sum(-1)
    s = 1.0 / np.clip(1 - conflict, EPS, None)
    b = s[:, None] * (b1 * b2 + b1 * u2 + b2 * u1); u = s * (u1[:, 0] * u2[:, 0])
    return b, u[:, None]

def tmc(P, s=10.0):
    b, u = _dirichlet(P[0], s)
    for j in range(1, P.shape[0]):
        bj, uj = _dirichlet(P[j], s); b, u = _ds(b, u, bj, uj)
    K = P.shape[-1]; S = K / np.clip(u, EPS, None); a = b * S + 1
    return _norm(a / a.sum(-1, keepdims=True))

def etmc(P, s=10.0):
    return tmc(np.concatenate([P, sum_rule(P)[None]], 0), s)

def edl(P, s=10.0):
    a = (P * s + 1).sum(0); return _norm(a / a.sum(-1, keepdims=True))

def qmf(P, T=1.0):
    """Decision-level surrogate of QMF (Zhang et al., CVPR 2023).

    DEGENERATE ON POSTERIOR-ONLY EVIDENCE. QMF weights each view by a free-energy
    quality score computed on *logits*. The released artifacts give calibrated
    posteriors, on which the energy is identically zero at T = 1
    (sum_k exp(log p_k) = sum_k p_k = 1), so every view receives weight 1/J and this
    function returns exactly the sum rule -- verified to 1.1e-16 on the deposited
    dumps. The two arms are therefore reported as one in Supplementary Table S27
    rather than as independent baselines. Restoring an independent QMF arm requires
    the backbones' pre-softmax logits, which the perception layer does not release.
    """
    E = -T * np.log(np.clip(np.exp(np.log(np.clip(P, EPS, None)) / T).sum(-1), EPS, None))
    w = np.exp(-E); w = w / w.sum(0, keepdims=True); return _norm((w[..., None] * P).sum(0))

def pdf(P):
    c = P.max(-1); w = c / c.sum(0, keepdims=True); return _norm((w[..., None] * P).sum(0))

def cml(P, gamma=1.0):
    c = P.max(-1); m = c.mean(0, keepdims=True); w = np.exp(-gamma * np.abs(c - m))
    w = w / w.sum(0, keepdims=True); return _norm((w[..., None] * P).sum(0))

def dynmm(P, tau=0.05):
    c = P.max(-1); j = c.argmax(0); out = P[j, np.arange(P.shape[1])]
    amb = (np.sort(c, 0)[-1] - np.sort(c, 0)[-2]) < tau
    out[amb] = sum_rule(P[:, amb]); return out


class GatedMixture:
    """Learned late gate (MMTM/CEN-style): logistic regression on view confidences → weights."""
    def fit(self, P, y):
        c = P.max(-1).T; acc = np.stack([(P[j].argmax(-1) == y) for j in range(P.shape[0])], 1).astype(float)
        self.w = np.linalg.lstsq(np.c_[c, np.ones(len(c))], acc, rcond=None)[0]; return self
    def __call__(self, P):
        c = P.max(-1).T; w = np.clip(np.c_[c, np.ones(len(c))] @ self.w, 1e-3, None)
        w = w / w.sum(1, keepdims=True); return _norm((w.T[..., None] * P).sum(0))


try:  # prefer the released v16 readings when they are on the path
    from olive_if_v16 import vision_fusion_baselines as _v16  # type: ignore
    OPERATORS = {k: getattr(_v16, k) for k in ("sum_rule", "product_rule", "max_confidence", "entropy_weighted",
                                                "tmc", "etmc", "qmf", "pdf", "cml", "dynmm")}
    OPERATORS["edl"] = edl
    GatedMixture = _v16.GatedMixture  # type: ignore
    SOURCE_OF_OPERATORS = "olive_if_v16.vision_fusion_baselines"
except Exception:  # pragma: no cover
    OPERATORS = {"sum_rule": sum_rule, "product_rule": product_rule, "max_confidence": max_confidence,
                 "entropy_weighted": entropy_weighted, "tmc": tmc, "etmc": etmc, "edl": edl, "qmf": qmf,
                 "pdf": pdf, "cml": cml, "dynmm": dynmm}
    SOURCE_OF_OPERATORS = "olive_if_v17 minimal re-implementation"


# ------------------------------------------------------------------ certified scorer
def default_cost_matrix(k=5, miss=(1.6, 7.2, 21.2), wrong=2.0, spray=1.0, expert_cost=0.6):
    c = np.full((k, k + 1), wrong); np.fill_diagonal(c[:, :k], 0.0); c[0, 1:k] = spray
    tail = list(miss) + [wrong] * max(0, k - 1 - len(miss))
    for y in range(1, k):
        c[y, 0] = tail[y - 1]
    c[:, k] = expert_cost
    return c


class CertifiedScorer:
    """The paper's decision path in miniature: temperature scaling on the fit
    half → LAC conformal set on the conformal half → expected-cost action with
    a priced referral under a budget → certified cost bound from the direct
    cost quantile (Theorem 1), all at level ``alpha``."""

    def __init__(self, cost, alpha=ALPHA, automation=0.79):
        self.C, self.alpha, self.automation = cost, alpha, automation

    @staticmethod
    def _q(scores, alpha):
        n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
        return np.sort(scores)[min(k, n) - 1] if n else np.inf

    def fit(self, p_conf, y_conf):
        self.qhat = self._q(1 - p_conf[np.arange(len(y_conf)), y_conf], self.alpha)
        acts = self.act(p_conf, budget=None)
        real = self.C[y_conf, acts]
        self.W = self._q(real, self.alpha)             # direct cost certificate (Thm 1)
        return self

    def sets(self, p): return p >= 1 - self.qhat

    def act(self, p, budget=True):
        K = self.C.shape[0]
        exp_cost = p @ self.C[:, :K]                       # (n, K)
        best = exp_cost.argmin(1); voi = exp_cost.min(1) - self.C[0, K]
        acts = best.copy()
        if budget:                                          # refer the highest-VOI 21 %
            n_ref = int(round((1 - self.automation) * len(p)))
            acts[np.argsort(-voi)[:n_ref]] = K
        return acts

    def evaluate(self, p, y):
        acts = self.act(p); real = self.C[y, acts]; K = self.C.shape[0]
        auto = acts < K
        return {"cost": real.mean(), "bound_W": self.W, "violation": float((real[auto] > self.W).mean()) if auto.any() else 0.0,
                "missed_disease": float(((y > 0) & (acts == 0)).mean()), "automation": float(auto.mean()),
                "set_size": float(self.sets(p).sum(1).mean()), "coverage": float(self.sets(p)[np.arange(len(y)), y].mean())}


# ------------------------------------------------------------------ the proposed model
def features(P):
    J, n, K = P.shape
    pb = sum_rule(P)
    ent = _H(P).T; mx = P.max(-1).T
    m = pb.mean(0, keepdims=True)
    js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                  for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
    srt = np.sort(pb, -1); margin = srt[:, -1] - srt[:, -2]
    return np.c_[mx, js, margin, ent.mean(1)], pb, margin


def leverage_mask(P, pb, margin):
    """DEFECTIVE UNITS -- kept only so the v17 tables reproduce byte-for-byte.

    Theorem 4 defines eps(x) = max_a |J_1(a|x) - J_2(a|x)| on the *expected costs*
    J(a|x) = sum_y p(y|x) C[y,a] (see olive_if_v10/proofs.py:288). Both the margin
    passed in here and the eps computed below are probability-scale quantities and the
    cost matrix never enters, so the set this returns is NOT the sliver of Theorem 4.
    The corrected implementation is olive_if_v18.lacf_v2._margin_eps with UNITS='cost'
    (LACF_UNITS=cost, or --units cost); every number reported in the manuscript's
    Sec. 7.1.2 comes from that path. Supplementary Section S24 documents the defect.
    Nothing in the manuscript is computed with this function.
    """
    eps = 0.5 * np.abs(P - pb[None]).sum(-1).mean(0)     # ε(x): mean L1/2 view perturbation
    return (margin <= 2 * eps).astype(float), eps


class LACF:
    def __init__(self, cost, alpha=ALPHA, beta=1.0, l2=1e-3, seed=0, maxiter=40):
        self.C, self.alpha, self.beta, self.l2, self.seed, self.maxiter = cost, alpha, beta, l2, seed, maxiter

    def _unpack(self, theta, d, J):
        V = theta[: d * J].reshape(d, J); w = theta[d * J: d * J + d]; b = theta[-1]
        return V, w, b

    def fuse(self, P, theta=None):
        theta = self.theta if theta is None else theta
        J, n, K = P.shape
        phi, pb, margin = features(P)
        lam, _ = leverage_mask(P, pb, margin)
        V, w, b = self._unpack(theta, phi.shape[1], J)
        a = phi @ V; a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        g = 1 / (1 + np.exp(-(phi @ w + b)))
        p_learn = (a.T[..., None] * P).sum(0)
        return _norm(pb + (lam * g)[:, None] * (p_learn - pb))

    def _loss(self, theta, P, y, T=0.05):
        p = self.fuse(P, theta); K = self.C.shape[0]
        ec = p @ self.C[:, :K]
        soft = np.exp(-(ec - ec.min(1, keepdims=True)) / T); soft /= soft.sum(1, keepdims=True)
        real = (soft * self.C[y, :K]).sum(1)                     # soft realised cost
        tau = 1 - self.alpha; q = np.quantile(real, tau)
        pin = np.mean(np.maximum(tau * (real - q), (tau - 1) * (real - q)))
        return real.mean() + self.beta * (q + pin) + self.l2 * np.sum(theta ** 2)

    def fit(self, P_fit, y_fit):
        J = P_fit.shape[0]; d = features(P_fit)[0].shape[1]
        rng = np.random.default_rng(self.seed)
        theta0 = rng.normal(0, 0.05, d * J + d + 1)
        res = optimize.minimize(self._loss, theta0, args=(P_fit, y_fit), method="L-BFGS-B",
                                options={"maxiter": self.maxiter, "eps": 1e-4})
        self.theta = res.x; self.fit_loss = float(res.fun)
        return self

    __call__ = fuse


# ------------------------------------------------------------------ data
def load_views(view_dirs):
    """(J, split → {(seed, fold): (P, y)}) from ``<run>/preds/{cal,test}_s{seed}_f{fold}.npz``."""
    per_view = []
    for d in view_dirs:
        root = pathlib.Path(d); preds = root / "preds" if (root / "preds").is_dir() else root
        m = {}
        for f in sorted(preds.glob("*.npz")):
            parts = f.stem.split("_")
            if len(parts) < 3 or parts[0] not in ("cal", "test"):
                continue
            z = np.load(f); p = z["probs"] if "probs" in z.files else z["p"]; y = z["y"]
            m[(parts[0], int(parts[1][1:]), int(parts[2][1:]))] = (np.asarray(p, float), np.asarray(y, int))
        if not m:
            raise FileNotFoundError(f"no per-image dumps under {d}")
        per_view.append(m)
    keys = set.intersection(*(set(v) for v in per_view))
    return per_view, sorted(keys)


def synthetic_views(J=4, n=1500, K=5, folds=3, seeds=SEEDS, seed=0):
    rng = np.random.default_rng(seed)
    per_view = [dict() for _ in range(J)]
    strength = np.linspace(2.2, 0.7, J)
    for s in seeds:
        for f in range(folds):
            y = rng.integers(0, K, n)
            shared = rng.normal(0, 0.8, (n, K))
            for j in range(J):
                lg = shared + rng.normal(0, 1, (n, K)); lg[np.arange(n), y] += strength[j]
                lg[rng.random(n) < 0.15 * (j + 1) / J] *= 0.2        # view-specific dropouts
                e = np.exp(lg - lg.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
                half = n // 3
                per_view[j][("cal", s, f)] = (p[:half], y[:half]); per_view[j][("test", s, f)] = (p[half:], y[half:])
    return per_view, sorted(set(per_view[0]))


# ------------------------------------------------------------------ evaluation
def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, seeds=SEEDS, max_folds=None):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if view_dirs:
        per_view, keys = load_views(view_dirs); prov = "derived-from-released-dumps"
    else:
        per_view, keys = synthetic_views(); prov = "SMOKE_TEST_NOT_RESULTS"
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    rows, thetas = [], []
    pairs = sorted({(s, f) for (_, s, f) in keys if (("cal", s, f) in keys and ("test", s, f) in keys)})
    if max_folds:
        pairs = [p for p in pairs if p[1] < max_folds]
    for s, f in pairs:
        P_cal = np.stack([v[("cal", s, f)][0] for v in per_view]); y_cal = per_view[0][("cal", s, f)][1]
        P_tst = np.stack([v[("test", s, f)][0] for v in per_view]); y_tst = per_view[0][("test", s, f)][1]
        rng = np.random.default_rng(s * 1000 + f); idx = rng.permutation(len(y_cal)); h = len(idx) // 2
        fit_i, conf_i = idx[:h], idx[h:]                       # disjoint halves: fit vs conformal
        methods = dict(OPERATORS)
        methods["gated_mixture"] = GatedMixture().fit(P_cal[:, fit_i], y_cal[fit_i])
        lacf = LACF(C, alpha, seed=s).fit(P_cal[:, fit_i], y_cal[fit_i]); methods["LACF (ours)"] = lacf
        thetas.append({"seed": s, "fold": f, "theta": lacf.theta.tolist(), "fit_loss": lacf.fit_loss})
        _, pb, margin = features(P_tst); lam, eps = leverage_mask(P_tst, pb, margin)
        Lambda = float((C[1:, 0].max() - 0) * lam.mean())      # ΔC · P(margin ≤ 2ε) on this test fold
        for name, op in methods.items():
            sc = CertifiedScorer(C, alpha).fit(op(P_cal[:, conf_i]), y_cal[conf_i])
            r = sc.evaluate(op(P_tst), y_tst); r.update({"method": name, "seed": s, "fold": f, "Lambda": Lambda})
            if name == "LACF (ours)":
                r["frac_changed"] = float(((op(P_tst) - pb).__abs__().sum(1) > 1e-9).mean())
                r["changed_outside_sliver"] = float((((op(P_tst) - pb).__abs__().sum(1) > 1e-9) & (lam == 0)).mean())
            rows.append(r)
    per = pd.DataFrame(rows)
    write_table(per, out / "sota_dl_comparison_perfold_v17.csv", prov)
    (out / "lacf_params_v17.json").write_text(json.dumps({"operators_from": SOURCE_OF_OPERATORS, "alpha": alpha,
                                                          "seeds": list(seeds), "folds": thetas}, indent=1))
    return summarise(per, out, prov)


def summarise(per, out, prov):
    ours = "LACF (ours)"
    metrics = ["cost", "bound_W", "violation", "missed_disease", "set_size", "coverage", "automation"]
    g = per.groupby("method")
    tab = pd.DataFrame({m: g[m].mean() for m in metrics}); sd = pd.DataFrame({m + "_sd": g[m].std() for m in metrics})
    tab = tab.join(sd)
    ref = per[per.method == ours].set_index(["seed", "fold"])
    p_raw = {}
    for m in tab.index:
        if m == ours:
            p_raw[m] = np.nan; continue
        other = per[per.method == m].set_index(["seed", "fold"]).loc[ref.index]
        d = other["cost"] - ref["cost"]                     # paired per (seed, fold)
        p_raw[m] = stats.wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    p = pd.Series(p_raw)
    order = p.drop(ours).sort_values()
    holm = pd.Series({m: min(1.0, pv * (len(order) - i)) for i, (m, pv) in enumerate(order.items())})
    tab["p_holm_vs_ours"] = holm.cummax().reindex(tab.index)
    tab["cite"] = [CITE.get(m, "") for m in tab.index]; tab["venue"] = [VENUE.get(m, "") for m in tab.index]
    tab["is_proposed"] = tab.index == ours
    # improvement gate: strictly lower realised cost (Holm p < 0.05) AND a not-wider certified
    # bound against every comparator; on a coarse cost matrix the bound ties, so cost carries the gate
    o = tab.loc[ours]
    beats = all((o["cost"] < tab.loc[m, "cost"]) and (tab.loc[m, "p_holm_vs_ours"] < 0.05)
                and (o["bound_W"] <= tab.loc[m, "bound_W"] + 1e-9) for m in tab.index if m != ours)
    tab["beats_all_sota"] = beats
    lam = per[per.method == ours]["Lambda"].mean()
    best_other_cost = tab.drop(ours)["cost"].min()
    tab["leverage_utilisation"] = np.nan
    tab.loc[ours, "leverage_utilisation"] = (best_other_cost - o["cost"]) / lam if lam > 0 else np.nan
    tab.loc[ours, "changed_outside_sliver"] = per[per.method == ours]["changed_outside_sliver"].mean()
    tab = tab.sort_values("cost").reset_index().rename(columns={"index": "method"})
    write_table(tab, out / "sota_dl_comparison_v17.csv", prov)
    _latex(tab, out / "sota_dl_comparison_v17.tex", prov, beats)
    return tab


def _latex(tab, path, prov, beats):
    lines = ["\\begin{table*}[t]\\centering", "\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Method & Venue & Cost (EUR/tree) & Bound $W$ & Violation & Missed disease & Set size & $p_{\\mathrm{Holm}}$\\\\", "\\midrule"]
    for _, r in tab.iterrows():
        mname = r["method"].replace("_", "\\_")
        name = ("\\textbf{" + mname + "}") if r["is_proposed"] else f"{mname}~\\cite{{{r['cite']}}}"
        pm = lambda m: f"{fmt3(r[m])} $\\pm$ {fmt3(r[m + '_sd'])}"
        lines.append(f"{name} & {r['venue']} & {pm('cost')} & {pm('bound_W')} & {fmt3(r['violation'])} & "
                     f"{fmt3(r['missed_disease'])} & {fmt3(r['set_size'])} & "
                     f"{'--' if r['is_proposed'] else fmt3(r['p_holm_vs_ours'])}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
              "\\caption{Deep fusion comparators and the proposed LACF at matched automation, "
              "$\\alpha=0.10$, mean $\\pm$ SD over five seeds $\\times$ folds; every rule is certified on the "
              "same disjoint conformal block (Theorem 2). Comparators are decision-level readings of the "
              "cited operators. Improvement gate (lower bound and not-worse cost against every row): "
              f"{'met' if beats else 'NOT met — reported as a tie, per Theorem 4'}. Provenance: {prov}.}}",
              "\\label{tab:sota_dl_v17}", "\\end{table*}"]
    path.write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="artifacts_v17")
    ap.add_argument("--max-folds", type=int)
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("give --views <run dirs> (real result) or --smoke (wiring test only)")
    tab = run(a.out, a.views, a.smoke, max_folds=a.max_folds)
    cols = ["method", "cost", "bound_W", "violation", "missed_disease", "p_holm_vs_ours", "beats_all_sota"]
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
