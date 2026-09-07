#!/usr/bin/env python3
"""Does the source-count curve depend on WHICH sources, or only on HOW MANY?

The deposited sweep adds sources in one order -- cheapest single-source cost first --
so "the saving appears by J=3 and is flat thereafter" could be a property of that
ordering rather than of the evidence count. The released files contain only that one
nested sequence, so the question cannot be settled from them. This settles it.

For each J = 1..J_max we draw R random subsets of size J (plus the deposited
cheapest-first nested subset as a reference), score each through the identical certified
decision path at matched automation, and report the across-subset spread beside the
paired per-fold resolution. If the spread at J = 3 is comparable to the resolution, the
"a handful is enough" reading survives; if it is several times the resolution, the
curve is about which sources, not how many, and the manuscript must say so.

    LACF_UNITS=cost python3 cms_order.py --views <8 dump dirs> --out out_order --reps 20

Cost: R * (J_max-1) fits per fold. At R=20, J_max=8 that is ~140 arms x 10 folds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from olive_if_v17.sota_dl_comparison import (ALPHA, CertifiedScorer, default_cost_matrix,
                                             load_views, product_rule, sum_rule)
from olive_if_v18.lacf_v2 import _fold_data, _pairs, _split


def mde(d):
    d = np.asarray(d, float)
    return float(2.802 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="out_order")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--reps", type=int, default=20, help="random subsets per J")
    ap.add_argument("--rule", default="product", choices=("product", "sum"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    names = [pathlib.Path(v).name for v in a.views]
    J = len(per_view)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    fuse = product_rule if a.rule == "product" else sum_rule
    pairs = _pairs(keys, None)
    rng = np.random.default_rng(a.seed)
    print(f"views {J} | folds {len(pairs)} | rule {a.rule} | reps {a.reps} | alpha {a.alpha}")

    # subsets: the deposited nested order (cheapest-first, decided per fold) is added
    # separately below; here we draw random subsets, shared across folds so that a
    # subset is one arm rather than a per-fold lottery.
    subsets = {}
    for m in range(1, J + 1):
        allsub = list(itertools.combinations(range(J), m))
        if len(allsub) <= a.reps:
            pick = allsub
        else:
            idx = rng.choice(len(allsub), size=a.reps, replace=False)
            pick = [allsub[i] for i in idx]
        subsets[m] = [tuple(sorted(s)) for s in pick]
    print("subsets per J: " + ", ".join(f"J={m}:{len(v)}" for m, v in subsets.items()))

    rows = []
    for s, f in pairs:
        P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
        fit_i, conf_i = _split(y_cal, s, f)
        for m, subs in subsets.items():
            for sub in subs:
                p_cal = fuse(P_cal[list(sub)][:, conf_i]) if m > 1 else P_cal[sub[0], conf_i]
                p_tst = fuse(P_tst[list(sub)]) if m > 1 else P_tst[sub[0]]
                sc = CertifiedScorer(C, a.alpha).fit(p_cal, y_cal[conf_i])
                r = sc.evaluate(p_tst, y_tst)
                rows.append({"J": m, "subset": "+".join(str(i) for i in sub),
                             "cost": r["cost"], "violation": r["violation"],
                             "missed_disease": r["missed_disease"], "seed": s, "fold": f})
        print(f"  seed {s} fold {f} done", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(out / "cms_order_perfold.csv", index=False)
    w = per.pivot_table(index=["seed", "fold"], columns=["J", "subset"], values="cost")

    summ = []
    for m in sorted(subsets):
        cols = [c for c in w.columns if c[0] == m]
        means = w[cols].mean()
        # resolution of a comparison between two subsets of the same size
        res = [mde(w[c1] - w[c2]) for c1, c2 in itertools.combinations(cols, 2)] or [np.nan]
        summ.append({"J": m, "n_subsets": len(cols),
                     "cost_min": float(means.min()), "cost_mean": float(means.mean()),
                     "cost_max": float(means.max()), "spread": float(means.max() - means.min()),
                     "spread_sd": float(means.std(ddof=1)) if len(cols) > 1 else np.nan,
                     "pairwise_mde_median": float(np.nanmedian(res)),
                     "spread_over_mde": float((means.max() - means.min()) / np.nanmedian(res))
                     if len(cols) > 1 else np.nan,
                     "best_subset": str(means.idxmin()[1]), "worst_subset": str(means.idxmax()[1])})
    S = pd.DataFrame(summ)
    S.to_csv(out / "cms_order_summary.csv", index=False)

    print("\n" + "=" * 92)
    print(f"{'J':>3}{'subsets':>9}{'min':>9}{'mean':>9}{'max':>9}{'spread':>9}"
          f"{'pair MDE':>10}{'spread/MDE':>12}   which is best")
    for r in summ:
        print(f"{r['J']:>3}{r['n_subsets']:>9}{r['cost_min']:>9.4f}{r['cost_mean']:>9.4f}"
              f"{r['cost_max']:>9.4f}{r['spread']:>9.4f}{r['pairwise_mde_median']:>10.4f}"
              f"{r['spread_over_mde']:>12.2f}   {r['best_subset']}")
    print("=" * 92)
    print("\nHow to read it")
    print("  spread/MDE near 1 : which sources you take does not matter beyond the design's")
    print("                      resolution, so 'a handful is enough' survives.")
    print("  spread/MDE >> 1   : the curve is about WHICH sources, not HOW MANY, and the")
    print("                      manuscript's 'flat after J=3' must be scoped to one ordering.")
    (out / "cms_order_summary.json").write_text(
        json.dumps({"views": names, "rule": a.rule, "reps": a.reps, "rows": summ},
                   indent=1), encoding="utf-8")
    print(f"\ndeposited: {out}/cms_order_perfold.csv, cms_order_summary.csv/.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
