#!/usr/bin/env python3
"""Intervals and paired tests for the deep + agronomic-descriptor fusion table.

Regenerates every interval and every paired comparison the manuscript reports for that
table, from the deposited per-fold file and nothing else.

    python3 agro_intervals.py --perfold hetero_recert_lam0_perfold.csv --out .

Writes
    agro_arm_intervals.csv     one row per (certification block, arm): mean cost over the
                               thirty folds and its 95 % t interval
    agro_paired_tests.csv      one row per ordered pair within a block: paired per-fold
                               difference, its 95 % interval, the paired t p-value, this
                               design's minimum detectable effect and whether the pair
                               resolves

Conventions, the same as everywhere else in this deposit.
  * The unit of analysis is the fold. Thirty rows, three seeds by ten folds; the interval is
    a Student t interval on those thirty per-fold means, not on images.
  * Intervals and tests carry the Nadeau-Bengio fold-variance inflation, which Section 5.9
    of the manuscript declares for every significance test it makes: the variance of a
    k-fold mean is inflated by (1/k + rho) rather than 1/k, with rho = n_test / n_train.
    Here k = 30, n_test = 250 and n_train = 2250, so the standard error is multiplied by
    sqrt((1/30 + 250/2250)/(1/30)) = 2.0817. Without it these thirty numbers would be
    treated as independent, which they are not: they are three seeds over ten folds of the
    same 2,500 images, with overlapping training sets.
  * MDE is the two-sided 80 %-power minimum detectable effect for the paired per-fold
    difference, 2.802 * SE_inflated. A pair whose effect is smaller than its MDE is
    reported as a tie and nothing is claimed from its sign.
  * No multiplicity correction is applied inside this file. The manuscript applies Holm
    where it makes a claim across the family.
  * Cost only. The missed-disease column of the manuscript's table is on the corrected
    metric, whose per-fold file is not part of this deposit; this file does not reproduce it
    and does not pretend to.

Note on one arm. `proposed` and `late_sum` carry identical per-fold costs in both blocks
(max |difference| = 0). Equation (1) at lambda = 0 reduces to the late sum when there are
two sources, so the manuscript prints one row for the two and this script asserts the
identity rather than assuming it.
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd
from scipy import stats

ARMS = ["evidence_murphy", "evidence_dempster", "late_product", "late_sum",
        "proposed", "deep_only", "interp_only"]
Z80 = 2.802      # two-sided 80 % power at alpha = 0.05
N_TEST = 250     # images scored per fold
N_TRAIN = 2250   # images not scored per fold


def nb_factor(k, n_test=N_TEST, n_train=N_TRAIN):
    """Nadeau-Bengio SE inflation: variance (1/k + rho) instead of 1/k."""
    rho = n_test / n_train
    return float(np.sqrt((1.0 / k + rho) / (1.0 / k)))


def interval(v):
    """Mean and 95 % interval, Nadeau-Bengio inflated."""
    v = np.asarray(v, float)
    n = len(v)
    m = v.mean()
    se = v.std(ddof=1) / np.sqrt(n) * nb_factor(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfold", required=True,
                    help="hetero_recert_lam0_perfold.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.perfold)

    need = {"arm", "certification", "seed", "fold", "mean_cost"}
    missing = need - set(d.columns)
    if missing:
        sys.exit(f"per-fold file is missing columns: {sorted(missing)}")

    arm_rows, pair_rows = [], []
    for block in sorted(d.certification.unique()):
        piv = (d[d.certification == block]
               .pivot_table(index=["seed", "fold"], columns="arm", values="mean_cost"))
        present = [x for x in ARMS if x in piv.columns]

        # the identity the manuscript's table asserts
        if {"proposed", "late_sum"} <= set(piv.columns):
            gap = float(np.abs(piv["proposed"] - piv["late_sum"]).max())
            assert gap == 0.0, f"{block}: proposed and late_sum differ by {gap}"

        for arm in present:
            m, lo, hi, n = interval(piv[arm])
            arm_rows.append(dict(certification=block, arm=arm, n_folds=n,
                                 mean_cost=round(m, 6),
                                 ci_lo=round(lo, 6), ci_hi=round(hi, 6)))

        for i, x in enumerate(present):
            for y in present[i + 1:]:
                diff = piv[y] - piv[x]          # positive means x is cheaper
                m, lo, hi, n = interval(diff)
                se = diff.std(ddof=1) / np.sqrt(n) * nb_factor(n)
                # two arms with identical per-fold costs (Eq. (1) at lambda = 0 is the
                # late sum; Dempster and the late product coincide here) give se = 0
                p = 1.0 if se == 0 else 2 * stats.t.sf(abs(m / se), n - 1)
                mde = Z80 * se
                pair_rows.append(dict(
                    certification=block, cheaper_arm=x, dearer_arm=y, n_folds=n,
                    saving=round(m, 6), ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                    p_paired_t_nb=float(f"{p:.3e}"), mde=round(mde, 6),
                    nb_factor=round(nb_factor(n), 4),
                    resolvable=bool(abs(m) > mde)))

    ai = pd.DataFrame(arm_rows).sort_values(["certification", "mean_cost"])
    pt = pd.DataFrame(pair_rows).sort_values(["certification", "cheaper_arm", "dearer_arm"])
    ai.to_csv(out / "agro_arm_intervals.csv", index=False)
    pt.to_csv(out / "agro_paired_tests.csv", index=False)

    print(f"wrote {out/'agro_arm_intervals.csv'} ({len(ai)} rows)")
    print(f"wrote {out/'agro_paired_tests.csv'} ({len(pt)} rows)\n")
    own = pt[pt.certification == "own_set"]
    print("own_set, the block the manuscript's headline comes from:")
    for _, r in own.iterrows():
        if r.dearer_arm == "deep_only" or \
           (r.cheaper_arm in ("evidence_murphy", "evidence_dempster")
                and r.dearer_arm in ("late_sum", "evidence_dempster")):
            flag = "resolved" if r.resolvable else "TIE"
            print(f"  {r.cheaper_arm:18s} < {r.dearer_arm:18s} "
                  f"{r.saving:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] "
                  f"p={r.p_paired_t_nb:.1e} MDE={r.mde:.3f}  {flag}")


if __name__ == "__main__":
    main()
