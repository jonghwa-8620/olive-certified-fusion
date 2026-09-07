#!/usr/bin/env python3
"""Re-count the rule-pair separations from the two control experiments, correctly.

The two control scripts (`controls/exp1_fairfight_passionfruit.py` and
`controls/exp2_lambda_and_automated.py`) wrote per-fold rows and their own pair tables.
Their pair tables are WRONG in two ways and this script supersedes them; the per-fold rows
they wrote are correct and are what this script reads.

  1. ARM MIXING. `fair_fight` emits two rows per competitor per fold -- one under the
     granted equal certificate (`arm = conformalised_competitor`) and one under the rule's
     own self-published bound (`arm = as_in_table_1`) -- but only one row for the proposed
     rule. Pivoting on `rule` alone averaged the two competitor arms while leaving the
     proposed rule single, so every competitor was compared as an average of two different
     systems. The arms genuinely differ (up to 0.159 in cost). The manuscript's comparison
     is at equal certificate, so only `conformalised_competitor` and `proposed` belong in
     it.

  2. THE NADEAU-BENGIO RATIO. The control scripts derived rho = n_test / n_train from
     IMAGE counts taken off the dumps (250/335 = 0.746), giving an inflation of 4.84. Every
     call site in the released code (`olive_eswa_v2/stats/resampling.py`, called from
     `s4_holm_ledger.py`, `exp4_showcase_e8.py`, `nb_recompute.py` and others) passes FOLD
     counts, `n_train=9, n_test=1`, so rho = 1/9 and the inflation is 2.0817 at k = 30.
     The image-count version is 2.3x too conservative. The corrected value reproduces the
     MDE the manuscript's own passionfruit table reports independently (0.089 here against
     0.099 there at miss = 7.2), which is the check that settles which is right.

Everything else -- the per-fold costs, the folds, the rules, the cost matrices -- is
unchanged and comes straight from the deposited per-fold files.

    python3 reanalyse_ties.py --tie-controls tie_controls_perfold.csv \
        --passionfruit fairfight_passionfruit_perfold.csv --out .

Writes ties_recounted.csv (one row per pair per setting) and ties_summary.csv (the counts
the manuscript quotes), and prints both.
"""
import argparse, itertools, pathlib, sys
import numpy as np
import pandas as pd

Z80 = 2.802
EQUAL_CERT = ("conformalised_competitor", "proposed")


def nb_inflation(k, n_train=9, n_test=1):
    """The released convention: variance (1/k + n_test/n_train) instead of 1/k."""
    return float(np.sqrt((1.0 / k + n_test / n_train) / (1.0 / k)))


def pairs(df, value="mean_cost"):
    df = df[df["arm"].isin(EQUAL_CERT)]
    piv = df.pivot_table(index=["seed", "fold"], columns="rule", values=value)
    k = len(piv)
    infl = nb_inflation(k)
    out = []
    for a, b in itertools.combinations(list(piv.columns), 2):
        d = piv[a] - piv[b]
        se = d.std(ddof=1) / np.sqrt(k) * infl
        mde = Z80 * se
        out.append(dict(rule_a=a, rule_b=b, n_folds=k, nb_inflation=round(infl, 4),
                        diff=round(float(d.mean()), 6),
                        ci_lo=round(float(d.mean() - 1.96 * se), 6),
                        ci_hi=round(float(d.mean() + 1.96 * se), 6),
                        mde=round(float(mde), 6),
                        separates=bool(abs(d.mean()) > mde)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tie-controls", default="tie_controls_perfold.csv")
    ap.add_argument("--passionfruit", default="fairfight_passionfruit_perfold.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows, summ = [], []

    p = pathlib.Path(a.tie_controls)
    if p.exists():
        d = pd.read_csv(p)
        d["mean_cost_auto"] = d["mean_bound"] - d["bound_slack"]
        for lam in sorted(d.lam.unique()):
            for col in ("mean_cost", "mean_cost_auto"):
                r = pairs(d[d.lam == lam], col)
                r.insert(0, "setting", f"olive lam={lam} {col}")
                rows.append(r)
                w = r[(r.rule_a == "proposed_hurwicz") | (r.rule_b == "proposed_hurwicz")]
                summ.append(dict(corpus="olive", knob=f"lam={lam}", column=col,
                                 pairs_total=len(r), pairs_separating=int(r.separates.sum()),
                                 mde_median=round(float(r.mde.median()), 4),
                                 vs_proposed_total=len(w),
                                 vs_proposed_tied=int((~w.separates).sum()),
                                 vs_proposed_separating=int(w.separates.sum())))
    else:
        print(f"[warn] {p} not found, skipping the olive controls", file=sys.stderr)

    p = pathlib.Path(a.passionfruit)
    if p.exists():
        d = pd.read_csv(p)
        for miss in sorted(d.miss.unique()):
            r = pairs(d[d.miss == miss], "mean_cost")
            r.insert(0, "setting", f"passionfruit miss={miss} mean_cost")
            rows.append(r)
            w = r[(r.rule_a == "proposed_hurwicz") | (r.rule_b == "proposed_hurwicz")]
            summ.append(dict(corpus="passionfruit", knob=f"miss={miss}", column="mean_cost",
                             pairs_total=len(r), pairs_separating=int(r.separates.sum()),
                             mde_median=round(float(r.mde.median()), 4),
                             vs_proposed_total=len(w),
                             vs_proposed_tied=int((~w.separates).sum()),
                             vs_proposed_separating=int(w.separates.sum())))
    else:
        print(f"[warn] {p} not found, skipping passionfruit", file=sys.stderr)

    if not rows:
        sys.exit("no input found")

    allp = pd.concat(rows, ignore_index=True)
    s = pd.DataFrame(summ)
    allp.to_csv(out / "ties_recounted.csv", index=False)
    s.to_csv(out / "ties_summary.csv", index=False)

    pd.set_option("display.width", 160)
    print(s.to_string(index=False))
    print(f"\nwrote {out/'ties_recounted.csv'} and {out/'ties_summary.csv'}")
    print("\nThe manuscript quotes the vs_proposed columns. The pairs_separating column is a")
    print("different statistic -- every ordered pair of the eleven, including competitor")
    print("against competitor -- and the manuscript makes no claim about it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
