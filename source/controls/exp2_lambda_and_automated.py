#!/usr/bin/env python
"""E2 and E3. Two controls on the eleven-rule tie, from the olive dumps.

E2. THE TIE AT lambda > 0. At the deployed lambda = 0 the conformal set provably cannot
change the action (Section 5.7 measures the action-change rate as exactly 0.000 for every
alpha), so the eleven rules are eleven ways of combining ONE Bayesian mass with a simple
support function, and a referee can argue the tie is entailed by the operating point rather
than measured. This runs the identical comparison at lambda in {0.25, 0.5}, where the set
does enter the objective, and reports how many pairs separate at each.

E3. THE TIE WITHOUT THE REFERRAL CONSTANT. At matched automation about 21% of images carry
an identical expert cost in every arm. That shared constant compresses the differences and
inflates the fold variance, which is what the MDE measures. `fair_fight` computes the
realised cost on the automated subset internally but does not return it; this script adds
`mean_cost_auto` by monkey-patching the return, changing nothing about how the numbers are
produced, and reports the comparison on that column too.

Neither is a new measurement of anything: both re-run released code on released dumps at
settings the manuscript already sweeps elsewhere. They exist so that "the operator is not a
lever" is stated with the two obvious alternative explanations ruled out rather than
unmentioned.

    python exp2_lambda_and_automated.py \
        --run artifacts/runs/B0__b100 --cost-model configs/cost_model.yaml \
        --out artifacts_verify/tables/tie_controls

Writes <out>_perfold.csv, <out>_pairs.csv and prints a table of separating-pair counts by
(lambda, cost column).
"""
import argparse, itertools, pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
Z80 = 2.802


def nb_factor(k, n_test, n_train):
    return float(np.sqrt((1.0 / k + n_test / n_train) / (1.0 / k)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/runs/B0__b100")
    ap.add_argument("--cost-model", default="configs/cost_model.yaml")
    ap.add_argument("--out", default="artifacts_verify/tables/tie_controls")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--lams", default="0.0,0.25,0.5")
    ap.add_argument("--automation", type=float, default=0.79)
    ap.add_argument("--expert-accuracy", type=float, default=0.93)
    ap.add_argument("--inspection-cost", type=float, default=1.0)
    a = ap.parse_args(argv)

    from olive_eswa_v3.costs_ext import AgronomicCostModel, ExpertModel
    from olive_eswa_v3.io_dumps import load_folds, calibrate_all
    from olive_eswa_v9 import fair_fight as FF

    # --- E3: expose the automated-subset cost that run_fold already computes.
    # We re-derive it from the returned bound_slack identity rather than editing the
    # module: realised_auto.mean() = mean_bound - bound_slack. That is exact, uses only
    # values fair_fight already returns, and leaves the released code untouched.
    def with_auto(df):
        df = df.copy()
        missing = {"mean_bound", "bound_slack"} - set(df.columns)
        if missing:
            sys.exit("fair_fight no longer returns %s; the identity this script uses "
                     "(realised_auto = mean_bound - bound_slack) is void." % sorted(missing))
        df["mean_cost_auto"] = df["mean_bound"] - df["bound_slack"]
        # Sanity: on a zero-diagonal cost matrix the realised cost cannot be negative.
        # NOTE it CAN exceed mean_bound. A self-published bound that is too tight has
        # negative slack by construction -- dempster, dubois_prade, pcr5 and pignistic all
        # do on these dumps -- and that is the paper's point about self-published bounds,
        # not a defect. Do not add an upper check here.
        bad = df[df.mean_cost_auto < -1e-9]
        if len(bad):
            sys.exit("realised automated cost went negative on %d rows; inspect fair_fight"
                     % len(bad))
        neg = int((df.bound_slack < 0).sum())
        if neg:
            print("[e23] %d of %d rows have negative bound slack (self-published bounds "
                  "that are violated on average); expected" % (neg, len(df)))
        return df

    fcs = calibrate_all(load_folds(a.run))
    Cm = AgronomicCostModel.from_yaml(a.cost_model).matrix()
    expert = ExpertModel(accuracy=a.expert_accuracy, inspection_cost=a.inspection_cost)
    n_test = int(np.median([len(fc.fold.test.y) for fc in fcs]))
    n_train = int(np.median([len(fc.fold.cal.y) for fc in fcs]))
    print("[e23] %d folds, n_test~%d, n_cal~%d" % (len(fcs), n_test, n_train))
    print("[e23] rules: %s" % (FF.COMPETITORS,))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    perfold, pairs = [], []
    for lam in [float(x) for x in a.lams.split(",")]:
        df = with_auto(pd.DataFrame(FF.run(fcs, Cm, expert, a.alpha, lam, a.automation)))
        df["lam"] = lam
        perfold.append(df)
        for col in ("mean_cost", "mean_cost_auto"):
            piv = df.pivot_table(index=["seed", "fold"], columns="rule", values=col)
            k = len(piv)
            infl = nb_factor(k, n_test, n_train)
            for r1, r2 in itertools.combinations(list(piv.columns), 2):
                d = piv[r1] - piv[r2]
                se = d.std(ddof=1) / np.sqrt(k) * infl
                pairs.append(dict(
                    lam=lam, column=col, rule_a=r1, rule_b=r2, n_folds=k,
                    diff=round(float(d.mean()), 6), mde=round(float(Z80 * se), 6),
                    nb_factor=round(infl, 4),
                    separates=bool(abs(d.mean()) > Z80 * se)))

    pf = pd.concat(perfold)
    pp = pd.DataFrame(pairs)
    pf.to_csv(str(out) + "_perfold.csv", index=False)
    pp.to_csv(str(out) + "_pairs.csv", index=False)

    print("\n[e23] separating pairs, out of %d" % (pp.groupby(['lam','column']).size().iloc[0]))
    print("%-6s %-16s %-10s %-9s %s" % ("lam", "column", "separates", "MDE(med)", "action-change"))
    for (lam, col), g in pp.groupby(["lam", "column"]):
        print("%-6.2f %-16s %-10s %-9.4f %s"
              % (lam, col, "%d/%d" % (g.separates.sum(), len(g)), g.mde.median(),
                 "0.000 by construction" if lam == 0 else "non-zero"))
    print("\n[e23] wrote %s_{perfold,pairs}.csv" % out)
    print("[e23] E2 is the lam rows; E3 is the mean_cost_auto rows. If the tie holds in")
    print("[e23] both, the two obvious alternative explanations are ruled out; if it does")
    print("[e23] not, the manuscript's claim is scoped to lambda = 0 with referral on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
