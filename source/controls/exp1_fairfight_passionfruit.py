#!/usr/bin/env python
"""E1. The eleven-rule equal-certificate comparison, on the passionfruit corpus.

The manuscript's central negative result --- nine of eleven rules tie on cost once each is
granted the same certificate --- is measured on olive, where the rule-pair resolution is
MDE = 0.098. Supplementary Section S0.3 already reports that the passionfruit corpus, with
3.6x the images, resolves to MDE = 0.032 and separates 24 of its 28 rule pairs. A referee
will ask what the eleven-rule table looks like there, and the answer decides whether the
tie is a property of the operators or of this design's resolution. This script runs it.

It reuses `olive_eswa_v9.fair_fight` unchanged --- the same code that produced the
manuscript's Table 10 --- and supplies only the corpus's class list and cost matrix, so a
difference in the output is a difference in the corpus and not in the analysis. The frame
size is a package-level constant and `olive_eswa_v6.mass` builds its 2^K tables at import
time, so CLASSES is rebound BEFORE mass is first imported, exactly as
`corpus_certificate.py` does.

The cost matrix is instantiated from olive's own three declared parameters with ONE miss
cost shared by every disease, and the run is repeated at the three miss costs olive's
matrix instantiates (1.6, 7.2, 21.2). This is the machinery under a stated cost structure,
not a claim about passionfruit agronomy.

    python exp1_fairfight_passionfruit.py \
        --run artifacts/runs/passionfruit__probe \
        --out artifacts_verify/tables/fairfight_passionfruit

Writes <out>_perfold.csv, <out>_summary.csv and <out>_pairs.csv, the last being the
55 rule pairs per miss cost (11 arms; the 28 of Supplementary Section S0.3 is the
eight-rule sub-family that section compares) with the paired per-fold difference, its Nadeau-Bengio
inflated interval, the MDE and whether the pair separates --- the same convention the
manuscript uses everywhere else.
"""
import argparse, itertools, json, pathlib, sys
import numpy as np

sys.path.insert(0, ".")

SPRAY, WRONG_MULT, RESIDUAL, ORGAN_CONF = 3.0, 1.5, 0.6, 0.5
MISS_GRID = (1.6, 7.2, 21.2)
Z80 = 2.802


def build_C(classes, miss):
    """Olive's declared parameters, one miss cost for every disease class."""
    K = len(classes)
    healthy = tuple(i for i, c in enumerate(classes)
                    if c.endswith("_healthy") or c.endswith("_normal"))
    diseased = tuple(i for i in range(K) if i not in healthy)
    C = np.zeros((K, K), float)
    for y in range(K):
        for a in range(K):
            if y == a:
                continue
            if y in healthy and a in healthy:
                C[y, a] = ORGAN_CONF
            elif y in healthy:
                C[y, a] = SPRAY
            elif a in healthy:
                C[y, a] = miss
            else:
                C[y, a] = SPRAY * WRONG_MULT + RESIDUAL * miss
    return C, healthy, diseased


def nb_factor(k, n_test, n_train):
    return float(np.sqrt((1.0 / k + n_test / n_train) / (1.0 / k)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/runs/passionfruit__probe")
    ap.add_argument("--out", default="artifacts_verify/tables/fairfight_passionfruit")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--automation", type=float, default=0.79)
    ap.add_argument("--expert-accuracy", type=float, default=0.93)
    ap.add_argument("--inspection-cost", type=float, default=1.0)
    ap.add_argument("--miss-grid", default=",".join(str(m) for m in MISS_GRID))
    a = ap.parse_args(argv)

    meta = json.loads((pathlib.Path(a.run) / "corpus_meta.json").read_text())
    classes = meta["classes"]
    _, healthy, diseased = build_C(classes, 1.0)
    print("[e1] %s" % a.run)
    print("[e1] %d classes: %s" % (len(classes), classes))

    import olive_eswa_v3 as V3, olive_eswa_v6 as V6
    for mod in (V3, V6):
        mod.CLASSES = tuple(classes)
        mod.HEALTHY = healthy
        mod.DISEASED = diseased
        if hasattr(mod, "K_CLASSES"):
            mod.K_CLASSES = len(classes)
    V6.K_CLASSES = len(classes)
    print("[e1] frame rebound to K=%d before mass import" % len(classes))

    import pandas as pd
    import olive_eswa_v3.costs_ext as CE
    from olive_eswa_v3.costs_ext import ExpertModel
    from olive_eswa_v3.io_dumps import load_folds, calibrate_all
    from olive_eswa_v6 import mass as MASS
    from olive_eswa_v9 import fair_fight as FF
    CE.CLASSES, CE.HEALTHY, CE.DISEASED = tuple(classes), healthy, diseased
    assert MASS.N_SUBSETS == 1 << len(classes)
    print("[e1] mass power set = %d subsets  OK" % MASS.N_SUBSETS)
    print("[e1] rules: %s" % (FF.COMPETITORS,))

    fcs = calibrate_all(load_folds(a.run))
    expert = ExpertModel(accuracy=a.expert_accuracy, inspection_cost=a.inspection_cost)
    n_test = int(np.median([len(fc.fold.test.y) for fc in fcs]))
    n_train = int(np.median([len(fc.fold.cal.y) for fc in fcs]))
    print("[e1] %d folds, n_test~%d, n_cal~%d" % (len(fcs), n_test, n_train))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    perfold, summ, pairs = [], [], []
    for miss in [float(x) for x in a.miss_grid.split(",")]:
        C, _, _ = build_C(classes, miss)
        rows = FF.run(fcs, C, expert, a.alpha, a.lam, a.automation)
        df = pd.DataFrame(rows)
        df["miss"] = miss
        perfold.append(df)

        g = df.groupby("rule").agg(
            cost=("mean_cost", "mean"), cost_sd=("mean_cost", "std"),
            missed=("missed_disease_rate", "mean"),
            violation=("bound_violation", "mean"),
            set_size=("mean_set_size", "mean"),
            coverage=("coverage", "mean"), n_folds=("mean_cost", "size")).reset_index()
        g["miss"] = miss
        summ.append(g)

        piv = df.pivot_table(index=["seed", "fold"], columns="rule", values="mean_cost")
        k = len(piv)
        infl = nb_factor(k, n_test, n_train)
        rules = list(piv.columns)
        for r1, r2 in itertools.combinations(rules, 2):
            d = piv[r1] - piv[r2]
            se = d.std(ddof=1) / np.sqrt(k) * infl
            mde = Z80 * se
            pairs.append(dict(
                miss=miss, rule_a=r1, rule_b=r2, n_folds=k,
                diff=round(float(d.mean()), 6),
                ci_lo=round(float(d.mean() - 1.96 * se), 6),
                ci_hi=round(float(d.mean() + 1.96 * se), 6),
                mde=round(float(mde), 6), nb_factor=round(infl, 4),
                separates=bool(abs(d.mean()) > mde)))
        sep = sum(1 for p in pairs if p["miss"] == miss and p["separates"])
        tot = sum(1 for p in pairs if p["miss"] == miss)
        print("[e1] miss=%5.2f  MDE(median)=%.4f  separates %d/%d pairs"
              % (miss, np.median([p["mde"] for p in pairs if p["miss"] == miss]), sep, tot))

    pd.concat(perfold).to_csv(str(out) + "_perfold.csv", index=False)
    pd.concat(summ).to_csv(str(out) + "_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(str(out) + "_pairs.csv", index=False)
    print("\n[e1] wrote %s_{perfold,summary,pairs}.csv" % out)
    print("[e1] the number the manuscript needs is the separates/total line above,")
    print("[e1] beside olive's 'nine of eleven tie' at MDE = 0.098.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
