#!/usr/bin/env python
"""E4. Does the empty-set convention manufacture the EUR 0.057 gap?

THE OBJECTION. Table 10 reports the pignistic arm EUR 0.057 cheaper than the proposed
rule at equal certificate, and Section 5.3 explains the gap mechanically: the proposed
rule's certified set is the narrowest in the table (1.75 against 2.13-2.21), a narrower
set is empty more often (4.8% against at most 0.4%), the deployed convention refers every
image whose set is empty, and referral is what costs money (automation 0.744 against
0.792).  A referee will answer that the paper CHOSE the convention that turns an empty set
into a referral, and that under either of the two other conventions it reports in Section
5.1 an empty set does not force a referral at all -- so automation returns to 0.79 for
every rule, the two arms are already indistinguishable on the images both automate
(-0.009 against a resolution of 0.037), and the gap has nothing left to come from.  If
that is right, "the price of a sharper certificate" is a property of an accounting choice
and not of the certificate.

WHAT THIS RUNS. The identical equal-certificate comparison of `olive_eswa_v9.fair_fight`,
at the deployed lambda = 0, scored under all three empty-set conventions of
`emptyset/bound_audit.py` -- the same three names, the same semantics:

    refer      an empty set is referred and leaves the automated denominator.
               This is the deployed convention and reproduces Table 10.
    vacuous    an empty set is automated; its published bound is the vacuous one,
               which is what `mass.conformal_bba` already emits for it (m(Theta) = 1),
               so this convention differs from `refer` ONLY in the referral.
    strict     an empty set is automated and counted as an immediate violation.
               Cost and automation are identical to `vacuous`; only the violation
               column differs.  Reported so the conservative reading is on the record.

Nothing about any rule's DECISION changes under any convention: the action is the same
argmin of the same objective.  What changes is which images are handed to the expert.

The released modules are imported unchanged and are not edited.  The two scoring
functions are re-implemented here with a `convention` argument because that argument is
the experiment; every other line is the released line, and the `refer` column is the
self-check that proves it (it must reproduce the deposited per-fold table).

    python exp3_emptyset_counterfactual.py \
        --run artifacts/runs/B0__b100 --cost-model configs/cost_model.yaml \
        --out artifacts_verify/tables/emptyset_cf

Writes <out>_perfold.csv, <out>_summary.csv and <out>_pairs.csv, and prints the one line
the manuscript needs: the proposed-minus-pignistic paired difference and its
Nadeau-Bengio minimum detectable effect under each convention.
"""
import argparse, itertools, pathlib, sys, time
import numpy as np

sys.path.insert(0, ".")

CONVENTIONS = ("refer", "vacuous", "strict")
Z80 = 2.802


def nb_inflation(k, n_train=9, n_test=1):
    """The factor the released `stats.resampling` applies, at fold counts.

    Every call site in the released package passes n_train=9, n_test=1 (ten folds, one
    held out), so rho = 1/9 and the factor at k = 30 is 2.0817.  Image counts give 4.84
    and are wrong; `reanalyse_ties.py` documents that defect.
    """
    return float(np.sqrt((1.0 / k + n_test / n_train) / (1.0 / k)))


def score_arm(ev, sets_for_bound, obj, bound, C, expert, automation, convention,
              missed_disease, realised_cost):
    """One rule, one convention.  Released logic with the empty-set branch exposed.

    `obj` and `bound` are whatever the rule's own released path produced; this function
    only decides who is referred and how empty automated rows enter the violation count.
    """
    C = np.asarray(C, float)
    n = obj.shape[0]
    rows = np.arange(n)
    a_star = obj.argmin(1)
    j_star = obj[rows, a_star]
    b_star = bound[rows, a_star].astype(float).copy()

    ref_cost = ev.probs @ expert.expected_cost_given_truth(C)
    voi = j_star - ref_cost
    n_ref = int(round((1.0 - automation) * n))
    base_refer = np.zeros(n, bool)
    base_refer[np.argsort(-voi)[:n_ref]] = True

    empty = ~sets_for_bound.any(1)

    if convention == "refer":
        refer = base_refer | empty
    else:
        refer = base_refer.copy()
        if convention == "strict":
            # any realised cost exceeds -inf, so the row counts as a violation
            b_star[empty] = -np.inf
        # `vacuous`: conformal_bba already maps the empty set to m(Theta) = 1, so
        # b_star[empty] is already the vacuous bound.  Nothing to change.

    action = np.where(refer, -1, a_star)
    cost = realised_cost(ev.y, action, C, expert)
    auto = ~refer
    if not auto.any():
        return None
    realised_auto = C[ev.y[auto], action[auto]]
    finite = np.isfinite(b_star[auto])
    return {
        "convention": convention,
        "mean_cost": float(cost.mean()),
        "missed_disease_rate": float(missed_disease(ev.y, action).mean()),
        "automation": float(auto.mean()),
        "bound_violation": float((realised_auto > b_star[auto] + 1e-9).mean()),
        "mean_bound": float(np.mean(b_star[auto][finite])) if finite.any() else np.nan,
        "mean_set_size": float(sets_for_bound.sum(1).mean()),
        "empty_set_rate": float(empty.mean()),
        "coverage": float(sets_for_bound[rows, ev.y].mean()),
        "n_automated": int(auto.sum()),
        "n": n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/runs/B0__b100")
    ap.add_argument("--cost-model", default="configs/cost_model.yaml")
    ap.add_argument("--out", default="artifacts_verify/tables/emptyset_cf")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--automation", type=float, default=0.79)
    ap.add_argument("--expert-accuracy", type=float, default=0.93)
    ap.add_argument("--inspection-cost", type=float, default=1.0)
    a = ap.parse_args(argv)

    import pandas as pd
    from olive_eswa_v3.costs_ext import (AgronomicCostModel, ExpertModel,
                                         missed_disease, realised_cost)
    from olive_eswa_v3.io_dumps import load_folds, calibrate_all
    from olive_eswa_v6.sources import build_evidence
    from olive_eswa_v6.fusion_baselines import rule_scores
    from olive_eswa_v6 import mass as M
    from olive_eswa_v9 import fair_fight as FF
    from olive_eswa_v9 import split_conf

    fcs = calibrate_all(load_folds(a.run))
    C = AgronomicCostModel.from_yaml(a.cost_model).matrix()
    expert = ExpertModel(accuracy=a.expert_accuracy,
                         inspection_cost=a.inspection_cost)
    print("[e4] %d folds, rules: %s" % (len(fcs), (FF.COMPETITORS,)))
    print("[e4] conventions: %s" % (CONVENTIONS,))
    print("[e4] lambda=%.2f alpha=%.2f automation=%.3f" % (a.lam, a.alpha, a.automation))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    perfold = []
    t0 = time.time()
    for i, fc in enumerate(fcs):
        i_a, i_b = split_conf(fc.i_conf, fc.fold.seed, fc.fold.fold)
        fc_a = FF._rebind(fc, i_a)
        ev_test = build_evidence(fc_a, alpha=a.alpha, mondrian=True)
        ev_calb = FF._evidence_on(fc_a, i_b, alpha=a.alpha, mondrian=True)

        # --- the proposed arm: its certificate is S2 itself
        s = rule_scores(ev_test, "proposed_hurwicz", C, a.lam)
        for conv in CONVENTIONS:
            r = score_arm(ev_test, ev_test.sets, s["objective"], s["bound"], C, expert,
                          a.automation, conv, missed_disease, realised_cost)
            if r is None:
                continue
            r.update({"rule": "proposed_hurwicz", "arm": "proposed",
                      "seed": fc.fold.seed, "fold": fc.fold.fold})
            perfold.append(r)

        # --- every competitor, conformalised on the disjoint block
        for rule in FF.COMPETITORS:
            s_cal = FF.rule_class_scores(ev_calb, rule)
            s_test = FF.rule_class_scores(ev_test, rule)
            sets_rule = FF.conformalise(s_cal, ev_calb.y, s_test, a.alpha, True)
            obj = rule_scores(ev_test, rule, C, a.lam)["objective"]
            bound = M.upper_expected_cost(M.conformal_bba(sets_rule, ev_test.alpha), C)
            for conv in CONVENTIONS:
                r = score_arm(ev_test, sets_rule, obj, bound, C, expert,
                              a.automation, conv, missed_disease, realised_cost)
                if r is None:
                    continue
                r.update({"rule": rule, "arm": "conformalised_competitor",
                          "seed": fc.fold.seed, "fold": fc.fold.fold})
                perfold.append(r)

        el = time.time() - t0
        print("[e4] fold %2d/%d  %5.1fs elapsed, ~%5.1fs remaining"
              % (i + 1, len(fcs), el, el / (i + 1) * (len(fcs) - i - 1)), flush=True)

    pf = pd.DataFrame(perfold)
    pf.to_csv(str(out) + "_perfold.csv", index=False)

    g = (pf.groupby(["convention", "rule"])
           .agg(cost=("mean_cost", "mean"),
                automation=("automation", "mean"),
                violation=("bound_violation", "mean"),
                missed=("missed_disease_rate", "mean"),
                set_size=("mean_set_size", "mean"),
                empty_rate=("empty_set_rate", "mean"),
                n_folds=("mean_cost", "size")).reset_index())
    g.to_csv(str(out) + "_summary.csv", index=False)

    # --- the ten comparisons with the proposed rule, under each convention
    pairs = []
    for conv in CONVENTIONS:
        d = pf[pf.convention == conv]
        piv = d.pivot_table(index=["seed", "fold"], columns="rule", values="mean_cost")
        if "proposed_hurwicz" not in piv.columns:
            continue
        k = len(piv)
        infl = nb_inflation(k)
        for rule in piv.columns:
            if rule == "proposed_hurwicz":
                continue
            dd = (piv[rule] - piv["proposed_hurwicz"]).dropna()
            se = dd.std(ddof=1) / np.sqrt(len(dd)) * infl
            pairs.append(dict(convention=conv, competitor=rule, n_folds=len(dd),
                              delta=round(float(dd.mean()), 6),
                              mde=round(float(Z80 * se), 6),
                              nb_factor=round(infl, 4),
                              separates=bool(abs(dd.mean()) > Z80 * se)))
    pp = pd.DataFrame(pairs)
    pp.to_csv(str(out) + "_pairs.csv", index=False)

    # ------------------------------------------------------------------ the answer ---
    print("\n" + "=" * 78)
    print("[e4] SELF-CHECK. Under `refer` these must match Table 10 of the manuscript:")
    ref = g[g.convention == "refer"].set_index("rule")
    for r in ("proposed_hurwicz", "pignistic_only"):
        if r in ref.index:
            print("[e4]   %-18s cost %.4f  automation %.4f  (Table 10: %s)"
                  % (r, ref.loc[r, "cost"], ref.loc[r, "automation"],
                     "0.634 / 0.744" if r == "proposed_hurwicz" else "0.577 / 0.792"))
    print("=" * 78)
    print("[e4] THE COUNTERFACTUAL. proposed vs pignistic, by convention:")
    print("%-10s %10s %10s %9s %9s  %s"
          % ("convention", "delta", "MDE", "autom_p", "autom_q", "verdict"))
    for conv in CONVENTIONS:
        row = pp[(pp.convention == conv) & (pp.competitor == "pignistic_only")]
        gg = g[g.convention == conv].set_index("rule")
        if row.empty:
            continue
        row = row.iloc[0]
        print("%-10s %+10.4f %10.4f %9.4f %9.4f  %s"
              % (conv, row.delta, row.mde,
                 gg.loc["proposed_hurwicz", "automation"],
                 gg.loc["pignistic_only", "automation"],
                 "SEPARATES" if row.separates else "tie"))
    print("=" * 78)
    print("[e4] separating competitors out of ten, by convention:")
    for conv in CONVENTIONS:
        s = pp[pp.convention == conv]
        print("[e4]   %-10s %d/%d" % (conv, int(s.separates.sum()), len(s)))
    print("\n[e4] READ IT LIKE THIS.")
    print("[e4]   If the gap SURVIVES under `vacuous`, the referee's objection fails and")
    print("[e4]   the manuscript can say so: the EUR 0.057 is not an artefact of the")
    print("[e4]   convention, and Section 5.3 stands as written.")
    print("[e4]   If the gap DISAPPEARS under `vacuous`, the mechanism paragraph is")
    print("[e4]   correct but the *price* framing is convention-dependent, and the")
    print("[e4]   manuscript must scope it to the deployed convention explicitly.")
    print("[e4] wrote %s_{perfold,summary,pairs}.csv" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
