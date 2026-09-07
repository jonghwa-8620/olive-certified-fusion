# Source archive — Sections 5.2 and 5.4 of the manuscript, and Supplementary Section S24

(Headings below that read "Sec. 7.1.2" / "Sec. 7.1.3" are the numbering of the long
version this code was written for; in the submitted manuscript they are Section 5.4
(the leverage threshold and LACF-v2) and Section 5.2 (what fusion buys, source counts).
Supplementary Section S0 carries the full renumbering map. The equal-certificate comparison
of eleven rules, which these notes call "the fair fight", is Section 5.3 and Table 8 in the
submitted manuscript; `controls/exp2_lambda_and_automated.py` and `tie_controls/reanalyse_ties.py`
are what produce and recount it, and the manuscript's counts are those of `ties_summary.csv`:
seven of the ten competitors are not separated from the proposed rule at lambda = 0, and three
are - EDL, TMC, and pignistic-only, the last in the competitor's favour.)

This archive contains the code behind Section 7.1.2 of the manuscript and Supplementary
Section S24. It is the falsification-attempt harness only; the certified-acquisition pipeline
of Sections 3-6 is released separately.

## Layout

    olive_if_v17/          shared machinery: the eleven fixed fusion operators plus the learned gated mixture, the
                           CertifiedScorer decision path (temperature -> LAC set ->
                           expected-cost action with priced referral -> direct cost
                           quantile), the cost matrix, dump loading
    olive_if_v18/          LACF-v2 itself
      lacf_v2.py           the model, the sliver, the thirteen-arm comparison, the ablation
      theorem9_lacf.py     the exploratory branch's statement and its three machine checks
                           (NOT claimed in the manuscript -- see Supp. S24)
      run_all_v18.py       driver; writes run_manifest_v18.json beside every output
    cms_order.py           the random-subset sweep behind Sec. 7.1.3's source-count curve
    agro_intervals.py      intervals and paired tests for the deep + agronomic-descriptor
                           fusion table, from the per-fold file deposited under
                           results/agro/. Cost only; see that folder's README
    olive_if_v19/          the source-count measurement of Sec. 7.1.3
      certified_multisource.py  Eq. (1) fed by J fused sources; sweeps J and compares
                           against every operator and the p-value merges. NOTE: the
                           product-rule fusion at J = 8 IS `product_rule`; this module
                           sweeps the evidence count, it does not add an operator
      decal.py             action-cell decision calibration (Zhao et al. 2021), reported
                           as its own arm and not an improvement on these dumps. Its cell
                           partition is by treatment action (K cells); the referral action
                           is not a cell, so the referral ranking is not recalibrated. The
                           run()/summarise()/_latex() pipeline in this file is unused by
                           the manuscript -- only the DECAL class is.
    configs/cost_model.yaml  the agronomic cost model: ten parameters, each with a value,
                           a [low, high] range used by the joint Monte-Carlo sweep, and a
                           `source` field that either cites the olive-pathology or market
                           literature or declares the parameter an operational scenario
                           assumption. Supplementary Section S25 tabulates it. No formal
                           multi-expert elicitation was run and the file says so.
    controls/              two controls on the eleven-rule tie, added after review. Both
                           re-run released code on released dumps; neither changes how any
                           number is produced.
      exp1_fairfight_passionfruit.py
                           the eleven-rule equal-certificate comparison on the passionfruit
                           corpus, whose rule-pair resolution is 0.032 against olive's
                           0.098. Reuses olive_eswa_v9.fair_fight unchanged and supplies
                           only the class list and cost matrix, rebinding the frame size
                           before olive_eswa_v6.mass builds its 2^K tables, exactly as
                           corpus_certificate.py does. Outputs go to results/tie_controls/
                           as fairfight_passionfruit_{perfold,summary,pairs}.csv.
      exp2_lambda_and_automated.py
                           the same comparison at lambda in {0, 0.25, 0.5}, where the
                           conformal set does enter the objective, and on the automated
                           subset with the shared referral cost removed. The automated-
                           subset cost is recovered from values fair_fight already returns
                           (realised_auto = mean_bound - bound_slack) with a guard that
                           aborts if that identity ever stops holding, so the released
                           module is not edited. Outputs go to results/tie_controls/.
      reanalyse_ties.py    re-aggregates the two exp1/exp2 pair tables after two defects
                           found in them: the Nadeau-Bengio ratio had been formed from
                           image counts (4.8189) instead of the fold counts every released
                           call site passes (n_train=9, n_test=1, giving 2.0817 at k=30),
                           and the competitor arms `conformalised_competitor` and
                           `as_in_table_1` had been pooled although they differ by up to
                           0.159. It reads the *_pairs.csv marked _SUPERSEDED and writes
                           results/tie_controls/ties_recounted.csv and ties_summary.csv,
                           which are the numbers the manuscript quotes. A byte-identical
                           copy sits beside its outputs in results/tie_controls/.
      exp3_emptyset_counterfactual.py
                           the equal-certificate table re-scored under all three empty-set
                           conventions of emptyset/bound_audit.py. Answers whether the
                           EUR 0.057 gap between the proposed and pignistic arms is a
                           property of the certificate or of the convention that refers an
                           image whose set is empty. It is the convention: under either
                           non-referring convention the two arms are the same policy and
                           the paired difference is exactly 0.000 on all thirty folds.
                           Outputs go to results/emptyset/ as emptyset_cf_{perfold,
                           summary,pairs}.csv. A byte-identical copy sits beside its
                           outputs there.
    experiments/           the sweeps of Sec. 7.1.2, one script per sweep
    patches/              idempotent patches applied to the harness during the revision,
                           each with --check and --revert

## The units of the sliver — read this first

Theorem 4 defines the perturbation and the margin on **expected costs**:

    J(a | x)  = sum_y p(y | x) C[y, a]
    eps(x)    = max_a |J_1(a | x) - J_2(a | x)|
    margin(x) = second-smallest J(a | x) - smallest J(a | x)
    S         = { x : margin(x) <= 2 eps(x) }

The first version of this harness computed `margin` as the gap between the two largest
posterior probabilities and `eps` as the mean total-variation distance between view
posteriors and the base. Both are probability-scale quantities; the cost matrix never
entered. The set they define is not the sliver of Theorem 4, and on the olive corpus the two
differ by a great deal (pi_S 0.094 against 0.602; P(R|S) 1.000 against 0.347). Supplementary
Section S24 documents the defect, what it did to the first version of Section 7.1.2, and how
it was found.

`patches/patch_costunits.py` is the correction. It adds a module global `UNITS` to
`olive_if_v18/lacf_v2.py` with two settings:

    UNITS = "prob"   the original, defective, probability-scale definition
    UNITS = "cost"   Theorem 4's own definition

The default is `"prob"` so that the earlier deposited tables reproduce byte-for-byte. Every
number in the manuscript comes from `"cost"`. Select it with `--units cost` on the driver, or
with the environment variable `LACF_UNITS=cost`, which the standalone sweep scripts also read:

    export LACF_UNITS=cost
    python -m olive_if_v18.run_all_v18 --views <dump dirs> --base product_rule --units cost --out main

`olive_if_v17/sota_dl_comparison.py:leverage_mask` still carries the defective definition and
a docstring saying so. It is kept only so that the earlier v17 tables reproduce; nothing in
the manuscript is computed with it.

## Two other things a reader will run into

**The base operator.** `features()` originally hard-coded the sum rule as `p_base`. Because
the model is *defined* as `p_base` off the sliver, the hypothesis class could not express any
operator better than the sum rule, which is ninth of thirteen. A `--base` argument was added;
the pre-patch originals of the three affected files are shipped under
`patches/pre_base_patch/` so the change can be diffed. The manuscript uses `product_rule`,
the cheapest fixed rule on these folds -- which is itself a selection, and Supplementary
Section S24 states the bias it introduces.

**A degenerate baseline.** `qmf` weights views by a free-energy quality score computed on
logits. The released artifacts give posteriors, on which that energy is identically zero, so
this decision-level surrogate of QMF *is* the sum rule (agreement 1e-16). Supplementary
Table S32 prints the two as one row rather than as two baselines. The other eleven operators
are decision-level surrogates too, operating on released posteriors rather than on each
method's own training pipeline; `sota_dl_comparison.py` says so in its docstring.

**"Theorem 9".** `olive_if_v18/theorem9_lacf.py` states and machine-checks a claim from the
exploratory branch. **That claim is not made in the manuscript** -- the experiment refuted
it -- and the statement is deliberately absent from the paper. It is kept in the archive
rather than deleted, with both the original and the corrected versions of its third check.
See Supplementary Section S24.

## Requirements

Python 3.11, numpy, scipy, pandas, matplotlib, and `autograd` for exact gradients (finite differences are
used automatically if it is absent, at a large cost in time). No GPU. The full set of sweeps
runs in roughly two hours on two cores.

## Provenance

Every driver run writes `run_manifest_v18.json` into its output directory: the absolute view
paths, a digest over each view's file names, sizes and modification times (metadata, not file
content), a content hash of each view's image-index array (so that a mixed corpus is
detectable after the fact), the base operator, the units, the library versions, the host and
the UTC timestamp. L-BFGS paths differ slightly across BLAS and Python builds -- we measured
+/-0.002 on the ablation costs between Linux/py3.11 and Windows/py3.10, an order below the
design's MDE -- so compare tables only within one manifest.
