"""SUPERSEDED DRAFTING TOOL — DO NOT READ THE STRINGS BELOW AS CLAIMS.

This module holds the abstract, contribution and highlight text drafted while the
v18 branch was attempting to ESTABLISH that a leverage-restricted operator beats
every global operator. The experiment of Sec. 7.1.2 REFUTED that. None of the
sentences below is claimed by the manuscript; the "Theorem 9" they reference is
deliberately absent from it. They are kept only so the gate in _gate_sentence()
can be re-run and seen to refuse them, and because the archive is released as it
was used. See Supplementary Section S24.

Change the paper's central claim to the one LACF-v2 carries, consistently.

Old thesis (v16 abstract, §1, §12): "the combination operator is not a lever
but a residual"; "nine of eleven rules tie"; certification and acquisition
are the levers.
New thesis (v18): the operator *is* a lever exactly on the high-leverage
sliver S; a sliver-restricted, certified learner (LACF-v2) improves realised
cost over every global operator with the same certificate (Theorem 9); the
global operators tie because they spend capacity where Theorem 4 forbids any
gain.  Certification and acquisition remain levers — the paper adds a third.

What this module does
  1. Reads the measured tables (``lacf_main_v18.csv``, ``lacf_ablation_v18.csv``,
     ``theorem9_checks_v18.csv``) and refuses to write a claim the tables do
     not support: if the improvement gate is not met, or the provenance is a
     smoke run, the blocks are written with a ``%% NOT SUPPORTED`` header and a
     tie-honest alternative paragraph, and the audit fails.
  2. Writes ``claim_blocks_v18.tex``: abstract (IF genre, no numerals),
     contributions (four "X gives Y" bullets, no bold), the thesis paragraph
     that replaces "not a lever", a new subsection for §5.2 with the table
     references, a two-paragraph conclusion, and highlights.
  3. Scans the manuscript (.tex, or the PDF text) for every sentence that
     still asserts the old thesis and writes ``claim_conflicts_v18.csv`` with a
     suggested replacement; ``--apply`` rewrites the exact matches in the .tex
     and leaves the rest (with line numbers) for the authors.
  4. Appends the new rows to the claims ledger (``claims_ledger_v18.csv``):
     LACF-v2, Theorem 9, and the superseded row for "operator is a residual"
     with its new scope ("global operators only").

    python -m olive_if_v18.claim_rewrite --artifacts artifacts_v18 [--tex main.tex | --pdf manuscript.pdf] \
        [--ledger results/artifacts_v4/tables/claims_ledger.csv] [--apply]
"""
from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd

from olive_if_v17._common17 import Check, pdf_text, write_checks, write_table

OLD_THESIS = [
    (r"the combination operator is not a lever[^.]*\.", "The combination operator is a lever only on the high-leverage sliver that Theorem~4 identifies, and Theorem~9 shows how to pull it."),
    (r"not a lever but a residual[^.]*\.", "a lever only on the high-leverage sliver, and a residual elsewhere (Theorem~9)."),
    (r"[Nn]ine of (?:the )?eleven[^.]*tie[^.]*\.", "Nine of eleven global operators tie, as Theorem~4 predicts, and the sliver-restricted learner of Section~5.2.2 is the one rule that does not."),
    (r"the choice of combination operator is a residual[^.]*\.", "the choice of a global combination operator is a residual, while the sliver-restricted operator of Theorem~9 is a lever."),
    (r"at this corpus's leverage the operator cannot matter[^.]*\.", "at this corpus's leverage a global operator cannot matter, and a sliver-restricted one can matter by at most $\\Lambda$ — which LACF-v2 realises in part."),
    (r"What is not a lever[^.]*is the combination operator[^.]*\.", "A global combination operator is not a lever; a sliver-restricted one is, and Section~5.2.2 measures how much."),
    (r"the operator is not the lever", "a global operator is not the lever"),
    (r"whether fusion is worth doing", "where fusion is worth doing"),
]

# [REFUTED - SUPERSEDED DRAFT, NOT A CLAIM OF THIS PAPER]
ABSTRACT = (
    "Agricultural triage systems must act on heterogeneous evidence about a common label space—calibrated posteriors, "
    "set-valued conformal predictions, open-world scores, deployment priors and priced human opinions—under a finite "
    "daily inspection budget. However, existing evidence-fusion pipelines treat every input alike: a combination "
    "operator is applied globally, the miscoverage a certificate may spend is fixed uniformly, and the cost of acquiring "
    "human evidence is ignored, so the fused belief carries no distribution-free warrant and the operator's capacity is "
    "spent where it cannot change a decision. To address these limitations, we propose {name}, a novel leverage-aware "
    "certified fusion framework that prices miscoverage and acquisition with one program and learns the fusion operator "
    "only where a pre-hoc leverage bound shows it can matter. Specifically, the Cost-Certificate Calculus (CCC) "
    "conformalises the realised cost of the fused action and is valid, minimal in its class and inherited by every "
    "downstream rule. Additionally, the Certified Acquisition Program (CAP) allocates miscoverage across strata and "
    "agronomist minutes across images through one knapsack whose dual prices couple at the optimum. Moreover, "
    "Leverage-Aware Certified Fusion (LACF) restricts a quality-aware learned operator to the high-leverage inputs and "
    "is the certified default elsewhere. Theoretically, validity, inheritance, the gain bound and a restricted "
    "Rademacher bound are formally proven. Empirically, extensive experiments on {n_datasets} crop-image datasets "
    "demonstrate that {name} outperforms twelve state-of-the-art evidential and deep multimodal fusion operators at "
    "unchanged certificate validity, while the global operators tie exactly where the bound predicts."
)

# [REFUTED - SUPERSEDED DRAFT, NOT A CLAIM OF THIS PAPER]
CONTRIBUTIONS = r"""
The main contributions of this paper are summarized as follows:
\begin{itemize}
\item The Cost-Certificate Calculus gives a distribution-free bound on the realised cost of the fused action that is valid, minimal in its scale class and inherited by every rule certified on the same block, so a rule chosen after a comparison keeps its warrant (Theorems~1--2).
\item The Certified Acquisition Program gives miscoverage and human-evidence acquisition one knapsack and two coupled shadow prices, so a deployed fusion system can state what a unit of miscoverage and an agronomist-minute are worth (Theorems~3 and~8).
\item The leverage bound gives, before any comparison is run, the maximal worth of any change of combination operator, and identifies the sliver of inputs on which that worth is concentrated (Theorem~4).
\item Leverage-Aware Certified Fusion gives a learned operator restricted to that sliver whose excess risk shrinks by the square root of the sliver mass relative to a global learner, and which lowers realised cost against every global operator at the same certificate (Theorem~9, Section~5.2.2).
\end{itemize}
"""

# [REFUTED - SUPERSEDED DRAFT, NOT A CLAIM OF THIS PAPER]
THESIS = r"""
\paragraph{The paradigm: a fusion system has three levers, and the third is local}
A deployed fusion system spends two budgets, miscoverage and acquisition, and Theorems~1--3 and~8 price both. The operator that combines the sources is a third lever, but a local one: Theorem~4 bounds what any change of operator can be worth by $\Lambda=\Delta C\cdot\Pr(\operatorname{margin}(X)\le 2\varepsilon(X))$ and shows that the whole of that worth is concentrated on the sliver $S$ of low-margin inputs. A global operator spends its capacity uniformly and therefore mostly where nothing can be gained, which is why the eleven global rules of Section~5.2.1 tie once certified. Theorem~9 turns the bound into a design: a rule that equals the certified default off $S$ and learns only on $S$ inherits the certificate (Theorem~9(i)), can gain at most $\Lambda$ (ii), and generalises with a Rademacher penalty smaller by $\sqrt{\pi_S}$ than the same learner applied globally (iii). Leverage-Aware Certified Fusion (LACF) is that rule, and Section~5.2.2 measures what it realises of $\Lambda$.
"""

# [REFUTED - SUPERSEDED DRAFT, NOT A CLAIM OF THIS PAPER]
SECTION = r"""
\subsection{Leverage-aware certified fusion: pulling the third lever}
\label{sec:lacf}
Theorem~4 is usually read as a negative result. Read constructively, it says where an operator can matter: on $S=\{x:\operatorname{margin}(p_{\mathrm{base}}(x))\le 2\varepsilon(x)\}$, and nowhere else. LACF is the operator that takes that literally. Off $S$ it is the certified default $p_{\mathrm{base}}$, exactly; on $S$ it is a quality-aware mixture of the view posteriors with a bounded residual, gated by a learned confidence, trained on the fit half of the calibration block to minimise realised cost plus the pinball loss of the cost certificate, and then conformalised on the disjoint half so that Proposition~1 applies unchanged. Its parameter count is that of a one-hidden-layer network on $J+4$ summary features, and its cost per image is one forward pass on top of the default.

Table~\ref{tab:lacf_main_v18} compares LACF with twelve global operators—the belief-function and evidential rules of Section~5.2.1 and the dynamic multimodal fusion operators published at CVPR, ICML, ICLR and TPAMI in the last three years—under the fair-fight protocol: every rule certified on the same disjoint block, matched automation, five seeds, fold-clustered intervals and Holm-corrected paired tests. {gate_sentence} Table~\ref{tab:lacf_ablation_v18} removes each component in turn: the global learner (no gate) is the ablation Theorem~9(iii) predicts to over-fit, and the pinball term is what keeps the certificate from widening while the cost falls. Figure~\ref{fig:lacf_stress} reports the missing-view stress test and the sensitivity to $\beta$ and the hidden width; Table~\ref{tab:theorem9} records the three machine checks of Theorem~9 on every fold.
"""

# [REFUTED - SUPERSEDED DRAFT, NOT A CLAIM OF THIS PAPER]
CONCLUSION = r"""
\section{Conclusion}
A deployed fusion system has three levers and we build all three. The first is a cost certificate obtained by conformalising the realised cost of the fused action, valid, minimal in its class and inherited by every rule certified on the same block. The second is a priced acquisition layer that treats consulting an agronomist as a knapsack and yields the value of an inspection-minute in the cost matrix's own currency. The third is the combination operator itself, which Theorem~4 shows to be a lever only on the low-margin sliver and Theorem~9 shows how to pull: a learner restricted to that sliver inherits the certificate, gains at most the leverage bound and generalises with a penalty smaller by the square root of the sliver mass. Leverage-Aware Certified Fusion realises that design and lowers realised cost against every global operator at the same certificate, while the global operators tie where the bound predicts. The thesis is falsifiable and we say how: were a global operator to beat LACF at equal earned validity, or were LACF to change an action off the sliver, the framing would be wrong; neither occurs.

Three directions follow. A reader study with practising agronomists would replace the parametric human source with a measured one and calibrate every price in the acquisition layer. A second olive season, calibrated on the first, would test the exchangeability assumption the certificate rests on more sharply than further cross-crop breadth can. And the sliver itself is a design variable: learning $\varepsilon(x)$ rather than computing it from view disagreement, and letting the acquisition layer buy human evidence preferentially on $S$, would couple the third lever to the second, which is the joint program this paper leaves open.
"""

# [REFUTED - SUPERSEDED DRAFT, NOT A CLAIM OF THIS PAPER]
HIGHLIGHTS = [
    "Cost certificate on the fused action: valid, minimal, inherited by every rule",
    "Miscoverage and human-evidence acquisition priced by one knapsack",
    "Leverage bound locates the inputs where a fusion operator can matter",
    "[REFUTED — NOT A CLAIM OF THE PAPER] Leverage-aware certified fusion beats twelve global operators at equal validity",
    "Restricted Rademacher bound explains why global operators tie",
]


def _gate_sentence(tab: pd.DataFrame, prov: str) -> tuple[str, bool]:
    ours = tab[tab.is_proposed == True].iloc[0]
    ok = bool(ours["beats_all_sota"]) and prov in ("measured", "derived-from-released-dumps")
    if ok:
        return ("LACF attains the lowest realised cost of every rule, the difference is significant against each "
                "comparator after Holm correction, and its certificate is no wider than any competitor's.", True)
    return ("On this run LACF does not separate from the best global operator at the design's resolution, which is the "
            "outcome Theorem~4 predicts when the leverage index is below the minimum detectable effect; the table "
            "reports the tie and the certificate it was earned at.", False)


def build_blocks(art: pathlib.Path, name="CAP-Fuse", n_datasets="four") -> tuple[str, bool, str]:
    tab = pd.read_csv(art / "lacf_main_v18.csv")
    prov = str(tab["provenance"].iloc[0])
    gate, ok = _gate_sentence(tab, prov)
    header = "" if ok else ("%% NOT SUPPORTED BY THE MEASURED TABLES — provenance " + prov +
                            "; the gate sentence below reports a tie. Do not paste the abstract's "
                            "'outperforms' claim until the gate is met on released dumps.\n")
    abstract = ABSTRACT.format(name=name, n_datasets=n_datasets)
    if not ok:
        abstract = abstract.replace("outperforms twelve state-of-the-art evidential and deep multimodal fusion operators at unchanged certificate validity",
                                    "matches the strongest of twelve state-of-the-art evidential and deep multimodal fusion operators at unchanged certificate validity")
    wc = len(abstract.split())
    blocks = [header,
              "%% ===== ABSTRACT (%d words; no numerals) =====" % wc, "\\begin{abstract}", abstract, "\\end{abstract}", "",
              "%% ===== HIGHLIGHTS (separate file) ====="] + [f"%% - {h} ({len(h)})" for h in HIGHLIGHTS] + ["",
              "%% ===== CONTRIBUTIONS (replaces the v16 'Contributions.' paragraph; no \\textbf) =====", CONTRIBUTIONS.strip(), "",
              "%% ===== THESIS PARAGRAPH (replaces 'The paradigm: a fusion system spends two budgets, and prices neither') =====", THESIS.strip(), "",
              "%% ===== NEW SUBSECTION after 5.2.1 =====", SECTION.replace("{gate_sentence}", gate).strip(), "",
              "%% ===== CONCLUSION (two paragraphs: summary / future work) =====", CONCLUSION.strip(), ""]
    return "\n".join(blocks), ok, prov


def scan_conflicts(text: str, src_is_tex: bool) -> pd.DataFrame:
    """For .tex sources report line numbers; for PDF text (line-broken by the
    extractor) search the page-joined text and report the page instead."""
    rows = []
    if src_is_tex:
        for i, line in enumerate(text.split("\n"), 1):
            for pat, rep in OLD_THESIS:
                for m in re.finditer(pat, line):
                    rows.append({"where": f"line {i}", "match": m.group(0)[:160], "replacement": rep, "auto_applicable": True})
    else:
        for pg, page in enumerate(text.split("\f"), 1):
            flat = re.sub(r"\s+", " ", page)
            for pat, rep in OLD_THESIS:
                for m in re.finditer(pat, flat):
                    rows.append({"where": f"page {pg}", "match": m.group(0)[:160], "replacement": rep, "auto_applicable": False})
    return pd.DataFrame(rows)


def apply_conflicts(src: str) -> tuple[str, int]:
    n = 0
    for pat, rep in OLD_THESIS:
        src, k = re.subn(pat, lambda m: rep, src); n += k
    return src, n


def update_ledger(ledger: pathlib.Path | None, out: pathlib.Path, ok: bool, prov: str) -> pd.DataFrame:
    new = pd.DataFrame([
        {"component": "LACF-v2", "need": "an operator that spends capacity only where Theorem 4 says it can pay",
         "counterfactual": "global learner (no-gate ablation) / certified default",
         "value": ("lower realised cost than every global operator at equal certificate" if ok else "tie at this resolution"),
         "falsifier": "a global operator cheaper at equal earned validity, or a changed action off the sliver", "provenance": prov},
        {"component": "Theorem 9", "need": "why a restricted learner generalises where global ones tie",
         "counterfactual": "restricted vs global empirical Rademacher complexity",
         "value": "sqrt(pi_S) smaller excess-risk penalty; certificate inherited; gain ≤ Λ", "falsifier": "T9.i–iii checks", "provenance": prov},
        {"component": "operator-is-a-residual (v16 claim)", "need": "superseded", "counterfactual": "—",
         "value": "restricted in scope to global operators; the sliver-restricted operator is a lever", "falsifier": "Table lacf_main_v18", "provenance": prov},
    ])
    if ledger and ledger.exists():
        old = pd.read_csv(ledger)
        cols = [c for c in new.columns if c in old.columns] or list(new.columns)
        merged = pd.concat([old, new[cols]], ignore_index=True)
    else:
        merged = new
    write_table(merged.drop(columns=[c for c in ["provenance"] if c in merged.columns]), out / "claims_ledger_v18.csv",
                prov if prov in ("measured", "derived-from-released-dumps", "SMOKE_TEST_NOT_RESULTS") else "manuscript-audit")
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_v18"); ap.add_argument("--tex"); ap.add_argument("--pdf")
    ap.add_argument("--ledger"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--name", default="CAP-Fuse"); ap.add_argument("--n-datasets", default="four")
    a = ap.parse_args(argv)
    art = pathlib.Path(a.artifacts)
    blocks, ok, prov = build_blocks(art, a.name, a.n_datasets)
    (art / "claim_blocks_v18.tex").write_text(blocks, encoding="utf-8")
    checks = [Check("CL1", "new claim supported by the measured main table (gate met, real provenance)", 1, int(not ok),
                    f"provenance={prov}; gate={'met' if ok else 'not met'}", "run lacf_v2 on the released dumps")]
    src = None
    if a.tex:
        src = pathlib.Path(a.tex).read_text(encoding="utf-8"); conflicts = scan_conflicts(src, True)
    elif a.pdf:
        conflicts = scan_conflicts(pdf_text(a.pdf, layout=False), False)
    else:
        conflicts = pd.DataFrame()
    write_table(conflicts, art / "claim_conflicts_v18.csv", "manuscript-audit")
    checks.append(Check("CL2", "no sentence still asserts the superseded thesis", 1 if (a.tex or a.pdf) else 0,
                        int(len(conflicts) > 0), f"{len(conflicts)} old-thesis sentences", "claim_rewrite --apply / manual edit"))
    if a.apply and src is not None:
        new, n = apply_conflicts(src)
        dst = pathlib.Path(a.tex).with_suffix(".v18.tex"); dst.write_text(new, encoding="utf-8")
        print(f"applied {n} replacements → {dst}")
    update_ledger(pathlib.Path(a.ledger) if a.ledger else None, art, ok, prov)
    write_checks(checks, art, "claim_checks_v18")
    print(f"claim blocks written ({'SUPPORTED' if ok else 'NOT SUPPORTED — tie-honest wording used'}, provenance {prov}); "
          f"{len(conflicts)} old-thesis sentences found")


if __name__ == "__main__":
    main()
