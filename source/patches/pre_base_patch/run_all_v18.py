"""One command for the v18 claim change.

    python -m olive_if_v18.run_all_v18 --views runs/b0 runs/b1 runs/b2 --pdf manuscript.pdf \
        [--tex main.tex --apply] [--ledger results/artifacts_v4/tables/claims_ledger.csv] --out artifacts_v18
    python -m olive_if_v18.run_all_v18 --smoke --max-folds 2 --pdf OLIVE_manuscript.pdf --out artifacts_v18

Order: model + tables (lacf_v2) → Theorem 9 checks + appendix → claim blocks,
conflict scan, ledger (claim_rewrite) → combined verdict.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from . import claim_rewrite, lacf_v2, theorem9_lacf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int); ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--pdf"); ap.add_argument("--tex"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ledger"); ap.add_argument("--out", default="artifacts_v18")
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("give --views or --smoke")
    data = (["--views"] + a.views) if a.views else ["--smoke"]
    common = data + ["--out", a.out] + (["--max-folds", str(a.max_folds)] if a.max_folds else []) + ["--maxiter", str(a.maxiter)]
    lacf_v2.main(common)
    theorem9_lacf.main(common)
    cr = ["--artifacts", a.out]
    if a.tex: cr += ["--tex", a.tex] + (["--apply"] if a.apply else [])
    elif a.pdf: cr += ["--pdf", a.pdf]
    if a.ledger: cr += ["--ledger", a.ledger]
    claim_rewrite.main(cr)
    out = pathlib.Path(a.out)
    verdict = {k: json.loads((out / f"{k}.json").read_text()) for k in ("theorem9_checks_v18", "claim_checks_v18") if (out / f"{k}.json").exists()}
    (out / "verdict_v18.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
