"""Shared helpers for olive_if_v17.

Honesty contract (inherited from v16 and enforced here):
  1. A check with no evidence is a FAIL, never a PASS.
  2. Every generated table carries a provenance column.
  3. Nothing is fabricated: when a required input is missing the module
     writes a recovery line and returns a FAIL verdict.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from typing import Iterable

import pandas as pd

PROVENANCE = ("measured", "derived-from-released-dumps",
              "measured-but-not-comparable", "SMOKE_TEST_NOT_RESULTS",
              "manuscript-audit")


@dataclasses.dataclass
class Check:
    item: str              # checklist id, e.g. "C07"
    name: str              # human name
    n_evidence: int        # how many objects were inspected
    n_bad: int             # how many violated
    detail: str = ""
    fix: str = ""          # what to run / edit to fix it
    verifiable: bool = True

    @property
    def status(self) -> str:
        if not self.verifiable:
            return "UNVERIFIABLE"
        if self.n_evidence <= 0:
            return "FAIL"          # rule 1: empty evidence cannot pass
        return "PASS" if self.n_bad == 0 else "FAIL"

    def row(self) -> dict:
        return {"item": self.item, "check": self.name, "status": self.status,
                "n_evidence": self.n_evidence, "n_bad": self.n_bad,
                "detail": self.detail[:400], "fix": self.fix}


def write_checks(checks: Iterable[Check], out: pathlib.Path, title: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([c.row() for c in checks])
    df.insert(0, "provenance", "manuscript-audit")
    df.to_csv(out / f"{title}.csv", index=False)
    lines = [f"# {title}", ""]
    for _, r in df.iterrows():
        lines.append(f"- **[{r['status']}]** {r['item']} {r['check']} "
                     f"(n={r['n_evidence']}, bad={r['n_bad']})")
        if r["detail"]:
            lines.append(f"  - {r['detail']}")
        if r["status"] != "PASS" and r["fix"]:
            lines.append(f"  - fix: {r['fix']}")
    n_fail = int((df["status"] == "FAIL").sum())
    n_unv = int((df["status"] == "UNVERIFIABLE").sum())
    verdict = ("READY" if n_fail == 0 else f"NOT READY — {n_fail} FAIL, {n_unv} UNVERIFIABLE")
    lines += ["", f"**Verdict: {verdict}**"]
    (out / f"{title}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{title}.json").write_text(json.dumps(
        {"n_checks": len(df), "n_fail": n_fail, "n_unverifiable": n_unv,
         "verdict": verdict}, indent=2))
    return df


def write_table(df: pd.DataFrame, path: pathlib.Path, provenance: str) -> None:
    assert provenance in PROVENANCE, provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        df = pd.DataFrame([{"note": "EMPTY — no rows produced; see recovery plan"}])
    df = df.copy()
    df.insert(0, "provenance", provenance)
    df.to_csv(path, index=False)


NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])")


def fmt3(x) -> str:
    """Uniform three-decimal formatting for table cells (checklist item 11)."""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def pdf_text(path: str | pathlib.Path, layout: bool = True) -> str:
    import subprocess
    args = ["pdftotext"] + (["-layout"] if layout else []) + [str(path), "-"]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def pdf_pages(path: str | pathlib.Path) -> int:
    import subprocess
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0
