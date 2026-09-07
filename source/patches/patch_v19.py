#!/usr/bin/env python3
"""Fix the five defects the adversarial screening found in olive_if_v19.

Idempotent.  Keeps a .bak_v19 copy.  Run once, then re-run the harness:

    python3 patch_v19.py
    LACF_UNITS=cost python -m olive_if_v19.certified_multisource --views <8 dump dirs> --out artifacts_v19

WHAT IS WRONG, AND WHY IT MATTERS
---------------------------------
1. REFERENCE ARM.  summarise() takes the reference to be `names[0]`, i.e. whatever
   directory came first on the command line, and calls it "the deployed single-source
   system".  With alphabetical views that is densenet121 (0.216), the third-cheapest
   source.  The manuscript's table compares against the CHEAPEST single source
   (mambavision, 0.208).  So the deposited summary disagrees with the paper it is cited
   to support: +0.035 against +0.027, and a Holm p of 0.132 against the paper's 0.001.
   The per-fold file is correct; only the summary is computed off the wrong baseline.
   Fix: reference = cheapest single source, and emit the arbitrary-source comparison as
   its own column, because it is the one comparison that needs no selection at all.

2. NO RUN MANIFEST.  Every v18 driver writes one; v19 writes none.  Nothing in
   results/v19/ records which eight dumps were used, IN WHAT ORDER (the order decides the
   reference arm and the forward-selection sequence), or the library versions.

3. TWO OPERATORS SILENTLY SKIPPED.  max_confidence and qmf are dropped without comment,
   so the deposit holds ten operator arms while the manuscript says eleven, and one arm
   the manuscript names by hand (max_confidence) is in none of the released v19 files.
   Fix: run max_confidence; keep qmf out but say why in the code -- on posterior-only
   evidence it is bit-identical to the sum rule (Supp. S24).

4. SELECTION OBJECTIVE MISMATCH.  Forward selection minimises a plug-in cost with NO
   referral, while every reported cost is scored with the referral budget on.  The
   manuscript blames J*'s failure on the 167-image fit block alone; the objective
   mismatch is a second cause.  Fix: select on the same objective the table reports.

5. THE MERGE PENALTY IS A CLIPPING ARTEFACT.  merge_mean and merge_bonf multiply by 2
   and by J and then the result is renormalised -- and a constant multiplier is a no-op
   under normalisation.  What survives is only np.minimum(1.0, .) saturating entries.
   So the measured merge cost is produced by reading a valid p-vector as a posterior,
   not by the dependence factor as the manuscript's mechanism paragraph implies.
   Fix: add the unclipped mean and min arms as controls, so the multiplier's isolated
   effect is visible in the deposit and the prose can name the real mechanism.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

TARGET = "olive_if_v19/certified_multisource.py"
MARK = "# --- screening patch v19 ---"

EDITS = [
    # ---- 3. operator skip list, documented -------------------------------------
    ('''            if oname in ("product_rule", "sum_rule", "qmf", "max_confidence"):
                continue''',
     '''            # product_rule and sum_rule are already run as the CMS arms above.
            # qmf is skipped because on posterior-only evidence its free-energy weight is
            # identically zero, making it bit-identical to sum_rule (Supp. S24); running it
            # would print the same numbers twice under a second citation.
            if oname in ("product_rule", "sum_rule", "qmf"):
                continue'''),

    # ---- 5. unclipped merge controls -------------------------------------------
    ('''MERGES = {"merge: Vovk-Wang mean": merge_mean, "merge: Bonferroni": merge_bonf, "merge: Fisher (invalid ref.)": merge_fisher}''',
     '''def merge_mean_raw(Pv):  return Pv.mean(0)                       # same rule, validity factor removed
def merge_min_raw(Pv):   return Pv.min(0)                        # same rule, validity factor removed
MERGES = {"merge: Vovk-Wang mean": merge_mean, "merge: Bonferroni": merge_bonf,
          "merge: Fisher (invalid ref.)": merge_fisher,
          # Controls. The merged p-vector is renormalised into a pseudo-posterior, under which a
          # constant multiplier is a no-op; the validity factors therefore reach the decision only
          # where they saturate at 1. These two arms are the same rules with the factor removed, so
          # the difference between each pair isolates what the saturation costs.
          "merge: mean, factor removed (control)": merge_mean_raw,
          "merge: min, factor removed (control)": merge_min_raw}'''),

    # ---- 4. selection objective matches the reported one ------------------------
    ('''        fit_cost = lambda p: C[y_cal[fi], (p @ C[:, :K]).argmin(1)].mean()''',
     '''        # Select on the SAME objective the table reports: the certified decision path with the
        # referral budget on. The earlier version minimised a plug-in cost with no referral, so
        # selection optimised a quantity that omits the layer which dominates the reported cost.
        _sel_scorer = CertifiedScorer(C, alpha)
        def fit_cost(p):
            sc = _sel_scorer.fit(p, y_cal[fi])
            return float(sc.evaluate(p, y_cal[fi])["cost"])'''),
]

# ---- 1. reference arm + arbitrary-source column --------------------------------
OLD_REF = '''def summarise(per, out, prov, names, J):
    ref_name = f"single: {names[0]}"                                   # the deployed single-source system'''
NEW_REF = MARK + '''
def summarise(per, out, prov, names, J):
    # The reference is the CHEAPEST single source, which is what the manuscript compares
    # against. It is an in-sample argmin over the same folds, so it is an oracle: a
    # practitioner cannot know which source is cheapest without labels. Two further
    # baselines are emitted beside it because they need no selection at all --
    # `gain_vs_mean_single` (the source you get by picking one arbitrarily) and
    # `gain_vs_loo_single` (choosing the cheapest source on the other nine folds).
    _sing = per[per.group == "single"].groupby("method")["cost"].mean()
    ref_name = _sing.idxmin()'''

OLD_TAIL = '''    tab["resolvable_vs_single"] = tab["gain_vs_single"].abs() >= tab["mde"]'''
NEW_TAIL = '''    tab["resolvable_vs_single"] = tab["gain_vs_single"].abs() >= tab["mde"]
    tab.loc[ref_name, ["gain_vs_single", "mde"]] = np.nan              # a row is not resolvable against itself
    tab.loc[ref_name, "resolvable_vs_single"] = False
    # --- selection-free baselines -------------------------------------------------
    wide = per[per.group == "single"].pivot_table(index=["seed", "fold"], columns="method", values="cost")
    mean_single = wide.mean(axis=1).reindex(ref.index)
    loo = []
    for k in range(len(wide)):
        rest = wide.drop(wide.index[k])
        loo.append(wide.iloc[k][rest.mean().idxmin()])
    loo_single = pd.Series(loo, index=wide.index).reindex(ref.index)
    g_mean, g_loo, m_mean, m_loo = {}, {}, {}, {}
    for m in tab.index:
        A = per[per.method == m].set_index(["seed", "fold"])["cost"].reindex(ref.index)
        d1 = mean_single - A; g_mean[m] = float(d1.mean()); m_mean[m] = _mde(d1)
        d2 = loo_single - A; g_loo[m] = float(d2.mean()); m_loo[m] = _mde(d2)
    tab["gain_vs_mean_single"] = pd.Series(g_mean); tab["mde_vs_mean_single"] = pd.Series(m_mean)
    tab["gain_vs_loo_single"] = pd.Series(g_loo); tab["mde_vs_loo_single"] = pd.Series(m_loo)
    tab["ref_arm"] = ref_name
    tab["mean_single_cost"] = float(mean_single.mean())
    tab["loo_single_cost"] = float(loo_single.mean())'''

# ---- 2. run manifest ------------------------------------------------------------
OLD_WRITE = '''    per = pd.DataFrame(rows)
    write_table(per, out / "cms_perfold_v19.csv", prov)
    return summarise(per, out, prov, names, J)'''
NEW_WRITE = '''    per = pd.DataFrame(rows)
    write_table(per, out / "cms_perfold_v19.csv", prov)
    _write_manifest_v19(out, view_dirs, names, alpha, prov)
    return summarise(per, out, prov, names, J)


def _write_manifest_v19(out_dir, view_dirs, names, alpha, prov):
    """Record what this run consumed. The VIEW ORDER is load-bearing: it fixes the
    forward-selection sequence, so it is written out explicitly."""
    import datetime, hashlib, json, os, platform, socket
    def digest(run):
        p = pathlib.Path(run); preds = p / "preds" if (p / "preds").is_dir() else p
        h = hashlib.sha256(); files = sorted(preds.glob("*.npz"))
        for f in files:
            st = f.stat(); h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}|".encode())
        return h.hexdigest()[:16], len(files)
    def ver(m):
        try: return __import__(m).__version__
        except Exception: return None
    rows = []
    for v in (view_dirs or []):
        d, n = digest(v)
        rows.append({"path": os.path.abspath(v), "view_digest": d, "n_dumps": n})
    man = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "host": socket.gethostname(), "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": ver("numpy"), "scipy": ver("scipy"),
           "pandas": ver("pandas"), "argv": sys.argv, "alpha": alpha, "provenance": prov,
           "view_order": list(names), "views": rows, "n_views": len(rows),
           "note": ("Costs are in the harness cost matrix's own units, not EUR/tree. "
                    "view_order fixes the forward-selection sequence and therefore J*.")}
    try:
        p = pathlib.Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "run_manifest_v19.json").write_text(json.dumps(man, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
        print(f"[manifest] {p / 'run_manifest_v19.json'}  views {len(rows)}")
    except Exception as e:                                            # noqa: BLE001
        print(f"[manifest] not written (ignored): {type(e).__name__}: {e}")'''

# ---- LaTeX header + withdrawn framing -------------------------------------------
OLD_TEX = '''"Arm & $J$ & Cost (EUR/tree) & Bound $W$ & Violation & Missed disease & Gain vs.\\\\ single [MDE] & $p_{\\\\mathrm{Holm}}$\\\\\\\\", "\\\\midrule"]'''
NEW_TEX = '''"Arm & $J$ & Cost (harness units) & Bound $W$ & Violation & Missed disease & Gain vs.\\\\ single [MDE] & $p_{\\\\mathrm{Holm}}$\\\\\\\\", "\\\\midrule"]'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    p, bak = root / TARGET, root / (TARGET + ".bak_v19")

    if not p.exists():
        print(f"* missing: {p}"); return 2
    if a.check:
        print(f"  {'patched' if MARK in p.read_text(encoding='utf-8') else 'original'}  {TARGET}")
        return 0
    if a.revert:
        if bak.exists():
            shutil.copy2(bak, p); bak.unlink(); print(f"  reverted {TARGET}")
        else:
            print("  no backup to revert")
        return 0

    src = p.read_text(encoding="utf-8")
    if MARK in src:
        print(f"  already patched, skipping  {TARGET}"); return 0
    edits = EDITS + [(OLD_REF, NEW_REF), (OLD_TAIL, NEW_TAIL), (OLD_WRITE, NEW_WRITE), (OLD_TEX, NEW_TEX)]
    for old, _ in edits:
        if src.count(old) != 1:
            print(f"* anchor found {src.count(old)} times (need 1): {old[:70]}")
            return 3
    if not bak.exists():
        shutil.copy2(p, bak)
    for old, new in edits:
        src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8")
    print(f"  patched  {TARGET}   (original -> {TARGET}.bak_v19)")
    print("\n  next:  LACF_UNITS=cost python -m olive_if_v19.certified_multisource \\")
    print("           --views <8 dump dirs> --out artifacts_v19")
    return 0


if __name__ == "__main__":
    sys.exit(main())
