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



# --- run manifest patch ---
def _write_run_manifest(out_dir, views, extra=None):
    """어떤 뷰로 돌렸는지를 출력 폴더에 남긴다. 실패해도 실행을 막지 않는다."""
    import datetime, hashlib, json, os, pathlib, platform, socket, sys
    try:
        import numpy as np
    except Exception:                                            # noqa: BLE001
        np = None

    def _dump_digest(run):
        p = pathlib.Path(run)
        preds = p / "preds" if (p / "preds").is_dir() else p
        h = hashlib.sha256()
        files = sorted(preds.glob("*.npz"))
        for f in files:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}|".encode())
        return h.hexdigest()[:16], len(files)

    def _index_digest(run):
        """첫 cal 덤프의 image index 해시 -- 뷰들이 같은 이미지 집합인지 확인용."""
        if np is None:
            return None
        p = pathlib.Path(run)
        preds = p / "preds" if (p / "preds").is_dir() else p
        for f in sorted(preds.glob("cal_*.npz")):
            try:
                z = np.load(f)
                key = "index" if "index" in z.files else ("y" if "y" in z.files else None)
                if key is None:
                    return None
                a = np.ascontiguousarray(np.asarray(z[key]))
                return f"{f.name}:{hashlib.sha256(a.tobytes()).hexdigest()[:16]}"
            except Exception:                                    # noqa: BLE001
                return None
        return None

    rows = []
    for v in (views or []):
        d, n = _dump_digest(v)
        rows.append({"path": os.path.abspath(v), "view_digest": d,
                     "n_dumps": n, "index_digest": _index_digest(v)})

    def _ver(m):
        try:
            return __import__(m).__version__
        except Exception:                                        # noqa: BLE001
            return None

    man = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _ver("numpy"), "scipy": _ver("scipy"), "pandas": _ver("pandas"),
        "argv": sys.argv,
        "views": rows,
        "n_views": len(rows),
        "distinct_index_digests": sorted({r["index_digest"] for r in rows
                                          if r["index_digest"]}),
        "platform_note": ("L-BFGS paths differ slightly across BLAS/python builds; we have "
                          "measured +/-0.002 on ablation costs between Linux/py3.11 and "
                          "Windows/py3.10, an order below this design's MDE. Compare tables "
                          "only within one manifest."),
    }
    if extra:
        man.update(extra)
    try:
        p = pathlib.Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "run_manifest_v18.json").write_text(
            json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
        n = len(man["distinct_index_digests"])
        print(f"[manifest] {p / 'run_manifest_v18.json'}  뷰 {len(rows)}개 · "
              f"index 지문 {n}종"
              + ("  ★ 뷰들이 서로 다른 이미지 집합입니다 -- 코퍼스가 섞였는지 확인하세요"
                 if n > 1 else ""))
    except Exception as e:                                       # noqa: BLE001
        print(f"[manifest] 기록 실패(무시): {type(e).__name__}: {e}")
# --- end run manifest patch ---


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int); ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--pdf"); ap.add_argument("--tex"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ledger"); ap.add_argument("--out", default="artifacts_v18")
    ap.add_argument("--base", default="sum_rule",
                    help="기반 연산자. 기본 sum_rule 은 v18 원 실행과 같다")
    ap.add_argument("--units", default=__import__("os").environ.get("LACF_UNITS", "prob"),
                    choices=("prob", "cost"),
                    help="margin/eps 단위. cost 는 Theorem 4 정의")
    a = ap.parse_args(argv)
    if not a.views and not a.smoke:
        ap.error("give --views or --smoke")
    data = (["--views"] + a.views) if a.views else ["--smoke"]
    common = data + ["--out", a.out] + (["--max-folds", str(a.max_folds)] if a.max_folds else []) + ["--maxiter", str(a.maxiter)] + ["--base", a.base] + ["--units", a.units]
    _write_run_manifest(a.out, a.views,
                        {"base": getattr(a, "base", None), "units": getattr(a, "units", "prob"),
                         "maxiter": a.maxiter,
                         "smoke": bool(a.smoke), "max_folds": a.max_folds})
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
