#!/usr/bin/env python3
"""산출물에 '어떤 뷰로 돌렸는가' 를 기록하게 만드는 패치.

왜 필요한가
-----------
본 실행이 사용한 뷰 경로가 어느 산출물에도 남지 않았다. 나중에 그것을 알아내려고
셸 히스토리를 뒤져야 했다. 이 감사가 막으려는 바로 그 종류의 결함이고, 심사자가
"이 표가 어느 덤프에서 나왔는가" 를 물으면 답할 근거가 파일에 없다는 뜻이다.

무엇을 남기는가
---------------
실행할 때마다 출력 폴더에 ``run_manifest_v18.json`` 을 쓴다.

  views          뷰 디렉터리의 절대 경로
  view_digest    뷰마다 preds/*.npz 의 (파일명, 크기, mtime) 을 SHA256 으로 접은 값
                 -- 내용이 바뀌면 값이 바뀐다. 덤프가 교체됐는지 사후에 확인 가능.
  index_digest   각 뷰의 첫 공통 cal 키에 들어 있는 image index 배열의 해시
                 -- 뷰들이 정말 같은 이미지 집합인지. 예전에 자동 선택이 올리브 1개와
                    카사바 2개를 섞은 적이 있어서 이 열이 필요하다.
  n_dumps        뷰별 덤프 수
  base           기반 연산자 (patch_base.py 적용 시)
  alpha, maxiter, argv, python, numpy/scipy/pandas 버전, host, utc
  platform_note  플랫폼 간 재현 오차가 있다는 사실을 파일 자체에 적어 둔다

플랫폼 간 ±0.002 차이를 우리가 실제로 관측했으므로(Linux/py3.11 vs Windows/py3.10),
어느 판에서 나온 표인지 파일이 스스로 말하게 하는 편이 안전하다.

멱등: 이미 패치된 파일은 건너뛴다. 원본은 .bak_prov 로 남긴다.

    python3 patch_provenance.py            패치
    python3 patch_provenance.py --check    상태 확인
    python3 patch_provenance.py --revert   되돌리기
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

TARGET = "olive_if_v18/run_all_v18.py"
MARK = "# --- run manifest patch ---"

HELPER = '''
''' + MARK + '''
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
'''

ANCHOR_DEF = "def main(argv=None):"
ANCHOR_CALL = "    lacf_v2.main(common)"
NEW_CALL = ('    _write_run_manifest(a.out, a.views,\n'
            '                        {"base": getattr(a, "base", None), "maxiter": a.maxiter,\n'
            '                         "smoke": bool(a.smoke), "max_folds": a.max_folds})\n'
            '    lacf_v2.main(common)')

EDITS = [(ANCHOR_DEF, HELPER + "\n\n" + ANCHOR_DEF), (ANCHOR_CALL, NEW_CALL)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    p, bak = root / TARGET, root / (TARGET + ".bak_prov")

    if not p.exists():
        print(f"★ 없음: {p}"); return 2
    if a.check:
        print(f"  {'패치됨' if MARK in p.read_text(encoding='utf-8') else '원본'}  {TARGET}")
        return 0
    if a.revert:
        if bak.exists():
            shutil.copy2(bak, p); bak.unlink(); print(f"  되돌림 {TARGET}")
        else:
            print("되돌릴 백업이 없습니다")
        return 0

    src = p.read_text(encoding="utf-8")
    if MARK in src:
        print(f"  이미 패치됨, 건너뜀  {TARGET}"); return 0
    if not bak.exists():
        shutil.copy2(p, bak)
    for old, new in EDITS:
        if src.count(old) != 1:
            print(f"★ 앵커가 {src.count(old)}번 발견됨 (1이어야 함): {old[:50]}")
            shutil.copy2(bak, p); bak.unlink(); return 3
        src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8")
    print(f"  패치 완료  {TARGET}   (원본 → {TARGET}.bak_prov)")
    print("\n확인:  python -m olive_if_v18.run_all_v18 --smoke --max-folds 2 --out /tmp/mf")
    print("       그 뒤 /tmp/mf/run_manifest_v18.json 을 보세요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
