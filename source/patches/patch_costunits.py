#!/usr/bin/env python3
"""features() 의 margin/eps 를 Theorem 4 의 비용 단위로 계산할 수 있게 하는 패치.

결함
-----
v18 하네스는 margin 을 '확률 상위 2개의 차이', eps 를 '뷰 사후분포의 평균 TV'
로 계산한다. 비용 행렬이 한 번도 들어가지 않는다. Theorem 4 는

    eps(x) = max_a |J_1(a|x) - J_2(a|x)|,   J(a|x) = sum_y p(y|x) C[y,a]

로 정의되고 v10 구현(proofs.py:288, theory.py:556)이 그렇게 되어 있다.
margin 도 같은 J 위의 최적/차선 간격이어야 한다. 따라서 §7.1.2 가 만든
슬리버는 Theorem 4 의 슬리버가 아니며, 같은 덤프에서 재면

    확률 단위   pi_S = 0.094,  P(R|S) = 1.000,  Lambda_eff = 0
    비용 단위   pi_S = 0.602,  P(R|S) = 0.347,  Lambda_eff = 8.36

로 포함관계가 깨진다. 즉 "이관이 이미 사 버렸기 때문에 동일하다" 는 설명은
구현된 정의에서만 성립한다.

패치 내용
---------
전역 UNITS ("prob" | "cost") 와 COST 를 추가한다. 기본값은 "prob" 이라
기존 산출물이 그대로 재현된다. run_all_v18 에 --units 를 추가한다.

    python3 patch_costunits.py            패치
    python3 patch_costunits.py --check    상태
    python3 patch_costunits.py --revert   되돌리기
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

MARK = "# --- cost-unit features patch ---"

OLD_FEAT = '''def features(P):
    J, n, K = P.shape
    pb = base_rule(P)          # 기반 연산자 (BASE_NAME)
    mx = P.max(-1).T
    js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                  for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
    srt = np.sort(pb, -1); margin = srt[:, -1] - srt[:, -2]
    eps = 0.5 * np.abs(P - pb[None]).sum(-1).mean(0)
    phi = np.c_[mx, js, margin, _H(P).mean(0), eps]
    return phi, pb, margin, eps'''

NEW_FEAT = MARK + '''
# margin/eps 를 어느 단위로 잴 것인가.
#   "prob"  v18 원 구현. 확률 상위 2개 차이와 뷰 사후분포 평균 TV. 비용 무관.
#   "cost"  Theorem 4 의 정의. 기대비용 J(a|x)=sum_y p(y|x) C[y,a] 위에서 잰다.
# 기본값을 "prob" 으로 두어 기존 산출물이 그대로 재현되게 한다.
UNITS = "prob"
COST = None


def set_units(units, cost=None):
    """단위를 지정한다. cost 단위를 쓰려면 비용 행렬이 반드시 있어야 한다."""
    global UNITS, COST
    if units not in ("prob", "cost"):
        raise SystemExit(f"알 수 없는 단위 '{units}'. prob 또는 cost")
    if units == "cost" and cost is None:
        raise SystemExit("units='cost' 에는 비용 행렬이 필요합니다")
    UNITS, COST = units, (None if cost is None else np.asarray(cost, float))
    return UNITS


def _margin_eps(P, pb):
    if UNITS == "prob":
        srt = np.sort(pb, -1)
        margin = srt[:, -1] - srt[:, -2]
        eps = 0.5 * np.abs(P - pb[None]).sum(-1).mean(0)
        return margin, eps
    C = COST                                   # (K, A), 마지막 열은 이관
    Jb = pb @ C
    srt = np.sort(Jb, -1)
    margin = srt[:, 1] - srt[:, 0]             # 최적/차선 기대비용 간격
    Jv = np.einsum("jnk,ka->jna", P, C)
    eps = np.abs(Jv - Jb[None]).max(-1).mean(0)
    return margin, eps


def features(P):
    J, n, K = P.shape
    pb = base_rule(P)          # 기반 연산자 (BASE_NAME)
    mx = P.max(-1).T
    js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                  for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
    margin, eps = _margin_eps(P, pb)
    phi = np.c_[mx, js, margin, _H(P).mean(0), eps]
    return phi, pb, margin, eps
# --- end cost-unit features patch ---'''

TARGETS = [("olive_if_v18/lacf_v2.py", [(OLD_FEAT, NEW_FEAT)])]

RUNNER = "olive_if_v18/run_all_v18.py"
RUN_EDITS = [
    ('ap.add_argument("--base"',
     'ap.add_argument("--units", default="prob", choices=("prob", "cost"),\n'
     '                    help="margin/eps 단위. cost 는 Theorem 4 정의")\n'
     '    ap.add_argument("--base"'),
]


def _apply(path, edits, bak_suffix):
    p = pathlib.Path(path)
    bak = pathlib.Path(str(path) + bak_suffix)
    if not p.exists():
        print(f"★ 없음: {p}")
        return False
    src = p.read_text(encoding="utf-8")
    if MARK in src:
        print(f"  이미 패치됨, 건너뜀  {path}")
        return True
    for old, _ in edits:
        if src.count(old) != 1:
            print(f"★ 앵커가 {src.count(old)}번 발견됨(1이어야 함)  {path}")
            return False
    if not bak.exists():
        shutil.copy2(p, bak)
    for old, new in edits:
        src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8")
    print(f"  패치 완료  {path}  (원본 → {bak.name})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()

    if a.check:
        for t, _ in TARGETS:
            p = root / t
            print(f"  {'패치됨' if p.exists() and MARK in p.read_text(encoding='utf-8') else '원본'}  {t}")
        return 0
    if a.revert:
        for t, _ in TARGETS:
            bak = root / (t + ".bak_units")
            if bak.exists():
                shutil.copy2(bak, root / t); bak.unlink(); print(f"  되돌림 {t}")
            else:
                print(f"  백업 없음 {t}")
        return 0

    ok = all(_apply(root / t, e, ".bak_units") for t, e in TARGETS)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
