#!/usr/bin/env python3
"""Theorem 4 정의(비용 단위)로 슬리버를 다시 재는 진단.

왜
---
v18 하네스의 features() 는

    margin = 확률 상위 2개 차이,  eps = 뷰 사후분포의 평균 TV

를 썼다. 둘 다 확률 단위이고 비용 행렬이 전혀 들어가지 않는다. Theorem 4 는

    eps(x) = max_a |J_1(a|x) - J_2(a|x)|,   J(a|x) = sum_y p(y|x) C[y,a]

로 정의되어 있고(olive_if_v10/proofs.py:288, theory.py:556 이 그렇게 구현),
margin 도 같은 J 위의 최적/차선 간격이어야 한다. 즉 §7.1.2 가 만든 슬리버는
Theorem 4 의 슬리버가 아니다.

이 스크립트는 같은 덤프 위에서 두 정의를 나란히 재고, 각각에 대해
  pi_S,  P(R|S),  Lambda_eff = Delta_C * P(S and not R)
를 폴드별로 보고한다. 모델을 다시 적합하지 않는다 -- 정의만 바꿔 잰다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, CertifiedScorer,
                                             default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import _fold_data, _pairs


def prob_units(P, pb):
    """v18 하네스가 실제로 쓴 정의 (확률 단위)."""
    srt = np.sort(pb, -1)
    margin = srt[:, -1] - srt[:, -2]
    eps = 0.5 * np.abs(P - pb[None]).sum(-1).mean(0)
    return margin, eps


def cost_units(P, pb, C, include_referral=True):
    """Theorem 4 의 정의 (비용 단위). J(a|x) = sum_y p(y|x) C[y,a]."""
    A = C.shape[1] if include_referral else C.shape[1] - 1
    Cs = C[:, :A]
    Jb = pb @ Cs                       # (n, A)
    srt = np.sort(Jb, -1)
    margin = srt[:, 1] - srt[:, 0]     # 최적과 차선 사이의 비용 간격
    Jv = np.einsum("jnk,ka->jna", P, Cs)
    eps = np.abs(Jv - Jb[None]).max(-1).mean(0)
    return margin, eps


def referral_set(pb, C, automation):
    """배포된 이관 규칙: VOI 상위 (1-automation)."""
    K = C.shape[1] - 1
    J = pb @ C
    voi = J[:, :K].min(1) - J[:, K]
    n = len(voi)
    m = int(round((1.0 - automation) * n))
    R = np.zeros(n, bool)
    if m > 0:
        R[np.argsort(-voi)[:m]] = True
    return R, voi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--automation", type=float, default=0.79)
    ap.add_argument("--out", default="out_costunit")
    a = ap.parse_args()

    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    dC = float(C.max() - C.min())
    pairs = _pairs(keys, None)
    base = OPERATORS[a.base]

    rows = []
    for fi, (s, f) in enumerate(pairs):
        _, _, P_tst, _ = _fold_data(per_view, s, f)
        pb = base(P_tst)
        R, _ = referral_set(pb, C, a.automation)
        rec = {"fold": fi, "n": len(pb), "referral_rate": float(R.mean())}
        for tag, (m, e) in {
            "prob": prob_units(P_tst, pb),
            "cost": cost_units(P_tst, pb, C, True),
            "cost_noref": cost_units(P_tst, pb, C, False),
        }.items():
            S = m <= a.k * e
            piS = float(S.mean())
            pRS = float(R[S].mean()) if S.any() else float("nan")
            rec |= {f"piS_{tag}": piS, f"pRS_{tag}": pRS,
                    f"lam_eff_{tag}": dC * float((S & ~R).mean())}
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "costunit_sliver_perfold.csv", index=False)
    agg = df.drop(columns=["fold"]).mean().to_dict()
    agg["delta_C"] = dC
    agg["base"] = a.base
    agg["k"] = a.k
    (out / "costunit_sliver.json").write_text(
        json.dumps(agg, indent=1), encoding="utf-8")

    print(f"기반 {a.base} · k={a.k} · 이관 {1 - a.automation:.3f} · Delta_C={dC:.3f}")
    print(f"{'정의':<26}{'pi_S':>9}{'P(R|S)':>10}{'Lambda_eff':>13}")
    for tag, name in [("prob", "v18 구현 (확률 단위)"),
                      ("cost", "Theorem 4 (비용 단위)"),
                      ("cost_noref", "Theorem 4 (이관 제외)")]:
        print(f"{name:<22}{agg[f'piS_{tag}']:>9.4f}{agg[f'pRS_{tag}']:>10.4f}"
              f"{agg[f'lam_eff_{tag}']:>13.4f}")
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
