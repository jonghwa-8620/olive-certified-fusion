#!/usr/bin/env python3
"""S ⊆ 이관집합 은 발견인가 산술인가 — R4 의 가장 큰 위험.

문제
----
R4 는 세 데이터셋에서 P(이관|S) 가 1.000 / 1.000 / 0.980 임을 보였다. 그런데
세 경우 모두 π_S < 이관율이다:

    올리브 b100  π_S 0.094 < 0.208      올리브 b050  π_S 0.039 < 0.208
    카사바       π_S 0.107 < 0.210

그리고 두 선택 규칙이 같은 양을 본다. sliver 는 margin ≤ 2ε, 이관은 VOI 상위
(1−a). VOI 가 margin 의 단조감소 함수에 가깝다면 π_S < 1−a 인 순간 포함관계는
**산술적으로** 따라 나온다 -- 코퍼스의 성질이 아니라 두 문턱값의 관계다.

심사자가 이걸 보면 §2.1 이 "측정된 발견" 에서 "정의의 귀결" 로 격하된다.
그러니 우리가 먼저 판정한다. 숨기는 게 아니라 정면으로 간다.

완전 공단조라면 예측은 정확히
    P(이관 | S) = min(1, (1−a) / π_S)
이다. 관측이 이 예측선을 따라가면 산술이고, 벗어나면 자료의 성질이 남아 있다.

무엇을 재는가
-------------
  1. 공단조성   Spearman ρ( margin − 2ε , VOI ). 강한 음수면 두 규칙이 같은 순서다.
  2. 예측 대조  각 k 에서 관측 P(이관|S) vs min(1, (1−a)/π_S)
  3. k 스윕     sliver 폭 k 를 키워 π_S 를 이관율 위로 올린다. 포함관계가
                π_S = 1−a 지점에서 깨지면 산술 해석이 확인된다. 그리고 깨진
                뒤에도 Λ_eff 와 천장이 열리는지를 본다 -- 열리는데 이득이
                없다면 그건 또 다른 이야기다.

판정에 따라 원고가 갈린다.
  산술이면   §2.1 을 명제로 쓰고 한 줄 증명을 붙인다. 실험은 공단조성 확인용.
             "우리 sliver 정의와 이관 규칙이 같은 점을 고르므로, 구성상 어떤
             sliver-제한 규칙도 실현 비용을 바꿀 수 없다" -- 더 강한 진술이다.
  아니면     경험적 발견으로 쓰되 왜 그런지 설명해야 한다.

학습이 없어 수 초. python3 sliver_arithmetic.py --views <런들...>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import _fold_data, _pairs, sliver


def parts(P, base_fn, C):
    """기반 확률, margin, ε, VOI. features() 와 같은 정의를 쓴다."""
    pb = base_fn(P)
    from olive_if_v18.lacf_v2 import _margin_eps      # 단위 전역(LACF_UNITS)을 따른다
    margin, eps = _margin_eps(P, pb)
    K = C.shape[0]
    exp_cost = pb @ C[:, :K]
    voi = exp_cost.min(1) - C[0, K]          # CertifiedScorer.act 과 같은 식
    return pb, margin, eps, voi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--automation", type=float, default=0.79)
    ap.add_argument("--ks", type=float, nargs="+",
                    default=[0.5, 1, 2, 3, 4, 6, 8, 12, 20],
                    help="sliver 폭. 2.0 이 논문의 정의 (margin ≤ 2ε)")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    dC = float(C[1:, 0].max())
    pairs = _pairs(keys, None)
    base_fn = OPERATORS[a.base]
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · 기반 {a.base} · 자동화율 {a.automation}")
    print(f"이관율 = 1 − a = {1 - a.automation:.3f}\n")

    rows, rho_rows = [], []
    for s, f in pairs:
        P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
        pb, margin, eps, voi = parts(P_tst, base_fn, C)
        n = len(voi)
        n_ref = int(round((1 - a.automation) * n))
        referred = np.zeros(n, bool)
        referred[np.argsort(-voi)[:n_ref]] = True
        slack = None

        # ---- 1. 공단조성: sliver 통계량과 VOI 가 같은 순서인가
        stat = margin - 2.0 * eps                      # 음수면 논문 정의의 S
        r_all = stats.spearmanr(stat, voi).statistic
        r_m = stats.spearmanr(margin, voi).statistic
        rho_rows.append({"seed": s, "fold": f, "rho_stat_voi": float(r_all),
                         "rho_margin_voi": float(r_m)})

        # ---- 2·3. k 스윕
        for k in a.ks:
            onS = sliver(margin, eps, k).astype(bool)
            piS = float(onS.mean())
            if piS <= 0:
                continue
            auto = ~referred
            pred = min(1.0, (1 - a.automation) / piS)   # 완전 공단조일 때의 예측
            obs = float(referred[onS].mean())
            if slack is None:
                acts = np.where(referred, K, (pb @ C[:, :K]).argmin(1))
                slack = C[y_tst, acts] - C[y_tst, :].min(1)
            rows.append({
                "seed": s, "fold": f, "k": k, "pi_S": piS,
                "P_refer_given_S_obs": obs, "P_refer_given_S_pred": pred,
                "gap": obs - pred,
                "pi_S_eff": float((onS & auto).mean()),
                "Lambda": dC * piS, "Lambda_eff": dC * float((onS & auto).mean()),
                "ceil_S_auto": float(slack[onS & auto].sum() / n) if (onS & auto).any() else 0.0,
            })
    df = pd.DataFrame(rows); rh = pd.DataFrame(rho_rows)
    df.to_csv(out / "sliver_arithmetic_perfold_v18.csv", index=False)

    print("=" * 96)
    print("1. 공단조성 — sliver 통계량(margin − 2ε)과 VOI 가 같은 순서인가")
    print("=" * 96)
    rs, rm = rh.rho_stat_voi, rh.rho_margin_voi
    print(f"  Spearman ρ(margin − 2ε, VOI)   {rs.mean():+.4f}  (SD {rs.std(ddof=1):.4f}, "
          f"범위 {rs.min():+.3f}~{rs.max():+.3f})")
    print(f"  Spearman ρ(margin, VOI)        {rm.mean():+.4f}  (SD {rm.std(ddof=1):.4f})")
    strong = abs(rs.mean()) > 0.9
    print("  → " + ("두 규칙이 사실상 같은 순서입니다. 포함관계는 산술의 귀결입니다."
                    if strong else
                    "완전 공단조는 아닙니다. 포함관계에 자료의 성질이 섞여 있습니다."))

    g = df.groupby("k").agg(pi_S=("pi_S", "mean"),
                            obs=("P_refer_given_S_obs", "mean"),
                            pred=("P_refer_given_S_pred", "mean"),
                            gap=("gap", "mean"), gap_sd=("gap", "std"),
                            Lambda_eff=("Lambda_eff", "mean"),
                            ceil=("ceil_S_auto", "mean")).reset_index()
    print("\n" + "=" * 96)
    print(f"2·3. k 스윕 — 이관율 {1-a.automation:.3f} 를 π_S 가 넘어서면 포함이 깨져야 한다")
    print("=" * 96)
    print("  pred = min(1, (1−a)/π_S) — 완전 공단조일 때의 예측값")
    print()
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    mad = float(g.gap.abs().mean())
    cross = g[g.pi_S > (1 - a.automation)]
    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print(f"  관측−예측 평균절대차 {mad:.4f}")
    if mad < 0.05 and strong:
        print("  → 산술입니다. §2.1 은 경험적 발견이 아니라 명제로 써야 합니다:")
        print("      'sliver 통계량과 VOI 가 공단조이고 π_S ≤ 1−a 이면 S ⊆ 이관집합이고,")
        print("       따라서 모든 sliver-제한 규칙의 실현 비용은 기반과 같다.'")
        print("     이쪽이 관찰보다 강한 진술입니다. 실험은 공단조성 확인용으로 남깁니다.")
    else:
        print("  → 예측선에서 벗어납니다. 포함관계에 자료의 성질이 남아 있으므로")
        print("     경험적 발견으로 쓰되, 왜 그런지 설명이 필요합니다.")
    if len(cross):
        c = cross.iloc[0]
        print(f"\n  π_S 가 이관율을 넘는 첫 지점: k={c.k:g} (π_S {c.pi_S:.3f})")
        print(f"    거기서 P(이관|S) {c.obs:.3f}, Λ_eff {c.Lambda_eff:.3f}, 천장 {c.ceil:.4f}")
        print("    ★ 여기서 레버리지가 열립니다. 열렸는데도 LACF 가 이득을 못 내면")
        print("      그건 별개의 사실이고 따로 보고해야 합니다 (R1 과 같은 이야기).")
    else:
        print(f"\n  k 를 {max(a.ks):g} 까지 키워도 π_S 가 이관율을 못 넘었습니다.")
        print("    sliver 정의가 이 코퍼스에서 구조적으로 좁다는 뜻입니다.")

    (out / "sliver_arithmetic_v18.json").write_text(json.dumps(
        {"base": a.base, "automation": a.automation, "n_folds": len(pairs),
         "rho_stat_voi_mean": float(rs.mean()), "rho_margin_voi_mean": float(rm.mean()),
         "mean_abs_gap": mad, "comonotone": bool(strong),
         "arithmetic": bool(mad < 0.05 and strong),
         "rows": g.to_dict("records")}, indent=1, ensure_ascii=False))
    print("\n예치: sliver_arithmetic_v18.json · sliver_arithmetic_perfold_v18.csv")


if __name__ == "__main__":
    main()
