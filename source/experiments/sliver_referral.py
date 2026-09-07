#!/usr/bin/env python3
"""sliver 가 이미 전문가 이관 영역인가 — 그렇다면 Theorem 4 의 Λ 를 고쳐야 한다.

관찰
----
게이트 진단에서 γ=0 일 때 sliver 위 게이트가 0.755 로 열리고 결정이 24.3%
바뀌었는데도 Δ비용이 10 폴드 전부 정확히 0.0000 이었다. 결정이 바뀌었는데
비용이 한 번도 안 움직이는 건 통계가 아니라 구조다.

CertifiedScorer.act() 를 보면 이유가 될 만한 것이 있다:

    voi  = exp_cost.min(1) - C[0, K]
    acts[argsort(-voi)[:n_ref]] = K          # VOI 상위 21% 를 전문가로 이관

VOI 가 높다 = 기대비용이 높다 = 불확실하다. sliver 는 정의상 불확실한 영역
(margin ≤ 2ε)이다. 이관된 이미지의 실현 비용은 C[y, K] = expert_cost 고정이라
사후확률이 어떻게 바뀌든 비용이 변하지 않는다.

가설:  sliver ⊆ (거의) 이관 영역.  따라서 **어떤 융합 연산자도 그곳의 실현
       비용을 바꿀 수 없다.**  Theorem 4 의 Λ = ΔC·π_S 는 실현 가능한
       레버리지를 과대평가한다. 옳은 양은

           Λ_eff = ΔC · π_S · (1 − P(이관 | S))

무엇을 재는가 (학습 없음, 기반 확률만으로)
------------------------------------------
  pi_S            sliver 비율
  refer_rate      전체 이관율 (설계상 0.21)
  P(refer | S)    sliver 중 이관되는 비율      ← 가설의 핵심
  P(S | refer)    이관 중 sliver 인 비율
  pi_S_eff        sliver 이면서 자동 처리되는 비율 = 실제로 손댈 수 있는 영역
  Lambda, Lambda_eff
  ceil_all        조합 상한: 모든 이미지에서 정답 기준 최적 행동을 골랐을 때의 절감
  ceil_S_auto     그 상한 중 '자동 처리되는 sliver' 에서 온 몫 ← 국소 규칙이
                  실제로 가져갈 수 있는 최대치. 최적화도 대리목적도 없는 진짜 천장.

ceil_S_auto 는 이관 배정을 고정한 채 계산하므로 보수적이다(규칙이 p 를 바꾸면
VOI 순위가 바뀌어 이관 대상도 달라질 수 있다). 그 점은 논문에 명시할 것.

    python3 sliver_referral.py --views <런들...> --bases product_rule sum_rule
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

from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, CertifiedScorer, _H,
                                             default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import _fold_data, _pairs, _split, sliver


def base_parts(P, base_fn):
    """features() 와 같은 margin/eps 를 기반만 갈아끼워 계산한다."""
    J, n, K = P.shape
    pb = base_fn(P)
    from olive_if_v18.lacf_v2 import _margin_eps      # 단위 전역(LACF_UNITS)을 따른다
    margin, eps = _margin_eps(P, pb)
    return pb, margin, eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--bases", nargs="+", default=["product_rule", "sum_rule"])
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    dC = float(C[1:, 0].max())
    pairs = _pairs(keys, None)
    seeds = sorted({s for s, _ in pairs})
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · K={K} · ΔC={dC}")
    print(f"시드 {seeds}\n")

    rows = []
    for s, f in pairs:
        P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
        _, conf_i = _split(y_cal, s, f)
        for bname in a.bases:
            op = OPERATORS[bname]
            pb, margin, eps = base_parts(P_tst, op)
            onS = sliver(margin, eps).astype(bool)

            sc = CertifiedScorer(C, a.alpha).fit(op(P_cal[:, conf_i]), y_cal[conf_i])
            acts = sc.act(pb)
            referred = acts == K
            auto = ~referred

            real = C[y_tst, acts]
            best_possible = C[y_tst, :].min(1)           # 정답을 알 때 고를 수 있는 최선
            slack = real - best_possible                 # 이미지별 개선 여지

            # 가설 2: 뷰 혼합이 구조적으로 무력한가.
            # p_mix 는 뷰들의 볼록결합이므로, 모든 뷰가 같은 비용최적 행동을 주면
            # 어떤 가중치로도 행동을 바꿀 수 없다. 잔차항만 남는데 그건 유계다.
            va = np.stack([np.argmin(P_tst[j] @ C[:, :K], 1) for j in range(P_tst.shape[0])])
            agree = (va == va[0]).all(0)
            rows.append({
                "views_agree": float(agree.mean()),
                "views_agree_on_S": float(agree[onS].mean()) if onS.any() else np.nan,
                "views_agree_S_auto": float(agree[onS & auto].mean()) if (onS & auto).any() else np.nan,
                "seed": s, "fold": f, "base": bname,
                "pi_S": float(onS.mean()),
                "refer_rate": float(referred.mean()),
                "P_refer_given_S": float(referred[onS].mean()) if onS.any() else np.nan,
                "P_S_given_refer": float(onS[referred].mean()) if referred.any() else np.nan,
                "pi_S_eff": float((onS & auto).mean()),
                "Lambda": dC * float(onS.mean()),
                "Lambda_eff": dC * float((onS & auto).mean()),
                "cost": float(real.mean()),
                "ceil_all": float(slack.mean()),
                "ceil_S_auto": float(slack[onS & auto].sum() / len(slack)) if (onS & auto).any() else 0.0,
                "ceil_auto": float(slack[auto].sum() / len(slack)) if auto.any() else 0.0,
            })
        print(f"  seed {s} fold {f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "sliver_referral_perfold_v18.csv", index=False)
    g = df.groupby("base").mean(numeric_only=True).drop(columns=["seed", "fold"]).reset_index()
    g.to_csv(out / "sliver_referral_v18.csv", index=False)

    print("\n" + "=" * 96)
    print("sliver 와 전문가 이관 영역의 관계")
    print("=" * 96)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    for _, r in g.iterrows():
        p = r["P_refer_given_S"]
        print(f"\n  [기반 {r['base']}]")
        print(f"    sliver 의 {100*p:.1f}% 가 이미 전문가로 이관됩니다 "
              f"(전체 이관율 {100*r['refer_rate']:.1f}%).")
        print(f"    손댈 수 있는 영역은 π_S={r['pi_S']:.3f} 가 아니라 "
              f"π_S_eff={r['pi_S_eff']:.3f} 입니다.")
        print(f"    Λ {r['Lambda']:.3f}  →  Λ_eff {r['Lambda_eff']:.3f}  "
              f"({100*(1-r['Lambda_eff']/r['Lambda']) if r['Lambda'] else 0:.0f}% 축소)")
        print(f"    정답을 알 때의 절감 상한: 전체 {r['ceil_all']:.4f} 중")
        print(f"      자동처리 sliver 에서 올 수 있는 몫 {r['ceil_S_auto']:.4f} "
              f"← 국소 규칙이 가져갈 수 있는 최대치")
        if p > 0.8:
            print(f"    → 가설 성립. sliver 는 사실상 이관 영역입니다. 융합 연산자가")
            print(f"      실현 비용을 바꿀 수 없는 구조적 이유가 확인됩니다.")
        elif p > 0.5:
            print(f"    → 부분 성립. 절반 이상이 이관되지만 전부는 아닙니다.")
            print(f"      Λ_eff 로 다시 재는 것은 타당하나, 나머지 {100*(1-p):.0f}% 에서")
            print(f"      왜 이득이 없었는지는 별도 설명이 필요합니다.")
        else:
            print(f"    → 가설 1 불성립. 이관만으로는 설명되지 않습니다.")
        ag = r["views_agree_S_auto"]
        print(f"    [가설 2] 자동처리 sliver 에서 뷰 8개가 같은 행동을 지목하는 비율 {100*ag:.1f}%")
        if ag > 0.9:
            print(f"      → 혼합이 구조적으로 무력합니다. p_mix 는 뷰들의 볼록결합이라,")
            print(f"        모든 뷰가 같은 행동을 주면 어떤 가중치로도 바꿀 수 없습니다.")
            print(f"        남는 것은 유계 잔차뿐이고 그것만으로는 행동이 안 바뀝니다.")
        elif ag > 0.6:
            print(f"      → 부분적으로 무력합니다. 나머지 {100*(1-ag):.0f}% 에서는 혼합이")
            print(f"        행동을 바꿀 수 있었는데 이득이 없었다는 뜻이므로, 그 부분은")
            print(f"        여전히 '바꿀 가치가 없었다' 로 설명해야 합니다.")
        else:
            print(f"      → 혼합은 행동을 바꿀 수 있었습니다. 가설 2 도 아닙니다.")

    print("\n  주의: 이관 배정을 기반 확률로 고정한 채 계산했습니다. 규칙이 p 를 바꾸면")
    print("        VOI 순위가 바뀌어 이관 대상도 달라질 수 있으므로 보수적 추정입니다.")
    print("        원고에 이 점을 명시할 것.")

    (out / "sliver_referral_v18.json").write_text(json.dumps(
        {"alpha": a.alpha, "delta_C": dC, "n_folds": len(pairs),
         "seeds": [int(x) for x in seeds], "bases": a.bases,
         "referral_held_fixed": True,
         "rows": g.to_dict("records")}, indent=1, ensure_ascii=False))
    print("\n예치: sliver_referral_v18.csv · sliver_referral_v18.json · sliver_referral_perfold_v18.csv")


if __name__ == "__main__":
    main()
