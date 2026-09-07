#!/usr/bin/env python3
"""자동화율 스윕 — 명제가 만든 예측을 검증한다. 이 실험이 논문의 부호를 바꾼다.

무엇이 밝혀졌나
---------------
sliver_referral.py 결과 (뷰 8개, 폴드 10, 기반 product_rule):

    P(이관 | sliver) = 1.0000      sliver 전체가 이미 전문가에게 간다
    pi_S_eff = 0.0000              손댈 수 있는 영역이 없다
    Lambda_eff = 0.0000            실현 가능한 레버리지가 0

이관된 이미지의 실현 비용은 C[y,K] 고정이므로, 국소 규칙이 그 위에서 무엇을
하든 비용이 변할 수 없다. γ 를 0 까지 내려 게이트를 열어도(0.76), 결정이 24%
바뀌어도, 비용은 10/10 폴드에서 정확히 0.1812 였다. 통계가 아니라 항등식이다.

우연이 아니다. 두 선택 규칙이 같은 양을 본다:
    sliver = margin ≤ 2ε              모델이 모르는 곳
    이관   = VOI 상위 (1−automation)  역시 모델이 모르는 곳
그리고 π_S = 0.094 < 이관율 0.208 이므로 sliver 가 이관 집합에 통째로 들어간다.

    명제. S ⊆ 이관집합 이면 모든 sliver-제한 규칙의 실현 비용은 기반과 같다.
          따라서 Λ = ΔC·π_S 는 과대평가이고, 옳은 양은
              Λ_eff = ΔC · π_S · P(자동 | S).

예측
----
자동화율 a 를 올리면 이관 예산 (1−a) 이 줄어, 어느 지점에서 sliver 가 이관
집합 밖으로 나온다. 그때부터 Λ_eff > 0 이 되고, **바로 그 지점부터 융합
연산자가 실제로 작동해야 한다.**

이 스크립트는 a 를 쓸어가며 Λ_eff(예측)와 실제 이득을 나란히 놓는다.

  예측이 맞으면   Λ_eff = 0 인 구간에서 이득 0, Λ_eff > 0 이 되는 지점부터 이득 발생.
                  → 음성 결과가 예측력 있는 양성 기여로 바뀐다. 높은 a 에서
                    LACF 가 전 연산자를 이기면 개선 게이트도 통과한다.
  예측이 틀리면   Λ_eff > 0 인데도 이득이 없다 → 이관만으로는 설명이 안 되고,
                  다른 원인(볼록결합의 한계 등)을 더 봐야 한다. 그것도 알아야 할 정보다.

주의: a 가 다르면 시스템이 다르다. 서로 다른 a 의 절대 비용을 "우리 방법이
      좋아졌다" 로 읽으면 안 된다. 각 a 안에서의 기반 대비 이득만 비교한다.

    python3 lacf_automation.py --views <런들...> --base product_rule --out artifacts_v18_real
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

import olive_if_v18.lacf_v2 as L
from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, CertifiedScorer, _H,
                                             default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import LACFv2, _fold_data, _pairs, _split, sliver

Z80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))


def make_features(base_fn, _COST_OVERRIDE=None):
    def features(P):
        J, n, K = P.shape
        pb = base_fn(P)
        mx = P.max(-1).T
        js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                      for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
        from olive_if_v18.lacf_v2 import _margin_eps   # 단위 전역(LACF_UNITS)을 따른다
        margin, eps = _margin_eps(P, pb, _COST_OVERRIDE)
        return np.c_[mx, js, margin, _H(P).mean(0), eps], pb, margin, eps
    return features


def score_at(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, alpha, automation):
    """_score 와 같되 자동화율을 지정한다."""
    sc = CertifiedScorer(C, alpha, automation=automation).fit(op(P_cal[:, conf_i]), y_cal[conf_i])
    return sc, sc.evaluate(op(P_tst), y_tst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--automation", type=float, nargs="+",
                    default=[0.79, 0.85, 0.90, 0.95, 1.00],
                    help="0.79 가 논문의 운용점")
    a = ap.parse_args()
    if a.base not in OPERATORS:
        sys.exit(f"알 수 없는 기반 '{a.base}'. 가능: {sorted(OPERATORS)}")

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    dC = float(C[1:, 0].max())
    pairs = _pairs(keys, None)
    seeds = sorted({s for s, _ in pairs})
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · K={K} · ΔC={dC} · 기반 {a.base}")
    print(f"시드 {seeds}" + ("   ← 하나뿐: SD 는 폴드 편차" if len(seeds) == 1 else ""))
    print(f"자동화율 {a.automation}   (0.79 = 논문 운용점)\n")

    orig = L.features
    rows = []
    try:
        L.features = make_features(OPERATORS[a.base])
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            _, pb_t, margin, eps = L.features(P_tst)
            onS = sliver(margin, eps).astype(bool)
            pi_S = float(onS.mean())

            # LACF 는 자동화율과 무관하게 학습된다 (손실은 이관을 모른다).
            # 자동화율은 평가 시점의 시스템 설정이므로 한 번 학습해 여러 a 로 잰다.
            m = LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter,
                       hidden=a.hidden).fit(P_cal[:, fit_i], y_cal[fit_i])
            lacf_op = lambda P, _m=m: np.asarray(_m.forward(P))

            for au in a.automation:
                sc_b, rb = score_at(OPERATORS[a.base], P_cal, y_cal, conf_i,
                                    P_tst, y_tst, C, a.alpha, au)
                acts_b = sc_b.act(pb_t)
                auto = acts_b < K
                pi_eff = float((onS & auto).mean())
                real = C[y_tst, acts_b]
                slack = real - C[y_tst, :].min(1)
                _, rl = score_at(lacf_op, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha, au)
                best_op = min(score_at(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha, au)[1]["cost"]
                              for op in OPERATORS.values())
                rows.append({
                    "seed": s, "fold": f, "automation": au,
                    "pi_S": pi_S, "pi_S_eff": pi_eff,
                    "Lambda": dC * pi_S, "Lambda_eff": dC * pi_eff,
                    "P_auto_given_S": float(auto[onS].mean()) if onS.any() else np.nan,
                    "ceil_S_auto": float(slack[onS & auto].sum() / len(slack)) if (onS & auto).any() else 0.0,
                    "base_cost": rb["cost"], "lacf_cost": rl["cost"],
                    "best_op_cost": best_op,
                    "gain": rb["cost"] - rl["cost"],
                    "gain_vs_best_op": best_op - rl["cost"],
                    "violation": rl.get("violation", np.nan),
                    "automation_realised": rl.get("automation", np.nan),
                })
            print(f"  seed {s} fold {f}  π_S={pi_S:.3f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "lacf_automation_perfold_v18.csv", index=False)

    n = len(pairs)
    g = df.groupby("automation").agg(
        pi_S=("pi_S", "mean"), pi_S_eff=("pi_S_eff", "mean"),
        P_auto_given_S=("P_auto_given_S", "mean"),
        Lambda_eff=("Lambda_eff", "mean"), ceil_S_auto=("ceil_S_auto", "mean"),
        base=("base_cost", "mean"), lacf=("lacf_cost", "mean"), best_op=("best_op_cost", "mean"),
        gain=("gain", "mean"), gain_sd=("gain", "std"),
        gain_vs_best=("gain_vs_best_op", "mean"), gain_vs_best_sd=("gain_vs_best_op", "std"),
        violation=("violation", "mean")).reset_index()
    g["mde"] = Z80 * g["gain_sd"] / np.sqrt(max(n, 1))
    g["mde_vs_best"] = Z80 * g["gain_vs_best_sd"] / np.sqrt(max(n, 1))
    g["realised"] = g["gain"] > g["mde"]
    g["beats_all"] = g["gain_vs_best"] > g["mde_vs_best"]
    g.to_csv(out / "lacf_automation_v18.csv", index=False)

    show = ["automation", "P_auto_given_S", "Lambda_eff", "ceil_S_auto",
            "base", "lacf", "best_op", "gain", "mde", "realised", "gain_vs_best", "beats_all"]
    print("\n" + "=" * 108)
    print(f"자동화율 스윕 — 기반 {a.base}, 폴드 {n}개")
    print("=" * 108)
    print("  Λ_eff = ΔC·π_S·P(자동|S) 가 예측, gain = 기반 − LACF 가 실측입니다.")
    print()
    print(g[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 108)
    print("판정")
    print("=" * 108)
    zero = g[g.Lambda_eff <= 1e-9]
    pos = g[g.Lambda_eff > 1e-9]
    if len(zero):
        bad = zero[zero.realised]
        print(f"  Λ_eff = 0 인 설정 {len(zero)}개: 이득이 MDE 를 넘은 것 {len(bad)}개"
              + ("   ← 예측대로" if len(bad) == 0 else "   ★ 예측 위반. 명제를 다시 봐야 합니다."))
    if len(pos):
        hit = pos[pos.realised]
        print(f"  Λ_eff > 0 인 설정 {len(pos)}개: 이득이 MDE 를 넘은 것 {len(hit)}개")
        if len(hit):
            print("\n  → 예측 성립. 레버리지가 열리는 지점부터 연산자가 실제로 작동합니다.")
            print(hit[["automation", "Lambda_eff", "gain", "mde", "gain_vs_best", "beats_all"]]
                  .to_string(index=False))
            if hit["beats_all"].any():
                w = hit[hit.beats_all]
                print(f"\n  ★ 그중 {len(w)}개 설정에서 LACF 가 13개 연산자 전부를 이깁니다"
                      f" (자동화율 {list(w.automation)}).")
                print("    개선 게이트 통과 조건입니다. 다만 그 자동화율이 운용상 타당한지")
                print("    (전문가 이관을 줄이면 놓친 병해가 늘어납니다) 반드시 함께 보고할 것.")
        else:
            print("\n  → 예측 불성립. 레버리지가 열려도 이득이 없습니다. 이관만으로는")
            print("    설명이 안 되니 다른 원인을 더 봐야 합니다 (볼록결합의 한계 등).")
            print("    논문에는 Λ_eff 정정만 쓰고 예측 부분은 빼는 것이 정직합니다.")

    print("\n  주의: 자동화율이 다르면 시스템이 다릅니다. 서로 다른 a 의 절대 비용을")
    print("        '좋아졌다' 로 읽지 마세요. 각 a 안에서의 기반 대비 이득만 비교합니다.")
    print("        이관을 줄이면 비용이 오르는 것이 정상입니다 -- 그게 이관의 값어치입니다.")

    (out / "lacf_automation_v18.json").write_text(json.dumps(
        {"base": a.base, "n_folds": n, "seeds": [int(x) for x in seeds],
         "hidden": a.hidden, "maxiter": a.maxiter, "delta_C": dC,
         "operating_point": 0.79, "rows": g.to_dict("records")},
        indent=1, ensure_ascii=False))
    print("\n예치: lacf_automation_v18.csv · lacf_automation_v18.json · lacf_automation_perfold_v18.csv")


if __name__ == "__main__":
    main()
