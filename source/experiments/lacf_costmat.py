#!/usr/bin/env python3
"""비용행렬 민감도 — 문제점 #2.

무엇이 걸려 있나
----------------
논문의 결과 전체가 비용행렬 하나에 의존한다:

    default_cost_matrix(K, miss=(1.6, 7.2, 21.2), wrong=2.0, spray=1.0, expert_cost=0.6)

이 논문의 문제를 비대칭으로 만드는 게 바로 그 행렬인데, 그것에 대한 민감도가
없다. "연산자는 레버가 아니다" 가 이 행렬에서만 참일 가능성을 배제하지 못한다.
심사자가 가장 쉽게 찌를 구멍이다.

무엇을 재는가
-------------
미검출 비용(miss)을 배수 r 로 쓸어간다. Λ = ΔC·π_S 이므로 r 을 키우면 이론적
레버리지가 비례해 커진다. π_S 는 사후확률만의 함수라 r 과 무관하다 -- 즉

    sliver 는 코퍼스가 정하고, 그 위의 레버리지 크기는 비용이 정한다.

그래서 질문이 선명해진다: **Λ 를 8배로 키워도 실현 이득이 0인가?**

  0 이 유지되면   → 결론이 비용행렬의 산물이 아니다. #2 가 막힌다.
  어느 r 부터 이득이 MDE 를 넘으면 → 그 임계값이 새 기여다. 음성 결과가
     "레버리지는 언제 실현 가능해지는가" 라는 양성 결과로 바뀐다.

비용 단위가 r 마다 다르므로 절대 비용은 r 사이에 비교할 수 없다. 비교는
반드시 (기반 − LACF)/Λ 로 한다 -- Λ 로 정규화한 레버리지 활용률이다.

    python3 lacf_costmat.py --views <런들...> --base product_rule --out artifacts_v18_real
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
from olive_if_v18.lacf_v2 import LACFv2, _fold_data, _pairs, _score, _split

Z80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))
MISS0 = (1.6, 7.2, 21.2)          # 원 논문의 미검출 비용 (EUR/tree)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="미검출 비용 배수. 1.0 이 논문의 행렬.")
    ap.add_argument("--expert", type=float, nargs="+", default=None,
                    help="전문가 이관 비용도 함께 쓸 때 (기본 0.6). 주면 스윕이 격자가 된다")
    a = ap.parse_args()
    if a.base not in OPERATORS:
        sys.exit(f"알 수 없는 기반 '{a.base}'. 가능: {sorted(OPERATORS)}")

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    pairs = _pairs(keys, None)
    seeds = sorted({s for s, _ in pairs})
    experts = a.expert if a.expert else [0.6]
    grid = [(r, e) for r in a.scales for e in experts]
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · K={K} · 기반 {a.base}")
    print(f"시드 {seeds}" + ("   ← 하나뿐: SD 는 폴드 편차" if len(seeds) == 1 else ""))
    print(f"미검출 배수 {a.scales} · 전문가비용 {experts} · 설정 {len(grid)}개\n")

    orig = L.features
    rows = []
    try:
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            for r, e in grid:
                C = default_cost_matrix(K, miss=tuple(m * r for m in MISS0), expert_cost=e)
                # 비용 단위에서는 슬리버가 비용 행렬에 의존하므로 설정마다 다시 잰다.
                L.features = make_features(OPERATORS[a.base], C)
                _, pb_t, margin, eps = L.features(P_tst)
                onS = L.sliver(margin, eps).astype(bool)
                pi_S = float(onS.mean())
                Lam = float(C[1:, 0].max() * pi_S)
                # 이관 집합은 VOI 기반이라 *비용행렬에 의존한다*. §2.1 의 핵심 포함관계
                # S ⊆ 이관집합 이 다른 비용행렬에서도 성립하는지가 그 절의 일반성을 정한다.
                sc = CertifiedScorer(C, a.alpha).fit(OPERATORS[a.base](P_cal[:, conf_i]), y_cal[conf_i])
                acts = sc.act(pb_t); auto = acts < K
                pi_eff = float((onS & auto).mean())
                slack = C[y_tst, acts] - C[y_tst, :].min(1)
                base_c = sc.evaluate(OPERATORS[a.base](P_tst), y_tst)["cost"]
                m = LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter,
                           hidden=a.hidden).fit(P_cal[:, fit_i], y_cal[fit_i])
                rr = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)
                # 모든 고정 연산자 중 최고 (그 비용행렬에서)
                best = min(_score(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)["cost"]
                           for op in OPERATORS.values())
                rows.append({"seed": s, "fold": f, "miss_scale": r, "expert_cost": e,
                             "pi_S": pi_S, "Lambda": Lam,
                             "P_refer_given_S": float((~auto)[onS].mean()) if onS.any() else np.nan,
                             "refer_rate": float((~auto).mean()),
                             "pi_S_eff": pi_eff, "Lambda_eff": float(C[1:, 0].max() * pi_eff),
                             "ceil_S_auto": float(slack[onS & auto].sum() / len(slack)) if (onS & auto).any() else 0.0,
                             "base_cost": base_c, "lacf_cost": rr["cost"], "best_op_cost": best,
                             "gain": base_c - rr["cost"],
                             "gain_norm": (base_c - rr["cost"]) / Lam if Lam > 0 else np.nan,
                             "violation": rr.get("violation", np.nan)})
            print(f"  seed {s} fold {f}  π_S={pi_S:.3f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "lacf_costmat_perfold_v18.csv", index=False)

    key = ["miss_scale", "expert_cost"]
    g = df.groupby(key).agg(pi_S=("pi_S", "mean"), Lambda=("Lambda", "mean"),
                            P_refer_given_S=("P_refer_given_S", "mean"),
                            Lambda_eff=("Lambda_eff", "mean"),
                            ceil_S_auto=("ceil_S_auto", "mean"),
                            base=("base_cost", "mean"), lacf=("lacf_cost", "mean"),
                            best_op=("best_op_cost", "mean"),
                            gain=("gain", "mean"), gain_sd=("gain", "std"),
                            gain_norm=("gain_norm", "mean"),
                            violation=("violation", "mean")).reset_index()
    n = len(df) // len(grid)
    g["mde"] = Z80 * g["gain_sd"] / np.sqrt(max(n, 1))
    g["realised"] = g["gain"] > g["mde"]        # 이득이 해상도를 넘는가 (부호 포함)
    g.to_csv(out / "lacf_costmat_v18.csv", index=False)

    print("\n" + "=" * 100)
    print(f"비용행렬 민감도 — 기반 {a.base}, 폴드 {n}개")
    print("=" * 100)
    print("  gain = 기반 − LACF  (양수면 LACF 가 이득). gain_norm = gain/Λ, r 사이 비교는 이것으로.")
    print()
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    hit = g[g["realised"]]
    print("\n" + "=" * 100)
    print("판정")
    print("=" * 100)
    import os
    if os.environ.get("LACF_UNITS", "prob") == "cost":
        print(f"  π_S 가 {g.pi_S.min():.3f} ~ {g.pi_S.max():.3f} 로 움직입니다 — Theorem 4 의 슬리버는")
        print(f"  비용 행렬의 함수이므로 이것이 정상입니다(확률 단위 구현에서는 상수였고, 그 불변성이")
        print(f"  결함의 신호였습니다).")
    else:
        print(f"  π_S = {g.pi_S.iloc[0]:.3f} 로 모든 r 에서 동일 — sliver 는 사후확률의 성질이고")
    print(f"  Λ 는 {g.Lambda.min():.3f} 에서 {g.Lambda.max():.3f} 까지 {g.Lambda.max()/g.Lambda.min():.0f}배 움직였습니다.")

    pr = g["P_refer_given_S"]
    print(f"\n  [§2.1 의 일반성]  P(이관 | S) 가 비용행렬에 따라 "
          f"{pr.min():.3f} ~ {pr.max():.3f} 범위입니다.")
    if pr.min() > 0.95:
        print("     → 모든 비용 설정에서 sliver 가 사실상 이관 영역입니다. §2.1 의 포함관계가")
        print("       비용행렬 하나에 기댄 것이 아님이 확인됩니다. Λ_eff ≈ 0 이 일반적입니다.")
    else:
        lo = g.loc[pr.idxmin()]
        print(f"     → 포함관계가 깨지는 설정이 있습니다 (miss_scale {lo.miss_scale}, "
              f"expert {lo.expert_cost} 에서 {pr.min():.3f}).")
        print(f"       §2.1 은 '이 비용행렬에서' 로 범위를 묶어 써야 합니다. 그 설정의")
        print(f"       Λ_eff={lo.Lambda_eff:.3f}, 천장 {lo.ceil_S_auto:.4f} 도 함께 보고할 것.")
    if len(hit) == 0:
        print(f"\n  → 어떤 비용 설정에서도 실현 이득이 MDE 를 넘지 않습니다.")
        print(f"     이론적 레버리지를 {g.Lambda.max()/g.Lambda.min():.0f}배 키워도 실현분은 0입니다.")
        print(f"     결론이 비용행렬의 산물이 아님이 확인됩니다 — #2 가 막힙니다.")
    else:
        print(f"\n  → {len(hit)}개 설정에서 이득이 MDE 를 넘습니다:")
        print(hit[key + ["Lambda", "gain", "mde", "gain_norm"]].to_string(index=False))
        print("\n     ★ 임계값이 존재합니다. 음성 결과가 '레버리지는 언제 실현 가능해지는가'")
        print("        라는 양성 기여로 바뀝니다. 다만 그 설정이 현실적인 비용인지")
        print("        (r 이 크면 미검출 비용이 비현실적으로 큰 것) 반드시 논의해야 합니다.")
    print("\n  주의: 비용 단위가 r 마다 다릅니다. 절대 비용을 r 사이에 비교하지 마세요.")

    (out / "lacf_costmat_v18.json").write_text(json.dumps(
        {"base": a.base, "n_folds": int(n), "seeds": [int(x) for x in seeds],
         "miss_base": list(MISS0), "scales": a.scales, "expert_costs": experts,
         "hidden": a.hidden, "maxiter": a.maxiter,
         "any_realised": bool(len(hit) > 0),
         "rows": g.to_dict("records")}, indent=1, ensure_ascii=False))
    print("\n예치: lacf_costmat_v18.csv · lacf_costmat_v18.json · lacf_costmat_perfold_v18.csv")


if __name__ == "__main__":
    main()
