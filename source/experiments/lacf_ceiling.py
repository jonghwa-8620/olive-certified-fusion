#!/usr/bin/env python3
"""LACF 가 진 이유가 '학습이 부족해서'가 아님을 보이는 세 실험.

게이트 미통과 자체는 논문에 유리하다 -- 연산자가 레버가 아니라는 논지를 깨려고
만든 모델조차 동률에 머문다는 뜻이니까. 다만 심사자는 즉시 이렇게 묻는다:

    "덜 학습시켜 놓고 안 된다고 하는 것 아닌가?"

그 질문을 세 방향에서 막는다.

  A. 천장(oracle)   테스트 폴드에 직접 적합시킨다. 반칙이지만 상한이다.
                    이 모델족이 이 데이터에서 살 수 있는 최대 이득이 얼마인가.
                    천장조차 작으면 학습을 아무리 잘해도 소용없다는 뜻이다.
  B. 용량 확대      hidden 4..64, maxiter 40..500. 기본값(H=8, 40회)이 좁다.
                    어떤 설정에서도 이기지 못함을 보인다.
  C. 해상도(MDE)    이 비교의 최소검출효과. 관측된 0.02 차이가 애초에
                    이 설계로 분해 가능한 크기인지 말한다. Theorem 4 와 같은 논리.

    python3 lacf_ceiling.py --views <런들...> --out artifacts_v18_real
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

from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, SEEDS, CertifiedScorer,
                                             default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import LACFv2, OURS, _fold_data, _pairs, _score, _split
from olive_if_v18.theorem9_lacf import features, sliver


def mde(sd_per_fold: float, n_folds: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """이 논문이 §10 에서 쓰는 것과 같은 관례: (z_{α/2}+z_β)·SD/√n."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd_per_fold / np.sqrt(max(n_folds, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--hidden", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--maxiter", type=int, nargs="+", default=[40, 150, 500])
    ap.add_argument("--base", default="product_rule",
                    help="기반 연산자. v18 원 실행은 이 인자가 없어 sum_rule 로 돌았다")
    a = ap.parse_args()
    from olive_if_v18.lacf_v2 import set_base
    set_base(a.base)
    print(f"기반 연산자: {a.base}")

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    pairs = _pairs(keys, None)
    print(f"뷰 {len(per_view)}개 · (시드,폴드) 쌍 {len(pairs)}개 · K={K} · α={a.alpha}")
    seeds = sorted({s for s, _ in pairs})
    print(f"시드 {seeds}  ← 하나뿐이면 SD 는 시드 편차가 아니라 폴드 편차입니다\n")

    ceil_rows, cap_rows = [], []
    for s, f in pairs:
        P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
        fit_i, conf_i = _split(y_cal, s, f)
        _, _, margin, eps = features(P_tst)
        pi_S = float(sliver(margin, eps).mean())
        Lambda = float(C[1:, 0].max() * pi_S)

        # ---- A. 천장: 테스트에 직접 적합 (반칙, 상한용) -------------------------
        oracle = LACFv2(C, a.alpha, seed=s, maxiter=500, hidden=32).fit(P_tst, y_tst)
        r = _score(oracle, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)
        r.update({"method": "LACF oracle (fit on test)", "seed": s, "fold": f,
                  "Lambda": Lambda, "pi_S": pi_S}); ceil_rows.append(r)
        # 같은 폴드의 최고 전역 연산자 (천장을 비교할 기준)
        best = min(((n, _score(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha))
                    for n, op in OPERATORS.items()), key=lambda t: t[1]["cost"])
        rb = dict(best[1]); rb.update({"method": f"best global ({best[0]})", "seed": s, "fold": f,
                                       "Lambda": Lambda, "pi_S": pi_S}); ceil_rows.append(rb)

        # ---- B. 용량 확대 ---------------------------------------------------------
        for H in a.hidden:
            for it in a.maxiter:
                m = LACFv2(C, a.alpha, seed=s, maxiter=it, hidden=H).fit(P_cal[:, fit_i], y_cal[fit_i])
                r = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)
                r.update({"hidden": H, "maxiter": it, "seed": s, "fold": f,
                          "n_params": m.n_params}); cap_rows.append(r)
        print(f"  seed {s} fold {f}  π_S={pi_S:.3f}  Λ={Lambda:.3f}", flush=True)

    # ------------------------------------------------------------------ 천장 요약
    ce = pd.DataFrame(ceil_rows)
    piv = ce.pivot_table(index=["seed", "fold"], columns="method", values="cost")
    oc = [c for c in piv.columns if c.startswith("LACF oracle")][0]
    bg = [c for c in piv.columns if c.startswith("best global")]
    piv["best_global"] = piv[bg].min(axis=1)
    piv["ceiling_gain"] = piv["best_global"] - piv[oc]
    ceil_sd = float(piv["ceiling_gain"].std(ddof=1))
    ceil_mde = mde(ceil_sd, len(piv))
    summ = {
        "n_folds": int(len(piv)),
        "seeds": [int(x) for x in seeds],
        "pi_S_mean": float(ce["pi_S"].mean()),
        "Lambda_mean": float(ce["Lambda"].mean()),
        "oracle_cost_mean": float(piv[oc].mean()),
        "best_global_cost_mean": float(piv["best_global"].mean()),
        "ceiling_gain_mean": float(piv["ceiling_gain"].mean()),
        "ceiling_gain_sd": ceil_sd,
        "ceiling_gain_mde": ceil_mde,
        "ceiling_gain_resolvable": bool(abs(piv["ceiling_gain"].mean()) > ceil_mde),
        "ceiling_gain_vs_Lambda": float(piv["ceiling_gain"].mean() / ce["Lambda"].mean()),
    }
    ce.to_csv(out / "lacf_ceiling_perfold_v18.csv", index=False)
    (out / "lacf_ceiling_v18.json").write_text(json.dumps(summ, indent=1))

    # ------------------------------------------------------------------ 용량 요약
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(out / "lacf_capacity_perfold_v18.csv", index=False)
    cs = cap.groupby(["hidden", "maxiter"]).agg(cost=("cost", "mean"), cost_sd=("cost", "std"),
                                                violation=("violation", "mean"),
                                                n_params=("n_params", "first")).reset_index()
    cs.to_csv(out / "lacf_capacity_v18.csv", index=False)

    print("\n" + "=" * 74)
    print("A. 천장 (테스트에 직접 적합한 오라클)")
    print("=" * 74)
    print(f"  최고 전역 연산자   {summ['best_global_cost_mean']:.4f}")
    print(f"  LACF 오라클        {summ['oracle_cost_mean']:.4f}")
    print(f"  천장 이득          {summ['ceiling_gain_mean']:+.4f}  (SD {ceil_sd:.4f}, n={len(piv)})")
    print(f"  이 비교의 MDE      {ceil_mde:.4f}")
    print(f"  분해 가능한가      {'예' if summ['ceiling_gain_resolvable'] else '아니오 — 설계 해상도 아래'}")
    print(f"  Λ 대비             {summ['ceiling_gain_vs_Lambda']:.3f}   (π_S={summ['pi_S_mean']:.3f}, Λ={summ['Lambda_mean']:.3f})")

    print("\n" + "=" * 74)
    print("B. 용량 확대 — 어떤 설정도 최고 전역 연산자를 못 이기면 '덜 학습' 반론이 막힌다")
    print("=" * 74)
    best_g = summ["best_global_cost_mean"]
    cs["beats_best_global"] = cs["cost"] < best_g
    print(cs.to_string(index=False))
    print(f"\n  최고 전역 연산자 {best_g:.4f} 를 이긴 설정: {int(cs['beats_best_global'].sum())} / {len(cs)}")
    print(f"  최저 비용 설정: H={cs.loc[cs.cost.idxmin(),'hidden']:.0f}, "
          f"maxiter={cs.loc[cs.cost.idxmin(),'maxiter']:.0f} → {cs.cost.min():.4f}")

    print("\n" + "=" * 74)
    print("C. 해상도")
    print("=" * 74)
    print(f"  관측된 LACF-vs-최고전역 차이가 MDE({ceil_mde:.4f}) 보다 작으면,")
    print(f"  '동률'은 모델의 실패가 아니라 설계의 해상도 문제다 — Theorem 4 와 같은 논리.")
    print(f"\n예치: lacf_ceiling_v18.json · lacf_ceiling_perfold_v18.csv · lacf_capacity_v18.csv")


if __name__ == "__main__":
    main()
