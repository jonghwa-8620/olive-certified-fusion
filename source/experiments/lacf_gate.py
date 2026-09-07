#!/usr/bin/env python3
"""게이트가 닫힌 이유를 가른다 — 문제점 #1.

무엇이 걸려 있나
----------------
product_rule 기반에서 LACF 는 10/10 폴드에서 기반과 정확히 일치했다(SD 0.0000).
해석이 두 가지다.

  (a) 모델이 "얻을 게 없다"를 옳게 학습했다.          ← 논문의 주장
  (b) 용량 벌점 γ 가 세서 데이터와 무관하게 닫혔다.    ← 하이퍼파라미터 인공물

(b) 라면 논문의 결론은 우리가 고른 γ=0.05 가 만든 것이지 문제의 성질이 아니다.
현재 이 둘을 가를 증거가 없다. 절제표의 no-capacity-penalty 는 γ=0 한 점만 주고,
게이트가 실제로 무엇을 했는지는 보고하지 않는다.

이 스크립트는 γ 를 쓸어가며 *게이트의 행동 자체*를 기록한다:

  gate_on_S     sliver 위 평균 g(x)      — 게이트가 열려는 있는가
  frac_changed  결정이 기반과 달라진 비율 — 열린 게이트가 실제로 결정을 바꿨는가
  move_on_S     sliver 위 평균 |p − p_base|₁ — 확률을 얼마나 움직였는가
  cost          그래서 이득을 봤는가

판정
----
  γ=0 에서 게이트가 활짝 열리고(gate_on_S↑, frac_changed↑) 비용이 그대로거나
  나빠지면  →  (a). 바꿀 수는 있었지만 바꿔봐야 소용없었다. 결론이 견고하다.

  γ=0 에서 게이트가 열리고 비용이 MDE 넘게 좋아지면  →  (b). 우리 결론은 γ 가
  만든 것이었다. 그러면 γ 를 다시 고르고 본 실행을 다시 해야 한다.

  γ=0 에서도 게이트가 닫혀 있으면  →  둘 다 아니고 최적화 문제다. 초기화나
  스케일을 봐야 한다.

    python3 lacf_gate.py --views <런들...> --base product_rule --out artifacts_v18_real
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
from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, _H, default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import LACFv2, _fold_data, _pairs, _score, _split

Z80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))


def make_features(base_fn):
    """lacf_base.py 와 같은 기반 주입. 패치된 패키지라면 set_base 를 대신 쓴다."""
    def features(P):
        J, n, K = P.shape
        pb = base_fn(P)
        mx = P.max(-1).T
        js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                      for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
        from olive_if_v18.lacf_v2 import _margin_eps      # 단위 전역(LACF_UNITS)을 따른다
        margin, eps = _margin_eps(P, pb)
        return np.c_[mx, js, margin, _H(P).mean(0), eps], pb, margin, eps
    return features


def decisions(p, C):
    """비용 최소 행동. C 는 (K, K+1) 이라 마지막 열은 전문가 이관."""
    return np.argmin(p @ C[:, :C.shape[0]], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0.0, 0.005, 0.05, 0.5])
    a = ap.parse_args()
    if a.base not in OPERATORS:
        sys.exit(f"알 수 없는 기반 '{a.base}'. 가능: {sorted(OPERATORS)}")

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    pairs = _pairs(keys, None)
    seeds = sorted({s for s, _ in pairs})
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · K={K} · 기반 {a.base}")
    print(f"시드 {seeds}" + ("   ← 하나뿐: SD 는 폴드 편차" if len(seeds) == 1 else ""))
    print(f"γ 스윕 {a.gammas} · H={a.hidden} · maxiter={a.maxiter}   (기본 γ=0.05)\n")

    orig = L.features
    rows = []
    try:
        L.features = make_features(OPERATORS[a.base])
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            base_cost = _score(OPERATORS[a.base], P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)["cost"]
            _, pb_t, margin, eps = L.features(P_tst)
            onS = L.sliver(margin, eps).astype(bool)
            d_base = decisions(pb_t, C)
            for g0 in a.gammas:
                m = LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter, hidden=a.hidden,
                           gamma=g0).fit(P_cal[:, fit_i], y_cal[fit_i])
                r = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)
                p, lam, gate, pb = m.forward(P_tst, return_parts=True)
                p = np.asarray(p); gate = np.asarray(gate)
                d_new = decisions(p, C)
                rows.append({
                    "seed": s, "fold": f, "gamma": g0,
                    "cost": r["cost"], "base_cost": base_cost,
                    "delta": r["cost"] - base_cost,
                    "violation": r.get("violation", np.nan),
                    "pi_S": float(onS.mean()),
                    "gate_on_S": float(gate[onS].mean()) if onS.any() else np.nan,
                    "gate_off_S": float(gate[~onS].mean()) if (~onS).any() else np.nan,
                    "frac_changed": float((d_new != d_base).mean()),
                    "frac_changed_on_S": float((d_new != d_base)[onS].mean()) if onS.any() else np.nan,
                    "move_on_S": float(np.abs(p - pb)[onS].sum(1).mean()) if onS.any() else np.nan,
                })
            print(f"  seed {s} fold {f}  π_S={onS.mean():.3f}  기반 {base_cost:.4f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "lacf_gate_perfold_v18.csv", index=False)

    g = df.groupby("gamma").agg(
        cost=("cost", "mean"), base=("base_cost", "mean"),
        delta=("delta", "mean"), delta_sd=("delta", "std"),
        gate_on_S=("gate_on_S", "mean"), gate_off_S=("gate_off_S", "mean"),
        frac_changed=("frac_changed", "mean"),
        frac_changed_on_S=("frac_changed_on_S", "mean"),
        move_on_S=("move_on_S", "mean"), violation=("violation", "mean")).reset_index()
    n = df["fold"].nunique() * max(len(seeds), 1)
    g["mde"] = Z80 * g["delta_sd"] / np.sqrt(max(n, 1))
    g["resolvable"] = g["delta"].abs() > g["mde"]
    g.to_csv(out / "lacf_gate_v18.csv", index=False)

    print("\n" + "=" * 92)
    print(f"게이트 진단 — 기반 {a.base}, 폴드 {n}개")
    print("=" * 92)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    z = g[g.gamma == 0.0]
    print("\n" + "=" * 92)
    print("판정")
    print("=" * 92)
    if len(z):
        z = z.iloc[0]
        opened = (z.gate_on_S > 0.2) or (z.frac_changed_on_S > 0.02)
        better = (z.delta < 0) and z.resolvable
        print(f"  γ=0 에서   게이트(S 위) {z.gate_on_S:.3f}   결정변경(S 위) {z.frac_changed_on_S:.3f}"
              f"   Δ비용 {z.delta:+.4f}  (MDE {z.mde:.4f})")
        if opened and not better:
            print("\n  → (a). 게이트는 열렸고 결정도 바꿨지만 이득이 없다.")
            print("     닫힌 것은 γ 때문이 아니라 바꿀 가치가 없어서다. 결론이 견고하다.")
        elif opened and better:
            print("\n  → (b). γ=0 에서 실제로 좋아진다. 지금 결론은 γ=0.05 가 만든 것이다.")
            print("     ★ 본 실행을 γ 를 다시 골라 재실행해야 한다. 원고를 쓰기 전에.")
        else:
            print("\n  → 어느 쪽도 아니다. γ=0 에서도 게이트가 닫혀 있으면 최적화 문제다")
            print("     (초기화, 특징 스케일, 기울기 소실). 결론을 쓰기 전에 원인을 봐야 한다.")
    print("\n  주의: frac_changed 는 결정이 달라진 비율일 뿐 좋아졌다는 뜻이 아니다.")
    print("        판정은 Δ비용과 MDE 로만 한다.")

    (out / "lacf_gate_v18.json").write_text(json.dumps(
        {"base": a.base, "n_folds": int(n), "seeds": [int(x) for x in seeds],
         "hidden": a.hidden, "maxiter": a.maxiter, "default_gamma": 0.05,
         "rows": g.to_dict("records")}, indent=1, ensure_ascii=False))
    print("\n예치: lacf_gate_v18.csv · lacf_gate_v18.json · lacf_gate_perfold_v18.csv")


if __name__ == "__main__":
    main()
