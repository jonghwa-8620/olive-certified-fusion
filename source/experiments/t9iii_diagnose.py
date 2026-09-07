#!/usr/bin/env python3
"""Theorem 9(iii) 기계 검증 실패의 원인을 가른다 — 정리가 틀렸나 검사가 틀렸나.

무슨 일이 있었나
----------------
product_rule 기반 본 실행에서 T9.iii 가 FAIL 했다:

    median ratio 1.148  vs  예측 sqrt(pi_S) 0.307      (10 폴드 중 7개가 1 초과)

정리는 제한 클래스의 Rademacher 복잡도가 전역의 sqrt(pi_S) 배라고 말하는데
측정값은 오히려 크다. 방향이 반대다. 그런데 검사 구현을 읽으면 네 가지 결함이
있고, 넷 다 실패를 만들 수 있다.

  (1) 중심이 다르다 -- 가장 큰 결함
      제한 후보는 use_gate=True 로 적합한 model.theta 주위에서,
      전역 후보는 use_gate=False 로 따로 적합한 glob.theta 주위에서 섭동한다.
      비율에 '제한의 효과' 와 '중심이 다른 효과' 가 섞인다. 정리는 같은 가설족을
      제한해서 쓸 때와 전역으로 쓸 때의 비교다.

  (2) 결정 규칙이 논문과 다르다
      검사는 acts = p.argmax(1) (확률 최대) 을 쓰는데 논문의 결정은
      argmin p @ C[:, :K] (비용 최소) 다. 정리의 ell 은 실현 비용 손실이다.

  (3) 증명 전제 위반
      증명 스케치는 contraction lemma 를 쓴다. 그건 손실이 Lipschitz 일 때
      성립하는데 argmax 는 계단함수다.

  (4) 표본이 작다
      n_mc=64, n_candidates=24. 24개 함수의 sup 을 64회로 추정한다.
      폴드별 값이 0.33~1.53 으로 튀는 것이 그 징후다.

이 스크립트가 하는 일
---------------------
  A. 통제된 재검증   같은 fitted theta 에서 같은 섭동을 뽑고, sliver 지시함수만
                     켜고 끄어 두 클래스를 만든다. 중심도 잡음도 동일하므로
                     비율은 오직 제한의 효과다. 결정 규칙은 논문과 같은 비용 최소.
  B. 추정량 수렴     n_mc 와 후보 수를 키우며 값이 어디로 수렴하는지 본다.
                     64/24 가 너무 작았는지가 여기서 드러난다.
  C. 척도 민감도     섭동 sigma 를 바꿔도 비율이 안정한지.
  D. 손실 정의 대조  argmax 손실(현행) vs argmin-비용 손실(논문). 검사가 다른
                     양을 재고 있었는지 직접 보여준다.

판정
----
  A 에서 비율 <= 1 이고 sqrt(pi_S) 근처면  -> 검사가 틀렸다. 검사를 고치고
      정리는 유지한다. 원고에는 고친 검사 결과를 싣는다.
  A 에서도 비율 > 1 이면                    -> 정리 (iii) 이 이 자료에서 지지되지
      않는다. (i)(ii) 만 남기고 (iii) 은 주장에서 빼야 한다. 숫자를 고르지 말 것.

    python3 t9iii_diagnose.py --views <런들...> --out out
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

import olive_if_v18.lacf_v2 as L
from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, _H, default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import LACFv2, _fold_data, _pairs, _split, sliver


def rademacher(losses: np.ndarray, n_mc: int, rng) -> float:
    """(n, m) 손실 행렬의 경험적 Rademacher 복잡도. m 개 후보에 대한 sup."""
    n = losses.shape[0]
    sig = rng.choice([-1.0, 1.0], size=(n_mc, n))
    return float(np.mean(np.max(sig @ losses, axis=1) / n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--n-mc", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--n-cand", type=int, nargs="+", default=[24, 100, 400])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    pairs = _pairs(keys, None)
    base_fn = OPERATORS[a.base]
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · 기반 {a.base}")
    print(f"n_mc {a.n_mc} · 후보수 {a.n_cand} · sigma {a.sigmas}")
    print(f"현행 검사값: n_mc=64, 후보 24, sigma 0.3, argmax 손실, 중심 분리\n")

    orig = L.features

    def feats(P):
        J, n, Kc = P.shape
        pb = base_fn(P)
        mx = P.max(-1).T
        js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                      for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
        from olive_if_v18.lacf_v2 import _margin_eps      # 단위 전역(LACF_UNITS)을 따른다
        margin, eps = _margin_eps(P, pb)
        return np.c_[mx, js, margin, _H(P).mean(0), eps], pb, margin, eps

    def act_cost(p):          # 논문의 결정: 기대비용 최소
        return np.argmin(np.asarray(p) @ C[:, :K], 1)

    rows = []
    try:
        L.features = feats
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            m = LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter).fit(P_cal[:, fit_i], y_cal[fit_i])
            _, pb, margin, eps = feats(P_tst)
            lam = sliver(margin, eps).astype(float)
            pi_S = float(lam.mean())
            base_act = {"cost": act_cost(pb), "argmax": np.asarray(pb).argmax(1)}

            rng = np.random.default_rng(s * 7919 + f)
            big = max(a.n_cand)
            for sg in a.sigmas:
                # ---- 같은 theta, 같은 섭동. sliver 지시함수만 켜고 끈다.
                perts = [m.theta + rng.normal(0, sg, m.theta.shape) for _ in range(big)]
                LS = {"cost": [], "argmax": []}
                LG = {"cost": [], "argmax": []}
                for th in perts:
                    m.use_gate = True
                    pr = np.asarray(m.forward(P_tst, th))
                    m.use_gate = False
                    pg = np.asarray(m.forward(P_tst, th))
                    m.use_gate = True
                    for rule, actf in (("cost", act_cost),
                                       ("argmax", lambda q: np.asarray(q).argmax(1))):
                        LS[rule].append(C[y_tst, actf(pr)] - C[y_tst, base_act[rule]])
                        LG[rule].append(C[y_tst, actf(pg)] - C[y_tst, base_act[rule]])
                for rule in ("cost", "argmax"):
                    Ms = np.stack(LS[rule], 1); Mg = np.stack(LG[rule], 1)
                    for nc in a.n_cand:
                        for nm in a.n_mc:
                            r2 = np.random.default_rng(s * 31 + f)
                            RS = rademacher(Ms[:, :nc], nm, r2)
                            r2 = np.random.default_rng(s * 31 + f)
                            RG = rademacher(Mg[:, :nc], nm, r2)
                            rows.append({"seed": s, "fold": f, "pi_S": pi_S,
                                         "sqrt_pi_S": float(np.sqrt(pi_S)),
                                         "sigma": sg, "rule": rule, "n_cand": nc, "n_mc": nm,
                                         "R_restricted": RS, "R_global": RG,
                                         "ratio": RS / RG if RG > 0 else np.nan})
            print(f"  seed {s} fold {f}  π_S={pi_S:.3f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "t9iii_diagnose_perfold_v18.csv", index=False)
    n = len(pairs)
    sq = float(df.sqrt_pi_S.median())

    def block(title, sub):
        g = sub.groupby(["rule", "sigma", "n_cand", "n_mc"]).agg(
            ratio_med=("ratio", "median"), ratio_mean=("ratio", "mean"),
            ratio_sd=("ratio", "std"),
            RS=("R_restricted", "mean"), RG=("R_global", "mean")).reset_index()
        g["E_ratio"] = g.RS / g.RG          # 기댓값 수준 비율 (정리가 말하는 것)
        print("\n" + "=" * 100); print(title); print("=" * 100)
        print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return g

    print(f"\n예측선  median sqrt(π_S) = {sq:.3f}")
    gc = block("A·B·C·D — 통제된 재검증 (같은 θ, 같은 섭동, sliver 지시함수만 차이)", df)
    gc.to_csv(out / "t9iii_diagnose_v18.csv", index=False)

    # 논문 결정 규칙 · 가장 큰 표본에서의 값이 판정 기준
    best = gc[(gc.rule == "cost") & (gc.n_cand == max(a.n_cand)) & (gc.n_mc == max(a.n_mc))]
    print("\n" + "=" * 100); print("판정"); print("=" * 100)
    for _, r in best.iterrows():
        ok = r.E_ratio <= 1.0
        print(f"  σ={r.sigma}  기댓값비율 E[R_S]/E[R_G] = {r.E_ratio:.3f}   "
              f"폴드중앙값 {r.ratio_med:.3f}   예측 {sq:.3f}   {'≤1 만족' if ok else '★ 1 초과'}")
    allok = bool((best.E_ratio <= 1.0).all())
    print()
    if allok:
        print("  → 통제된 비교에서는 비율이 1 이하입니다. 실패는 정리가 아니라 검사의")
        print("    결함(중심 분리 · argmax 손실 · 작은 표본)에서 왔습니다.")
        print("    theorem9_lacf.py 의 검사를 이 방식으로 고치고 정리는 유지합니다.")
    else:
        print("  ★ 통제해도 1 을 넘습니다. Theorem 9(iii) 은 이 자료에서 지지되지 않습니다.")
        print("    (i)(ii) 만 남기고 (iii) 은 주장에서 빼야 합니다. LACF 가 이기지 않으므로")
        print("    (iii) 이 맡던 '이기는 메커니즘' 설명은 어차피 필요 없습니다.")
    print("\n  참고: rule 열의 argmax 는 현행 검사가 쓰던 손실입니다. cost 행과 크게")
    print("        다르면 현행 검사가 정리의 ℓ 이 아닌 다른 양을 재고 있었다는 뜻입니다.")
    print("        n_mc·n_cand 를 키울수록 값이 안정되는지도 함께 보세요.")

    (out / "t9iii_diagnose_v18.json").write_text(json.dumps(
        {"base": a.base, "n_folds": n, "median_sqrt_pi_S": sq,
         "controlled_ratio_ok": allok,
         "rows": gc.to_dict("records")}, indent=1, ensure_ascii=False))
    print("\n예치: t9iii_diagnose_v18.csv · t9iii_diagnose_v18.json · t9iii_diagnose_perfold_v18.csv")


if __name__ == "__main__":
    main()
