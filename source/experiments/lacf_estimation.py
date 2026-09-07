#!/usr/bin/env python3
"""레버리지가 열린 구간에서 왜 못 가져가는가 — 추정 실패인가 모델족 한계인가.

지금까지 확정된 것
------------------
운용점 a=0.79 에서는 구조적 항등식이다:
    P(이관 | sliver) = 1.0000,  Λ_eff = 0,  천장 0.0000,  이득 정확히 0.0000
sliver 가 이관 집합에 통째로 들어가므로 어떤 융합 규칙도 실현 비용을 못 바꾼다.
γ 를 0 까지 내려 게이트를 열어도(0.76) 비용은 10/10 폴드에서 0.1812 그대로였다.

운용점 위는 다르다. 자동화율을 올리면 레버리지가 열리고 천장도 실재하는데,
학습기가 못 가져간다:

    a=0.90   Λ_eff 0.254   천장 0.0070   이득 +0.0000
    a=0.95   Λ_eff 0.924   천장 0.0311   이득 −0.0000
    a=1.00   Λ_eff 1.993   천장 0.0799   이득 +0.0036   (천장의 4.5%)

이건 구조가 아니라 추정의 문제다. 원인이 둘 중 하나인데 논문의 결론이 갈린다.

  (가) 표본 부족.  cal 335장, 적합에 쓰는 건 그 절반 ~167장. a=1.0 에서 손댈
       대상은 sliver 9.4% 뿐이라 학습 신호가 십수 장 수준이다.
       → 이것이 맞으면 v16 논지가 완성된다. "연산자가 아니라 획득이 레버다" 의
         직접 증거가 된다. 기울기에서 '얼마나 더 필요한가' 까지 나온다.

  (나) 모델족 한계.  자료를 아무리 줘도 이 가설공간이 천장에 못 간다.
       → 이것이 맞으면 획득 논지의 근거는 못 되고, LACF 의 설계 한계로만 쓴다.

가르는 법
---------
  A. 오라클    테스트 라벨로 직접 적합한다. 반칙이지만 표본 문제를 제거한다.
               천장의 상당분을 회수하면 (가), 오라클도 못 가져가면 (나).
  B. 표본 스윕 적합 표본을 25%/50%/100% 로 줄여가며 이득 추세를 본다.
               우상향이면 (가) 가 확정되고 기울기가 정량적 근거가 된다.
               평평하면 (나) 쪽이다.

A 와 B 가 엇갈리면 (오라클은 가져가는데 표본 추세는 평평하다 등) 어느 쪽도
단정하지 말고 둘 다 보고할 것. 그것도 정직한 결과다.

    python3 lacf_estimation.py --views <런들...> --out artifacts_v18_real
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


def score_at(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, alpha, au):
    sc = CertifiedScorer(C, alpha, automation=au).fit(op(P_cal[:, conf_i]), y_cal[conf_i])
    return sc, sc.evaluate(op(P_tst), y_tst)


def summarise(d, col, n):
    x = d[col].to_numpy(float); x = x[np.isfinite(x)]
    if x.size < 2:
        return dict(mean=float(x.mean()) if x.size else np.nan, sd=np.nan, mde=np.nan, resolvable=False)
    sd = float(x.std(ddof=1)); m = Z80 * sd / np.sqrt(max(n, 1))
    return dict(mean=float(x.mean()), sd=sd, mde=m, resolvable=bool(abs(x.mean()) > m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--automation", type=float, nargs="+", default=[0.95, 1.00],
                    help="레버리지가 열려 있는 구간만 본다 (0.79 는 Λ_eff=0 이라 볼 것이 없다)")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                    help="적합 표본 비율")
    ap.add_argument("--oracle-hidden", type=int, default=32)
    ap.add_argument("--oracle-maxiter", type=int, default=500)
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
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · K={K} · 기반 {a.base}")
    print(f"시드 {seeds}" + ("   ← 하나뿐: SD 는 폴드 편차" if len(seeds) == 1 else ""))
    print(f"자동화율 {a.automation} · 적합비율 {a.fracs} · "
          f"오라클 H={a.oracle_hidden}/maxiter={a.oracle_maxiter}\n")

    orig = L.features
    rows = []
    try:
        L.features = make_features(OPERATORS[a.base])
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            _, pb_t, margin, eps = L.features(P_tst)
            onS = sliver(margin, eps).astype(bool)
            rng = np.random.default_rng(s * 7919 + f)
            perm = rng.permutation(fit_i)

            # 오라클은 자동화율과 무관하게 한 번만 적합한다 (손실은 이관을 모른다)
            oracle = LACFv2(C, a.alpha, seed=s, maxiter=a.oracle_maxiter,
                            hidden=a.oracle_hidden).fit(P_tst, y_tst)
            subs = {}
            for fr in a.fracs:
                k = max(8, int(round(fr * len(perm))))
                idx = perm[:k]
                subs[fr] = (k, LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter,
                                      hidden=a.hidden).fit(P_cal[:, idx], y_cal[idx]))

            for au in a.automation:
                sc_b, rb = score_at(OPERATORS[a.base], P_cal, y_cal, conf_i,
                                    P_tst, y_tst, C, a.alpha, au)
                acts_b = sc_b.act(pb_t); auto = acts_b < K
                slack = C[y_tst, acts_b] - C[y_tst, :].min(1)
                ceil_S = float(slack[onS & auto].sum() / len(slack)) if (onS & auto).any() else 0.0
                base_c = rb["cost"]

                _, ro = score_at(lambda P, _m=oracle: np.asarray(_m.forward(P)),
                                 P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha, au)
                rec = {"seed": s, "fold": f, "automation": au,
                       "pi_S": float(onS.mean()),
                       "pi_S_eff": float((onS & auto).mean()),
                       "Lambda_eff": dC * float((onS & auto).mean()),
                       "ceil_S_auto": ceil_S, "base_cost": base_c,
                       "oracle_cost": ro["cost"], "oracle_gain": base_c - ro["cost"]}
                for fr, (k, m) in subs.items():
                    _, rr = score_at(lambda P, _m=m: np.asarray(_m.forward(P)),
                                     P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha, au)
                    rec[f"n_fit@{fr}"] = k
                    rec[f"cost@{fr}"] = rr["cost"]
                    rec[f"gain@{fr}"] = base_c - rr["cost"]
                rows.append(rec)
            print(f"  seed {s} fold {f}  π_S={onS.mean():.3f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "lacf_estimation_perfold_v18.csv", index=False)
    n = len(pairs)

    summ = {"base": a.base, "n_folds": n, "seeds": [int(x) for x in seeds],
            "hidden": a.hidden, "maxiter": a.maxiter, "fracs": a.fracs,
            "oracle": {"hidden": a.oracle_hidden, "maxiter": a.oracle_maxiter},
            "levels": {}}

    print("\n" + "=" * 104)
    print(f"추정 실패인가 모델족 한계인가 — 기반 {a.base}, 폴드 {n}개")
    print("=" * 104)
    for au in a.automation:
        d = df[df.automation == au]
        ceil = float(d.ceil_S_auto.mean())
        orc = summarise(d, "oracle_gain", n)
        print(f"\n  [자동화율 {au}]  Λ_eff {d.Lambda_eff.mean():.4f}   천장 {ceil:.4f}")
        print(f"    {'적합표본':<12}{'n':>6}{'비용':>10}{'이득':>10}{'MDE':>9}   천장대비")
        lvl = {"Lambda_eff": float(d.Lambda_eff.mean()), "ceiling": ceil, "fracs": {}}
        for fr in a.fracs:
            g = summarise(d, f"gain@{fr}", n)
            nf = int(d[f"n_fit@{fr}"].mean())
            share = g["mean"] / ceil if ceil > 0 else float("nan")
            print(f"    {f'{100*fr:.0f}%':<12}{nf:>6}{d[f'cost@{fr}'].mean():>10.4f}"
                  f"{g['mean']:>+10.4f}{g['mde']:>9.4f}   {100*share:>6.1f}%"
                  + ("  분해가능" if g["resolvable"] else ""))
            lvl["fracs"][str(fr)] = {**g, "n_fit": nf, "share_of_ceiling": share}
        share_o = orc["mean"] / ceil if ceil > 0 else float("nan")
        print(f"    {'오라클(반칙)':<11}{'test':>6}{d.oracle_cost.mean():>10.4f}"
              f"{orc['mean']:>+10.4f}{orc['mde']:>9.4f}   {100*share_o:>6.1f}%"
              + ("  분해가능" if orc["resolvable"] else ""))
        lvl["oracle"] = {**orc, "share_of_ceiling": share_o}

        # ---- 판정
        gains = [summarise(d, f"gain@{fr}", n)["mean"] for fr in a.fracs]
        rising = len(gains) >= 2 and gains[-1] > gains[0] + 1e-6
        oracle_gets = orc["resolvable"] and orc["mean"] > 0
        if oracle_gets and rising:
            v = "(가) 표본 부족. 오라클이 가져가고 표본을 늘릴수록 이득이 커집니다."
        elif oracle_gets and not rising:
            v = "(가) 쪽이지만 표본 추세가 평평합니다. 오라클만으로 단정하지 말고 둘 다 보고할 것."
        elif not oracle_gets and rising:
            v = "엇갈립니다. 오라클은 못 가져가는데 표본 추세는 우상향입니다. 둘 다 보고할 것."
        else:
            v = "(나) 모델족 한계. 정답 라벨을 줘도, 표본을 늘려도 천장에 못 갑니다."
        print(f"    → {v}")
        lvl["verdict"] = v
        summ["levels"][str(au)] = lvl

    (out / "lacf_estimation_v18.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print("\n" + "=" * 104)
    print("  주의: 오라클은 테스트 라벨로 적합하지만 같은 대리목적(CVaR+벌점)을 최적화하고")
    print("        컨포멀 층은 여전히 cal 에서 보정됩니다. 따라서 이 실험이 막는 것은")
    print("        '표본 부족' 이지 '목적함수가 틀렸다' 가 아닙니다. 원고에 그대로 쓸 것.")
    print("        적합 표본은 폴드 안에서만 줄일 수 있으므로 100% 위쪽은 외삽입니다.")
    print("\n예치: lacf_estimation_v18.json · lacf_estimation_perfold_v18.csv")


if __name__ == "__main__":
    main()
