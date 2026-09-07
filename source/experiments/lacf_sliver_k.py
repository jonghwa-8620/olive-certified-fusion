#!/usr/bin/env python3
"""sliver 폭을 넓혀 레버리지를 연 뒤에도 LACF 가 못 가져가는가 — §2.1 격하의 보완.

왜 필요한가
-----------
sliver_arithmetic.py 가 §2.1 을 산술로 판정했다:

    Spearman ρ(margin, VOI) = −0.982      두 규칙이 사실상 같은 순서
    관측 P(이관|S) 가 min(1, (1−a)/π_S) 예측선을 평균절대차 0.019 로 따라감

즉 논문의 k=2 에서 S ⊆ 이관집합 인 것은 π_S(0.094) < 이관율(0.210) 의 귀결이지
코퍼스의 성질이 아니다. §2.1 은 발견이 아니라 명제로 써야 하고, 그러면
"연산자는 레버가 아니다" 의 근거가 산술 하나에 얹히게 된다. 그건 약하다.

그런데 같은 스윕이 실험 하나를 만들어 준다. k 를 키워 π_S 가 이관율을 넘으면
포함이 깨지고 레버리지가 실제로 열린다:

    k=6   π_S 0.256   P(이관|S) 0.771   Λ_eff 1.281   천장 0.0173
    k=8   π_S 0.338   P(이관|S) 0.610   Λ_eff 2.832   천장 0.0318
    k=12  π_S 0.453   P(이관|S) 0.461   Λ_eff 5.207   천장 0.0442
    k=20  π_S 0.575   P(이관|S) 0.363   Λ_eff 7.793   천장 0.0506

**열린 구간에서 LACF 가 이득을 내는가.** 이건 산술이 아니다. 그리고 자동화율을
건드리지 않으므로(논문이 권하는 운용 그대로) R1 의 a=1.00 보다 방어하기 좋다.

  이득이 나오면   연산자가 레버가 되는 조건을 sliver 폭으로 특정한 것이다.
                  단, 그러면 논문의 k=2 가 왜 그 값인지 정당화해야 하고
                  그 k 에서 절제·정리검증을 다시 해야 한다. 공짜가 아니다.
  안 나오면       레버리지가 열려도 못 가져간다가 두 번째 독립 경로로 확인되어,
                  "연산자는 레버가 아니다" 가 산술에 기대지 않게 된다.

주의: k 가 커지면 sliver-제한이 느슨해져 Theorem 9(iii) 의 sqrt(π_S) 이득도
      사라진다. 즉 k 를 키운 LACF 는 점점 전역 학습기에 가까워진다. 이득이
      나오더라도 그것이 '국소성' 덕분인지 '그냥 학습' 덕분인지 구분해야 하므로
      같은 k 에서 no-gate(전역) 판을 함께 잰다.

    python3 lacf_sliver_k.py --views <런들...> --out out
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
from olive_if_v17.sota_dl_comparison import (ALPHA, OPERATORS, CertifiedScorer,
                                             default_cost_matrix, load_views)
from olive_if_v18.lacf_v2 import LACFv2, _fold_data, _pairs, _score, _split, sliver

Z80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))


def summarise(x, n):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2:
        return dict(mean=float(x.mean()) if x.size else np.nan, sd=np.nan,
                    mde=np.nan, resolvable=False)
    sd = float(x.std(ddof=1)); m = Z80 * sd / np.sqrt(max(n, 1))
    return dict(mean=float(x.mean()), sd=sd, mde=m, resolvable=bool(abs(x.mean()) > m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--ks", type=float, nargs="+", default=[2, 3, 4, 6, 8, 12, 20])
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    dC = float(C[1:, 0].max())
    pairs = _pairs(keys, None)
    seeds = sorted({s for s, _ in pairs})
    base_fn = OPERATORS[a.base]
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · 기반 {a.base} · k {a.ks}")
    print(f"시드 {seeds}" + ("   ← 하나뿐: SD 는 폴드 편차" if len(seeds) == 1 else "") + "\n")

    orig = L.features
    rows = []
    try:

        # features 의 기반만 갈아끼운다 (lacf_base.py 와 같은 방식)
        from olive_if_v17.sota_dl_comparison import _H

        def feats(P):
            J, n, Kc = P.shape
            pb = base_fn(P)
            mx = P.max(-1).T
            js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                          for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
            from olive_if_v18.lacf_v2 import _margin_eps  # 단위 전역(LACF_UNITS)을 따른다
            margin, eps = _margin_eps(P, pb)
            return np.c_[mx, js, margin, _H(P).mean(0), eps], pb, margin, eps
        L.features = feats

        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            _, pb_t, margin, eps = feats(P_tst)
            sc = CertifiedScorer(C, a.alpha).fit(base_fn(P_cal[:, conf_i]), y_cal[conf_i])
            acts = sc.act(pb_t); auto = acts < K
            slack = C[y_tst, acts] - C[y_tst, :].min(1)
            base_c = sc.evaluate(base_fn(P_tst), y_tst)["cost"]
            for k in a.ks:
                onS = sliver(margin, eps, k).astype(bool)
                piS = float(onS.mean()); pi_eff = float((onS & auto).mean())
                ceil = float(slack[onS & auto].sum() / len(slack)) if (onS & auto).any() else 0.0
                rec = {"seed": s, "fold": f, "k": k, "pi_S": piS, "pi_S_eff": pi_eff,
                       "P_refer_given_S": float((~auto)[onS].mean()) if onS.any() else np.nan,
                       "Lambda_eff": dC * pi_eff, "ceil_S_auto": ceil, "base_cost": base_c}
                for tag, gate in (("lacf", True), ("noGate", False)):
                    m = LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter, hidden=a.hidden,
                               sliver_k=k, use_gate=gate).fit(P_cal[:, fit_i], y_cal[fit_i])
                    r = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)
                    rec[f"{tag}_cost"] = r["cost"]
                    rec[f"{tag}_gain"] = base_c - r["cost"]
                    rec[f"{tag}_violation"] = r.get("violation", np.nan)
                rows.append(rec)
            print(f"  seed {s} fold {f}  기반 {base_c:.4f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "lacf_sliver_k_perfold_v18.csv", index=False)
    n = len(pairs)

    print("\n" + "=" * 104)
    print(f"sliver 폭 스윕 — 기반 {a.base}, 폴드 {n}개, 이관율 0.210")
    print("=" * 104)
    print(f"  {'k':>4} {'π_S':>7} {'P(이관|S)':>10} {'Λ_eff':>8} {'천장':>8} "
          f"{'LACF이득':>10} {'MDE':>8} {'천장대비':>9}   {'no-gate이득':>11}")
    summ = {"base": a.base, "n_folds": n, "seeds": [int(x) for x in seeds],
            "hidden": a.hidden, "maxiter": a.maxiter, "rows": []}
    hits = []
    for k in a.ks:
        d = df[df.k == k]
        ceil = float(d.ceil_S_auto.mean())
        g = summarise(d.lacf_gain, n); gn = summarise(d.noGate_gain, n)
        share = g["mean"] / ceil if ceil > 0 else float("nan")
        print(f"  {k:>4g} {d.pi_S.mean():>7.3f} {d.P_refer_given_S.mean():>10.3f} "
              f"{d.Lambda_eff.mean():>8.3f} {ceil:>8.4f} "
              f"{g['mean']:>+10.4f} {g['mde']:>8.4f} {100*share:>8.1f}%   {gn['mean']:>+11.4f}"
              + ("  ★분해가능" if g["resolvable"] and g["mean"] > 0 else ""))
        summ["rows"].append({"k": k, "pi_S": float(d.pi_S.mean()), "ceiling": ceil,
                             "Lambda_eff": float(d.Lambda_eff.mean()),
                             "lacf": g, "no_gate": gn, "share_of_ceiling": share})
        if g["resolvable"] and g["mean"] > 0:
            hits.append(k)

    print("\n" + "=" * 104)
    print("판정")
    print("=" * 104)
    if not hits:
        print("  → 레버리지를 열어도 (Λ_eff 0 → %.2f, 천장 0 → %.4f) 어떤 k 에서도"
              % (df.Lambda_eff.max(), df.ceil_S_auto.max()))
        print("    이득이 해상도를 넘지 않습니다. '연산자는 레버가 아니다' 가 §2.1 의")
        print("    산술에 기대지 않는 두 번째 근거를 얻습니다. §2.1 을 명제로 격하해도")
        print("    논지가 흔들리지 않습니다.")
    else:
        print(f"  → k = {hits} 에서 이득이 분해 가능합니다. 연산자가 레버가 되는 조건을")
        print("    sliver 폭으로 특정한 것입니다. 다만 다음을 반드시 함께 해야 합니다:")
        print("      · 논문의 k=2 가 왜 그 값인지 정당화 (Theorem 4 의 2ε 유도)")
        print("      · 그 k 에서 절제·Theorem 9 검증 재실행")
        print("      · no-gate 열과 비교 — 이득이 '국소성' 덕인지 '그냥 학습' 덕인지")
        print("        no-gate 가 비슷하면 국소 제한이 기여한 바가 없다는 뜻입니다")
    print("\n  주의: k 가 커지면 sliver-제한이 느슨해져 Theorem 9(iii) 의 sqrt(π_S) 이점도")
    print("        사라집니다. 큰 k 의 LACF 는 점점 전역 학습기입니다.")

    (out / "lacf_sliver_k_v18.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print("\n예치: lacf_sliver_k_v18.json · lacf_sliver_k_perfold_v18.csv")


if __name__ == "__main__":
    main()
