#!/usr/bin/env python3
"""라벨 접근 모델의 값이 옵티마이저 시드에 얼마나 흔들리는가 — R1 의 위험 ③.

문제
----
같은 덤프·같은 폴드 시드인데 두 실행이 다른 값을 냈다:

    서버   +0.0423  (천장의 53.0%)
    여기   +0.0325  (천장의 40.6%)

차이는 `--fracs` 목록뿐이었다. 그 목록이 폴드마다 소비하는 난수를 바꾸고,
그 결과 L-BFGS 초기값이 달라진다. 논문 숫자로 쓰기엔 위험하다 -- "왜 53% 가
아니고 40.6% 냐" 는 질문에 답할 수 없다.

이름도 바로잡는다. 이것은 오라클이 아니다:
  * 테스트 라벨로 적합하지만 **같은 대리목적**(CVaR + 벌점)을 최적화하고
  * 컨포멀 층은 여전히 cal 에서 보정된다
따라서 상한이 아니라 **모델족이 도달 가능함을 보이는 하한**이다. 실제 상한은
닫힌 형태의 ceil_S_auto 다. a=0.85 에서 이 값이 음수로 나온 것이 그 증거다 --
진짜 상한이라면 음수가 나올 수 없다.

    python3 oracle_stability.py --views <런들...> --seeds 0 1 2 3 4
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--base", default="product_rule")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--automation", type=float, nargs="+", default=[0.79, 0.95, 1.00])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="옵티마이저 초기화 시드. 폴드 시드와 별개다")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=500)
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    pairs = _pairs(keys, None)
    base_fn = OPERATORS[a.base]
    print(f"뷰 {len(per_view)}개 · 폴드 {len(pairs)}개 · 기반 {a.base}")
    print(f"자동화율 {a.automation} · 옵티마이저 시드 {a.seeds} · H={a.hidden}/{a.maxiter}\n")

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

    rows = []
    try:
        L.features = feats
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            _, conf_i = _split(y_cal, s, f)
            _, pb_t, margin, eps = feats(P_tst)
            onS = sliver(margin, eps).astype(bool)
            models = {sd: LACFv2(C, a.alpha, seed=sd, maxiter=a.maxiter,
                                 hidden=a.hidden).fit(P_tst, y_tst) for sd in a.seeds}
            for au in a.automation:
                sc = CertifiedScorer(C, a.alpha, automation=au).fit(
                    base_fn(P_cal[:, conf_i]), y_cal[conf_i])
                acts = sc.act(pb_t); auto = acts < K
                slack = C[y_tst, acts] - C[y_tst, :].min(1)
                ceil = float(slack[onS & auto].sum() / len(slack)) if (onS & auto).any() else 0.0
                base_c = sc.evaluate(base_fn(P_tst), y_tst)["cost"]
                for sd, m in models.items():
                    c = CertifiedScorer(C, a.alpha, automation=au).fit(
                        np.asarray(m.forward(P_cal[:, conf_i])), y_cal[conf_i]
                    ).evaluate(np.asarray(m.forward(P_tst)), y_tst)["cost"]
                    rows.append({"seed": s, "fold": f, "automation": au, "opt_seed": sd,
                                 "ceiling": ceil, "base_cost": base_c,
                                 "cost": c, "gain": base_c - c})
            print(f"  seed {s} fold {f}", flush=True)
    finally:
        L.features = orig

    df = pd.DataFrame(rows)
    df.to_csv(out / "oracle_stability_perfold_v18.csv", index=False)
    n = len(pairs)

    print("\n" + "=" * 96)
    print("라벨 접근 모델 — 옵티마이저 시드에 따른 흔들림")
    print("=" * 96)
    summ = {"base": a.base, "n_folds": n, "opt_seeds": a.seeds,
            "hidden": a.hidden, "maxiter": a.maxiter, "levels": {}}
    for au in a.automation:
        d = df[df.automation == au]
        ceil = float(d.ceiling.mean())
        per = d.groupby("opt_seed").gain.mean()
        pooled = d.groupby(["seed", "fold"]).gain.mean()          # 시드 평균한 뒤 폴드 편차
        sd_f = float(pooled.std(ddof=1)); mde = Z80 * sd_f / np.sqrt(n)
        print(f"\n  [자동화율 {au}]  천장 {ceil:.4f}")
        print("    시드별 이득: " + "  ".join(f"{k}:{v:+.4f}" for k, v in per.items()))
        print(f"    시드 간 편차 SD {per.std(ddof=1):.4f}   "
              f"범위 {per.min():+.4f}~{per.max():+.4f}")
        print(f"    시드 평균 이득 {pooled.mean():+.4f}  (폴드 SD {sd_f:.4f}, MDE {mde:.4f})"
              + ("  분해가능" if abs(pooled.mean()) > mde else "  MDE 아래"))
        if ceil > 0:
            print(f"    천장 대비 {100*pooled.mean()/ceil:.1f}%")
        summ["levels"][str(au)] = {
            "ceiling": ceil, "per_seed": {str(k): float(v) for k, v in per.items()},
            "across_seed_sd": float(per.std(ddof=1)),
            "mean_gain": float(pooled.mean()), "fold_sd": sd_f, "mde": float(mde),
            "resolvable": bool(abs(pooled.mean()) > mde),
            "share_of_ceiling": float(pooled.mean() / ceil) if ceil > 0 else None}

    print("\n" + "=" * 96)
    print("원고에 쓸 때")
    print("=" * 96)
    print("  · 이것을 '오라클' 이나 '천장' 이라 부르지 말 것. 같은 대리목적을 최적화하고")
    print("    컨포멀 층은 cal 에서 보정되므로 상한이 아니다. 실제 상한은 ceil_S_auto.")
    print("  · '라벨을 준 모델' 로 부르고, 모델족이 도달 가능함을 보이는 하한으로 쓸 것.")
    print("  · 시드 평균 ± 시드 간 SD 를 함께 보고할 것. 단일 시드 값을 쓰면")
    print("    다른 실행에서 재현되지 않는다.")

    (out / "oracle_stability_v18.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print("\n예치: oracle_stability_v18.json · oracle_stability_perfold_v18.csv")


if __name__ == "__main__":
    main()
