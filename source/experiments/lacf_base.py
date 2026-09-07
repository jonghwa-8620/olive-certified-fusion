#!/usr/bin/env python3
"""LACF 의 기반 연산자를 바꿔가며 재는 실험 — 취약점 J.

왜 필요한가
-----------
lacf_v2.features() 는 기반을 sum_rule 로 하드코딩한다:

    pb = sum_rule(P)

그리고 모델은  p = p_base + 1[x∈S]·g(x)·(p_res − p_base)  이므로 sliver 밖에서
LACF 는 *정확히* sum_rule 이다. 그런데 본 실행에서 sum_rule 은 13개 중 9위(0.194)
이고 product_rule 이 1위(0.181)다. 즉 LACF 의 가설공간은 최고 연산자를 표현조차
할 수 없었다. "10위"의 대부분은 새 층이 만든 손해가 아니라 기반에서 물려받은 것이다.

심사자가 이걸 먼저 발견하면 실험 전체가 흔들린다. 우리가 먼저 재면 결론이 바뀐다:
  "우리 모델이 제일 나쁘다"  →  "제한된 국소 층은 어떤 기반 위에서도 중립이다"
후자가 논문 논지와 일치하고 방어 가능하다.

무엇을 재는가
-------------
기반 b 마다, 폴드마다:
  * cost(b)                기반 연산자 자체
  * cost(LACF on b)        그 기반 위의 LACF (H 스윕)
  * π_S(b), Λ(b)           기반이 바뀌면 sliver 도 바뀐다 (margin 이 기반 기준이므로)
그리고 폴드 페어드 차이 + §10 관례의 MDE 로:
  * LACF − 기반            국소 층이 더하는가 빼는가
  * LACF − product_rule    고정 최고 규칙 대비
  * LACF − 폴드별오라클    달성 불가 기준선 (참고용, 논문에선 분리 표기)

기준선을 두 종류로 나눠 쓰는 게 핵심이다. lacf_ceiling.py 의 best_global 은
폴드마다 최고 연산자를 골라낸 값이라 어떤 고정 규칙으로도 달성할 수 없다.

    python3 lacf_base.py --views <런들...> --out artifacts_v18_real
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
from olive_if_v18.lacf_v2 import LACFv2, OURS, _fold_data, _pairs, _score, _split

Z80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))          # 2.802


def mde(sd_paired: float, n: int) -> float:
    """§10 과 같은 관례: 2.802 · SD(폴드별 페어드 차이) / √n."""
    return float(Z80 * sd_paired / np.sqrt(max(n, 1)))


def make_features(base_fn, _COST_OVERRIDE=None):
    """lacf_v2.features 와 같되 기반 연산자만 갈아끼운다.

    sliver 는 margin(p_base) 로 정의되므로 기반이 바뀌면 sliver 도 함께 바뀐다.
    그게 맞다 -- Theorem 4 의 sliver 는 기준 규칙에 상대적인 개념이다.
    """
    def features(P):
        J, n, K = P.shape
        pb = base_fn(P)
        mx = P.max(-1).T
        js = np.mean([0.5 * (_H(0.5 * (P[i] + P[j])) - 0.5 * _H(P[i]) - 0.5 * _H(P[j]))
                      for i in range(J) for j in range(i + 1, J)], 0) if J > 1 else np.zeros(n)
        from olive_if_v18.lacf_v2 import _margin_eps   # 단위 전역(LACF_UNITS)을 따른다
        margin, eps = _margin_eps(P, pb, _COST_OVERRIDE)
        phi = np.c_[mx, js, margin, _H(P).mean(0), eps]
        return phi, pb, margin, eps
    return features


def paired(df, a, b):
    """폴드 페어드 차이 a−b 를 §10 관례로 요약."""
    d = (df[a] - df[b]).to_numpy(float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return dict(n=int(d.size), mean=float("nan"), sd=float("nan"),
                    mde=float("nan"), resolvable=False)
    sd = float(d.std(ddof=1))
    m = mde(sd, d.size)
    return dict(n=int(d.size), mean=float(d.mean()), sd=sd, mde=m,
                resolvable=bool(abs(d.mean()) > m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts_v18_real")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--bases", nargs="+",
                    default=["sum_rule", "product_rule", "entropy_weighted"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--maxiter", type=int, default=150,
                    help="150 에서 이미 수렴 (40→150 은 움직이고 150→500 은 1e-5 이하)")
    a = ap.parse_args()

    for b in a.bases:
        if b not in OPERATORS:
            sys.exit(f"알 수 없는 기반 '{b}'. 가능: {sorted(OPERATORS)}")

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_view, keys = load_views(a.views)
    K = next(iter(per_view[0].values()))[0].shape[1]
    C = default_cost_matrix(K)
    pairs = _pairs(keys, None)
    seeds = sorted({s for s, _ in pairs})
    print(f"뷰 {len(per_view)}개 · (시드,폴드) 쌍 {len(pairs)}개 · K={K} · α={a.alpha}")
    print(f"시드 {seeds}" + ("   ← 하나뿐: SD 는 폴드 편차입니다" if len(seeds) == 1 else ""))
    print(f"기반 {a.bases} · hidden {a.hidden} · maxiter {a.maxiter}\n")

    orig_features = L.features
    rows = []
    try:
        for s, f in pairs:
            P_cal, y_cal, P_tst, y_tst = _fold_data(per_view, s, f)
            fit_i, conf_i = _split(y_cal, s, f)
            rec = {"seed": s, "fold": f}

            # ---- 고정 연산자 전부 (기반 후보 포함) — 기반 교체와 무관하므로 한 번만
            op_cost = {}
            for n, op in OPERATORS.items():
                op_cost[n] = _score(op, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)["cost"]
                rec[f"op::{n}"] = op_cost[n]
            # 폴드별 오라클 선택 — 어떤 고정 규칙으로도 달성 불가. 참고용으로만.
            rec["oracle_pick"] = min(op_cost.values())
            rec["oracle_pick_name"] = min(op_cost, key=op_cost.get)

            # ---- 기반을 갈아끼우며 LACF
            for bname in a.bases:
                L.features = make_features(OPERATORS[bname])
                _, _, margin, eps = L.features(P_tst)
                pi_S = float(L.sliver(margin, eps).mean())
                rec[f"piS::{bname}"] = pi_S
                rec[f"Lambda::{bname}"] = float(C[1:, 0].max() * pi_S)
                best = np.inf
                for H in a.hidden:
                    m = LACFv2(C, a.alpha, seed=s, maxiter=a.maxiter,
                               hidden=H).fit(P_cal[:, fit_i], y_cal[fit_i])
                    c = _score(m, P_cal, y_cal, conf_i, P_tst, y_tst, C, a.alpha)["cost"]
                    rec[f"lacf::{bname}::H{H}"] = c
                    best = min(best, c)
                rec[f"lacf::{bname}"] = best          # 그 기반에서의 최선 설정
            rows.append(rec)
            print("  seed %d fold %d  " % (s, f)
                  + "  ".join("%s π_S=%.3f LACF=%.4f (기반 %.4f)"
                              % (b, rec[f"piS::{b}"], rec[f"lacf::{b}"], rec[f"op::{b}"])
                              for b in a.bases), flush=True)
    finally:
        L.features = orig_features                    # 다른 모듈에 새지 않게 복원

    df = pd.DataFrame(rows)
    df.to_csv(out / "lacf_base_perfold_v18.csv", index=False)

    # -------------------------------------------------------------------- 요약
    summ = {"n_folds": int(len(df)), "seeds": [int(x) for x in seeds],
            "hidden": a.hidden, "maxiter": a.maxiter, "bases": a.bases,
            "single_seed": bool(len(seeds) == 1), "comparisons": {}}
    for b in a.bases:
        summ["comparisons"][b] = {
            "pi_S_mean": float(df[f"piS::{b}"].mean()),
            "Lambda_mean": float(df[f"Lambda::{b}"].mean()),
            "base_cost_mean": float(df[f"op::{b}"].mean()),
            "lacf_cost_mean": float(df[f"lacf::{b}"].mean()),
            "vs_own_base": paired(df, f"lacf::{b}", f"op::{b}"),
            "vs_product_rule": paired(df, f"lacf::{b}", "op::product_rule"),
            "vs_oracle_pick": paired(df, f"lacf::{b}", "oracle_pick"),
        }
    summ["product_rule_mean"] = float(df["op::product_rule"].mean())
    summ["oracle_pick_mean"] = float(df["oracle_pick"].mean())
    summ["oracle_pick_note"] = ("폴드마다 최고 연산자를 골라낸 값. 어떤 고정 규칙으로도 "
                                "달성할 수 없으므로 고정 규칙 기준선과 분리해 보고할 것.")
    (out / "lacf_base_v18.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))

    def line(tag, d):
        mark = "분해가능" if d["resolvable"] else "MDE 아래(동률)"
        return ("    %-22s %+.4f   SD %.4f   MDE %.4f   %s"
                % (tag, d["mean"], d["sd"], d["mde"], mark))

    print("\n" + "=" * 78)
    print("J. 기반 연산자를 바꾸면 결론이 바뀌는가")
    print("=" * 78)
    print("  고정 최고 규칙 product_rule  %.4f" % summ["product_rule_mean"])
    print("  폴드별 오라클 선택           %.4f   ← 고정 규칙으로 달성 불가. 별도 표기."
          % summ["oracle_pick_mean"])
    for b in a.bases:
        c = summ["comparisons"][b]
        print("\n  [기반 %s]  π_S=%.3f  Λ=%.3f" % (b, c["pi_S_mean"], c["Lambda_mean"]))
        print("    기반 자체            %.4f" % c["base_cost_mean"])
        print("    그 위의 LACF (최선)  %.4f" % c["lacf_cost_mean"])
        print(line("LACF − 자기 기반", c["vs_own_base"]))
        print(line("LACF − product_rule", c["vs_product_rule"]))
        print(line("LACF − 오라클선택", c["vs_oracle_pick"]))

    print("\n" + "=" * 78)
    print("읽는 법")
    print("=" * 78)
    print("  · 'LACF − 자기 기반' 이 어느 기반에서도 MDE 아래면:")
    print("      국소 층은 기반과 무관하게 중립이다 — 논문 논지 그대로, 방어 가능.")
    print("  · product_rule 기반에서 'LACF − product_rule' 이 MDE 아래면:")
    print("      '최고 규칙 위에 얹어도 더 못 얻는다' 가 되어 J 반론이 완전히 막힌다.")
    print("  · 어느 기반에서든 음수가 MDE 를 넘으면 그건 실제 개선이다 — 그때는")
    print("      게이트를 다시 돌려야 한다. 코드가 대신 판정해 주지 않는다.")
    print("\n예치: lacf_base_v18.json · lacf_base_perfold_v18.csv")


if __name__ == "__main__":
    main()
