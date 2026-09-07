#!/usr/bin/env python3
"""theorem9_lacf.py 의 T9.iii 검사를 고친다 — 정리가 아니라 검사가 틀렸다.

무슨 일이 있었나
----------------
product_rule 기반 본 실행에서 T9.iii 가 FAIL 했다 (median ratio 1.148, 10 폴드 중
7개가 1 초과). 그런데 t9iii_diagnose.py 로 통제해 다시 재니 비율이 1 이하였다:

    σ=0.1  E[R_S]/E[R_G] = 0.657      σ=0.3  0.655      σ=1.0  0.565

즉 실패는 정리가 아니라 검사 구현의 결함에서 왔다. 네 가지다.

  (1) 중심 분리 (가장 큼)
      제한 후보는 use_gate=True 로 적합한 model.theta 주위에서, 전역 후보는
      use_gate=False 로 따로 적합한 glob.theta 주위에서 섭동했다. 비율에 '제한의
      효과' 와 '중심이 다른 효과' 가 섞인다. 정리는 같은 가설족을 제한해 쓸 때와
      전역으로 쓸 때의 비교다.
      → 같은 theta, 같은 섭동을 쓰고 sliver 지시함수만 켜고 끈다.

  (2) 결정 규칙 불일치
      검사는 p.argmax(1) 을 썼는데 논문의 결정은 argmin p @ C[:, :K] 다.
      진단 결과 argmax 0.837 vs 비용최소 0.655 로 값이 실제로 다르다.
      → 비용 최소로 바꾼다.

  (3) 증명 전제 위반
      증명 스케치는 contraction lemma 를 쓰는데 argmax 는 Lipschitz 가 아니다.
      (2) 를 고치면 함께 해소된다.

  (4) 표본 부족
      n_mc=64, n_candidates=24. 진단에서 그 설정의 평균 비율 1.004, SD 0.86 이었고
      n_cand=100·n_mc=1024 에서 0.702, SD 0.23 으로 안정됐다.
      → 기본값을 각각 1024 와 100 으로 올린다.

판정 기준도 고친다
------------------
현행 기준은 "ratio ≤ 1 **이고** sqrt(π_S) 를 따른다" 였다. 뒤 절반은 정리가
함의하지 않는다. Theorem 9(iii) 의 우변은

    ΔC·sqrt(π_S)·R_n(H)  +  ΔC·sqrt(log(2/δ)/(2n))

이고, n=250·δ=0.05·ΔC=21.2 에서 **가산항이 1.82** 로 지배적이다. 이 표본 크기에서
sqrt(π_S) 항은 애초에 분해되지 않는다. 따라서 그것을 FAIL 기준으로 쓸 수 없다.
→ 기댓값 수준 비율 E[R_S] ≤ E[R_G] 만 판정에 쓰고, sqrt(π_S) 는 참고로 보고한다.

멱등: 이미 패치된 파일은 건너뛴다. 원본은 .bak_t9 로 남긴다.

    python3 patch_t9.py           패치
    python3 patch_t9.py --check   상태 확인
    python3 patch_t9.py --revert  되돌리기
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

TARGET = "olive_if_v18/theorem9_lacf.py"
MARK = "# --- T9.iii controlled-comparison patch ---"

OLD_RAD = '''def empirical_rademacher(loss_matrix: np.ndarray, n_mc: int = 64, seed: int = 0) -> float:
    """Monte-Carlo empirical Rademacher complexity of a finite loss class given
    as an (n, m) matrix of per-sample losses for m candidate functions."""
    rng = np.random.default_rng(seed); n = loss_matrix.shape[0]
    vals = []
    for _ in range(n_mc):
        sig = rng.choice([-1.0, 1.0], n)
        vals.append(np.max(sig @ loss_matrix) / n)
    return float(np.mean(vals))'''

NEW_RAD = MARK + '''
def empirical_rademacher(loss_matrix: np.ndarray, n_mc: int = 1024, seed: int = 0) -> float:
    """Monte-Carlo empirical Rademacher complexity of a finite loss class given
    as an (n, m) matrix of per-sample losses for m candidate functions.

    n_mc was 64, which the diagnosis showed is far too small: at 24 candidates it
    gave a mean ratio of 1.004 with SD 0.86, against 0.702 with SD 0.23 at
    100 candidates and 1024 draws. Vectorised so the larger budget costs little."""
    rng = np.random.default_rng(seed)
    n = loss_matrix.shape[0]
    sig = rng.choice([-1.0, 1.0], size=(n_mc, n))
    return float(np.mean(np.max(sig @ loss_matrix, axis=1) / n))'''

OLD_LOOP = '''        # T9.iii — candidate functions: random perturbations of the fitted parameters, restricted vs global
        rng = np.random.default_rng(s * 7 + f)
        Ls, Lg = [], []
        for _ in range(n_candidates):
            th = model.theta + rng.normal(0, 0.3, model.theta.shape)
            pr = model.forward(P_tst, th); acts = pr.argmax(1); Ls.append(C[y_tst, acts] - C[y_tst, pb.argmax(1)])
            thg = glob.theta + rng.normal(0, 0.3, glob.theta.shape)
            pg = glob.forward(P_tst, thg); Lg.append(C[y_tst, pg.argmax(1)] - C[y_tst, pb.argmax(1)])
        R_S = empirical_rademacher(np.stack(Ls, 1), seed=s); R_G = empirical_rademacher(np.stack(Lg, 1), seed=s)'''

NEW_LOOP = '''        # T9.iii — controlled comparison. The previous version perturbed two *different*
        # fitted centres (model.theta for the restricted class, glob.theta for the global
        # one), so the ratio mixed the effect of the restriction with the effect of the
        # differing centre; and it scored candidates by p.argmax, not by the paper's
        # cost-minimising action. Both are fixed here: one theta, one set of draws, and
        # the sliver indicator is the only difference between the two classes.
        rng = np.random.default_rng(s * 7 + f)
        _Kc = C.shape[0]
        _act = lambda q: np.argmin(np.asarray(q) @ C[:, :_Kc], 1)
        _base_act = _act(pb)
        Ls, Lg = [], []
        _was = model.use_gate
        for _ in range(n_candidates):
            th = model.theta + rng.normal(0, 0.3, model.theta.shape)
            model.use_gate = True
            pr = np.asarray(model.forward(P_tst, th))
            model.use_gate = False
            pg = np.asarray(model.forward(P_tst, th))
            Ls.append(C[y_tst, _act(pr)] - C[y_tst, _base_act])
            Lg.append(C[y_tst, _act(pg)] - C[y_tst, _base_act])
        model.use_gate = _was
        R_S = empirical_rademacher(np.stack(Ls, 1), seed=s); R_G = empirical_rademacher(np.stack(Lg, 1), seed=s)'''

OLD_CHECK = '''        Check("T9.iii", "restricted Rademacher ≤ global in expectation over folds; ratio ≤ 1 and tracks sqrt(π_S)", len(df),
              int(df.R_restricted.mean() > df.R_global.mean()) + int(df.ratio_R.median() > 1.0),
              f"median ratio {df.ratio_R.median():.3f} vs median sqrt(π_S) {df.sqrt_pi_S.median():.3f}", ""),'''

NEW_CHECK = '''        # The old criterion also required the ratio to "track sqrt(pi_S)". The theorem does
        # not imply that: its right-hand side carries an additive deviation term
        # ΔC·sqrt(log(2/δ)/(2n)) which, at n≈250, δ=0.05, ΔC=21.2, is ≈1.82 and dominates.
        # At this sample size the sqrt(pi_S) factor is not resolvable, so it is reported
        # for reference and only the expectation-level inequality is a pass/fail test.
        Check("T9.iii", "restricted Rademacher ≤ global in expectation over folds (controlled: same theta, cost-minimising action)", len(df),
              int(df.R_restricted.mean() > df.R_global.mean()),
              f"E[R_S]/E[R_G] = {df.R_restricted.mean() / df.R_global.mean():.3f}; "
              f"median per-fold ratio {df.ratio_R.median():.3f}; "
              f"median sqrt(π_S) {df.sqrt_pi_S.median():.3f} (reference only — the additive "
              f"deviation term dominates at this n)", ""),'''

OLD_SIG = "def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, max_folds=None, seeds=SEEDS, maxiter=40, n_candidates=24):"
NEW_SIG = "def run(out_dir, view_dirs=None, smoke=False, alpha=ALPHA, max_folds=None, seeds=SEEDS, maxiter=40, n_candidates=100):"

EDITS = [(OLD_RAD, NEW_RAD), (OLD_SIG, NEW_SIG), (OLD_LOOP, NEW_LOOP), (OLD_CHECK, NEW_CHECK)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="olive_if_v18 의 부모 디렉터리")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    p = root / TARGET
    bak = root / (TARGET + ".bak_t9")

    if not p.exists():
        print(f"★ 없음: {p}"); return 2

    if a.check:
        print(f"  {'패치됨' if MARK in p.read_text(encoding='utf-8') else '원본'}  {TARGET}")
        return 0

    if a.revert:
        if bak.exists():
            shutil.copy2(bak, p); bak.unlink(); print(f"  되돌림 {TARGET}")
        else:
            print("되돌릴 백업이 없습니다")
        return 0

    src = p.read_text(encoding="utf-8")
    if MARK in src:
        print(f"  이미 패치됨, 건너뜀  {TARGET}"); return 0
    if not bak.exists():
        shutil.copy2(p, bak)
    for old, new in EDITS:
        if src.count(old) != 1:
            print(f"★ 앵커가 {src.count(old)}번 발견됨 (1이어야 함)")
            print(f"   찾던 것: {old.splitlines()[0][:80]}...")
            shutil.copy2(bak, p); bak.unlink()
            return 3
        src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")
    print(f"  패치 완료  {TARGET}   (원본 → {TARGET}.bak_t9)")
    print("\n확인:")
    print("  python -m pytest olive_if_v18/tests olive_if_v17/tests -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
