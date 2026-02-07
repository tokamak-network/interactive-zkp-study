"""
PLONK Prover Round 4: 다항식 평가값 산출
==========================================

  ┌─────────────────────────────────────────────────┐
  │  Verifier → Prover: ζ  (Fiat-Shamir)           │
  │  Prover → Verifier: ā, b̄, c̄, s̄_σ1, s̄_σ2, z̄_ω │
  │                                                 │
  │  입력:  ζ 챌린지, 각 다항식                      │
  │  출력:  6개의 스칼라 평가값                      │
  └─────────────────────────────────────────────────┘

**이 라운드의 목적**:
  Verifier가 선택한 랜덤 점 ζ에서 각 다항식의 값을 "열어서" 보여준다.
  Schwartz-Zippel 보조정리에 의해, 랜덤 점에서 값이 일치하면
  다항식 자체가 같을 확률이 매우 높다.

**평가하는 다항식들**:
  1. ā = a(ζ)      — 왼쪽 배선 다항식
  2. b̄ = b(ζ)      — 오른쪽 배선 다항식
  3. c̄ = c(ζ)      — 출력 배선 다항식
  4. s̄_σ1 = S_σ1(ζ) — 첫 번째 순열 다항식
  5. s̄_σ2 = S_σ2(ζ) — 두 번째 순열 다항식
  6. z̄_ω = z(ζ·ω)   — 순열 누적자의 "다음 점" 평가

**왜 z(ζ·ω)가 필요한가?**
  순열 제약에서 z(ω·x) 항이 있다.
  ζ에서의 제약 확인 시 z(ζ)뿐 아니라 z(ζ·ω)도 필요하다.
  z(ζ)는 Round 5의 선형화에서 커밋먼트로 처리되지만,
  z(ζ·ω)는 명시적 스칼라 값으로 제공된다.

사용:
    이 모듈은 직접 호출하지 않고, prover.prove()를 통해 실행된다.
"""

from zkp.plonk.field import FR


def execute(state):
    """Round 4를 실행한다.

    Args:
        state: ProverState — Round 1~3의 다항식을 읽고,
               6개의 평가값과 ζ 챌린지를 기록한다.
    """
    # ── 1. ζ 챌린지 생성 ──
    state.zeta = state.transcript.challenge_scalar(b"zeta")
    zeta = state.zeta
    omega = state.omega
    pp = state.preprocessed

    # ── 2. 각 다항식을 ζ에서 평가 (Horner's method) ──

    # 배선 다항식 평가
    a_eval = state.a_poly.evaluate(zeta)
    b_eval = state.b_poly.evaluate(zeta)
    c_eval = state.c_poly.evaluate(zeta)

    # 순열 다항식 평가
    s_sigma1_eval = pp.s_sigma1_poly.evaluate(zeta)
    s_sigma2_eval = pp.s_sigma2_poly.evaluate(zeta)

    # z(ζ·ω): 순열 누적자를 "다음 도메인 점"에서 평가
    # ζ·ω는 ζ를 한 도메인 스텝만큼 이동한 점
    z_omega_eval = state.z_poly.evaluate(zeta * omega)

    # ── 3. 증명에 기록 ──
    state.proof.a_eval = a_eval
    state.proof.b_eval = b_eval
    state.proof.c_eval = c_eval
    state.proof.s_sigma1_eval = s_sigma1_eval
    state.proof.s_sigma2_eval = s_sigma2_eval
    state.proof.z_omega_eval = z_omega_eval

    # ── 4. 트랜스크립트에 평가값 추가 ──
    state.transcript.append_scalar(b"a_eval", a_eval)
    state.transcript.append_scalar(b"b_eval", b_eval)
    state.transcript.append_scalar(b"c_eval", c_eval)
    state.transcript.append_scalar(b"s_sigma1_eval", s_sigma1_eval)
    state.transcript.append_scalar(b"s_sigma2_eval", s_sigma2_eval)
    state.transcript.append_scalar(b"z_omega_eval", z_omega_eval)
