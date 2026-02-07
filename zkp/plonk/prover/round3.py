"""
PLONK Prover Round 3: 몫 다항식 t(x) 커밋먼트
================================================

  ┌─────────────────────────────────────────────────┐
  │  Verifier → Prover: α  (Fiat-Shamir)           │
  │  Prover → Verifier: [t_lo]₁, [t_mid]₁, [t_hi]₁│
  │                                                 │
  │  입력:  α 챌린지, 모든 다항식, 셀렉터            │
  │  출력:  몫 다항식의 3-분할 커밋먼트              │
  └─────────────────────────────────────────────────┘

**이 라운드가 가장 복잡한 이유**:
  모든 제약(게이트 제약 + 순열 제약 + 경계 제약)을 하나의 다항식으로 결합하고,
  Z_H(x)로 나누어 몫 다항식 t(x)를 계산해야 한다.

**세 가지 제약 항**:

  Term 1 — 게이트 제약:
    q_L(x)·a(x) + q_R(x)·b(x) + q_O(x)·c(x)
    + q_M(x)·a(x)·b(x) + q_C(x) + PI(x)

  Term 2 — 순열 제약 (α 배수):
    α · [
      (a(x) + β·x + γ)(b(x) + β·K1·x + γ)(c(x) + β·K2·x + γ) · z(x)
      - (a(x) + β·S_σ1(x) + γ)(b(x) + β·S_σ2(x) + γ)(c(x) + β·S_σ3(x) + γ) · z(ω·x)
    ]

  Term 3 — 경계 제약 (α² 배수):
    α² · (z(x) - 1) · L₁(x)
    (z(ω⁰) = 1 강제)

  결합:
    C(x) = Term1 + Term2 + Term3
    t(x) = C(x) / Z_H(x)

**구현 방식**:
  계수(coefficient) 표현에서 직접 다항식 곱셈과 나눗셈을 수행한다.
  (코셋 FFT 대신 계수 기반 연산 — 블라인딩에 의한 차수 증가 처리가 단순)

**t(x) 3-분할**:
  t(x)의 차수가 ~3n이므로 SRS 범위를 초과할 수 있다.
  t(x) = t_lo(x) + x^n · t_mid(x) + x^{2n} · t_hi(x) 로 분할하여
  각각을 별도로 커밋한다.

사용:
    이 모듈은 직접 호출하지 않고, prover.prove()를 통해 실행된다.
"""

from zkp.plonk.field import FR, get_root_of_unity
from zkp.plonk.polynomial import Polynomial, poly_div, lagrange_basis
from zkp.plonk.kzg import commit
from zkp.plonk.permutation import K1, K2


def execute(state):
    """Round 3을 실행한다.

    Args:
        state: ProverState — Round 1, 2의 결과를 읽고,
               t_lo_poly, t_mid_poly, t_hi_poly와 커밋먼트를 기록한다.
    """
    # ── 1. α 챌린지 생성 ──
    state.alpha = state.transcript.challenge_scalar(b"alpha")

    n = state.n
    omega = state.omega
    domain = state.domain
    alpha = state.alpha
    beta = state.beta
    gamma = state.gamma
    pp = state.preprocessed

    # 다항식 참조 (편의)
    a = state.a_poly
    b = state.b_poly
    c = state.c_poly
    z = state.z_poly
    pi = state.pi_poly

    # 셀렉터 다항식
    q_l = pp.q_l_poly
    q_r = pp.q_r_poly
    q_o = pp.q_o_poly
    q_m = pp.q_m_poly
    q_c = pp.q_c_poly

    # 순열 다항식
    s1 = pp.s_sigma1_poly
    s2 = pp.s_sigma2_poly
    s3 = pp.s_sigma3_poly

    # ── 2. z(ω·x) 다항식 구성 ──
    # z(ω·x)의 계수: cᵢ → ωⁱ · cᵢ
    # 왜? z(ω·x) = Σ cᵢ · (ω·x)ⁱ = Σ (cᵢ · ωⁱ) · xⁱ
    z_omega_coeffs = []
    omega_power = FR(1)
    for coeff in z.coeffs:
        z_omega_coeffs.append(coeff * omega_power)
        omega_power = omega_power * omega
    z_omega = Polynomial(z_omega_coeffs)

    # ── 3. x 다항식 (항등 다항식 id(x) = x) ──
    x_poly = Polynomial([FR(0), FR(1)])  # 0 + 1·x

    # ── 4. L₁(x) 다항식 구성 ──
    # L₁(x) = 첫 번째 Lagrange 기저
    l1 = lagrange_basis(domain, 0)

    # ── 5. 제약 다항식 C(x) 구성 ──

    # Term 1: 게이트 제약
    # q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C + PI
    term1 = q_l * a + q_r * b + q_o * c + q_m * (a * b) + q_c + pi

    # Term 2: 순열 제약
    # 분자: (a + β·x + γ)(b + β·K1·x + γ)(c + β·K2·x + γ) · z
    beta_x = x_poly * beta
    perm_num = (
        (a + beta_x + Polynomial([gamma]))
        * (b + x_poly * (beta * K1) + Polynomial([gamma]))
        * (c + x_poly * (beta * K2) + Polynomial([gamma]))
        * z
    )

    # 분모: (a + β·S_σ1 + γ)(b + β·S_σ2 + γ)(c + β·S_σ3 + γ) · z(ωx)
    perm_den = (
        (a + s1 * beta + Polynomial([gamma]))
        * (b + s2 * beta + Polynomial([gamma]))
        * (c + s3 * beta + Polynomial([gamma]))
        * z_omega
    )

    term2 = (perm_num - perm_den) * alpha

    # Term 3: 경계 제약
    # α² · (z - 1) · L₁
    z_minus_1 = z - Polynomial([FR(1)])
    term3 = z_minus_1 * l1 * (alpha * alpha)

    # 전체 제약 다항식
    constraint = term1 + term2 + term3

    # ── 6. Z_H(x)로 나누기 ──
    # t(x) = C(x) / Z_H(x) — 나머지가 0이어야 함
    zh = Polynomial.vanishing(n)
    t_poly, remainder = poly_div(constraint, zh)

    # 나머지 확인 (디버깅용)
    for coeff in remainder.coeffs:
        if coeff != FR(0):
            raise ValueError(
                "제약 다항식이 Z_H(x)로 나누어 떨어지지 않습니다. "
                "회로 또는 witness에 오류가 있습니다."
            )

    # ── 7. t(x) 3-분할 ──
    # t(x) = t_lo(x) + x^n · t_mid(x) + x^{2n} · t_hi(x)
    t_coeffs = list(t_poly.coeffs)

    # n 단위로 분할 (패딩)
    while len(t_coeffs) < 3 * n:
        t_coeffs.append(FR(0))

    t_lo_coeffs = t_coeffs[:n]
    t_mid_coeffs = t_coeffs[n:2*n]
    t_hi_coeffs = t_coeffs[2*n:3*n]

    # 3n을 초과하는 계수가 있으면 t_hi에 포함
    if len(t_coeffs) > 3 * n:
        t_hi_coeffs = t_coeffs[2*n:]

    state.t_lo_poly = Polynomial(t_lo_coeffs)
    state.t_mid_poly = Polynomial(t_mid_coeffs)
    state.t_hi_poly = Polynomial(t_hi_coeffs)

    # ── 8. KZG 커밋 + 트랜스크립트 업데이트 ──
    state.proof.t_lo_comm = commit(state.t_lo_poly, state.srs)
    state.proof.t_mid_comm = commit(state.t_mid_poly, state.srs)
    state.proof.t_hi_comm = commit(state.t_hi_poly, state.srs)

    state.transcript.append_point(b"t_lo_comm", state.proof.t_lo_comm)
    state.transcript.append_point(b"t_mid_comm", state.proof.t_mid_comm)
    state.transcript.append_point(b"t_hi_comm", state.proof.t_hi_comm)
