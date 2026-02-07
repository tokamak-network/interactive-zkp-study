"""
PLONK Prover Round 5: 선형화 + KZG 열기 증명
===============================================

  ┌─────────────────────────────────────────────────┐
  │  Verifier → Prover: v  (Fiat-Shamir)           │
  │  Prover → Verifier: [W_ζ]₁, [W_ζω]₁           │
  │                                                 │
  │  입력:  v 챌린지, Round 4 평가값                 │
  │  출력:  2개의 KZG 열기 증명                      │
  └─────────────────────────────────────────────────┘

**선형화 트릭(Linearization Trick)**:
  Verifier가 제약 방정식을 확인하려면 다항식의 "곱"의 값이 필요하지만,
  커밋먼트 형태에서 곱을 계산할 수 없다.

  핵심: Round 4에서 대부분의 다항식을 ζ에서 평가했으므로,
  다항식의 곱을 "스칼라 × 다항식" 형태로 바꿔서 커밋먼트 계산 가능하게 한다.

  r(x)는 ζ에서 평가했을 때 C(ζ) = t(ζ)·Z_H(ζ)와 같은 값을 내야 한다.

**구성**:
  r(x) = 게이트항 + 순열항 + 경계항 + 상수항

  게이트: q_M(x)·ā·b̄ + q_L(x)·ā + q_R(x)·b̄ + q_O(x)·c̄ + q_C(x) + PI(ζ)
  순열:   α·(ā+βζ+γ)(b̄+βK1ζ+γ)(c̄+βK2ζ+γ)·z(x)
        - α·(ā+βs̄1+γ)(b̄+βs̄2+γ)·β·z̄ω·S_σ3(x)
        - α·(ā+βs̄1+γ)(b̄+βs̄2+γ)·(c̄+γ)·z̄ω  ← 상수항
  경계:   α²·L₁(ζ)·z(x) - α²·L₁(ζ)  ← 상수항

사용:
    이 모듈은 직접 호출하지 않고, prover.prove()를 통해 실행된다.
"""

from zkp.plonk.field import FR
from zkp.plonk.polynomial import Polynomial, poly_div
from zkp.plonk.kzg import commit
from zkp.plonk.permutation import K1, K2
from zkp.plonk.utils import vanishing_poly_eval, lagrange_basis_eval


def execute(state):
    """Round 5를 실행한다.

    Args:
        state: ProverState — Round 1~4의 모든 결과를 읽고,
               W_zeta_comm, W_zeta_omega_comm을 기록한다.
    """
    # ── 1. v 챌린지 생성 ──
    state.v = state.transcript.challenge_scalar(b"v")
    v = state.v

    n = state.n
    zeta = state.zeta
    omega = state.omega
    alpha = state.alpha
    beta = state.beta
    gamma = state.gamma
    pp = state.preprocessed

    # Round 4 평가값 (스칼라)
    a_eval = state.proof.a_eval
    b_eval = state.proof.b_eval
    c_eval = state.proof.c_eval
    s_sigma1_eval = state.proof.s_sigma1_eval
    s_sigma2_eval = state.proof.s_sigma2_eval
    z_omega_eval = state.proof.z_omega_eval

    # ── 2. 공개 값 계산 ──
    pi_zeta = state.pi_poly.evaluate(zeta)
    l1_zeta = lagrange_basis_eval(0, n, omega, zeta)

    # ── 3. 선형화 다항식 r(x) 구성 ──
    # r(x)는 ζ에서 평가하면 C(ζ) = t(ζ)·Z_H(ζ)와 같은 값을 가진다.

    # Part 1: 게이트 제약 (선형화)
    # q_M(x)·ā·b̄ + q_L(x)·ā + q_R(x)·b̄ + q_O(x)·c̄ + q_C(x) + PI(ζ)
    r_poly = (
        pp.q_m_poly * (a_eval * b_eval)
        + pp.q_l_poly * a_eval
        + pp.q_r_poly * b_eval
        + pp.q_o_poly * c_eval
        + pp.q_c_poly
        + Polynomial([pi_zeta])
    )

    # Part 2: 순열 제약 (× α)
    # 원래 term2 = α · [num_poly · z(x) - den_poly · z(ωx)]
    # 선형화: z(x)와 S_σ3(x)만 다항식으로 남기고, 나머지를 ζ에서 평가
    #
    # num 부분: (ā + β·ζ + γ)(b̄ + β·K1·ζ + γ)(c̄ + β·K2·ζ + γ) · z(x)
    perm_z_scalar = (
        alpha
        * (a_eval + beta * zeta + gamma)
        * (b_eval + beta * K1 * zeta + gamma)
        * (c_eval + beta * K2 * zeta + gamma)
    )

    # den 부분의 선형화:
    # 원래: -(ā+β·S1(ζ)+γ)(b̄+β·S2(ζ)+γ)(c̄+β·S3(x)+γ)·z̄ω
    # = -(ā+β·s̄1+γ)(b̄+β·s̄2+γ)·z̄ω · (c̄ + β·S_σ3(x) + γ)
    # = -(ā+β·s̄1+γ)(b̄+β·s̄2+γ)·z̄ω · [β·S_σ3(x) + (c̄+γ)]
    # = -(ā+β·s̄1+γ)(b̄+β·s̄2+γ)·z̄ω·β · S_σ3(x)    ← 다항식 항
    #   -(ā+β·s̄1+γ)(b̄+β·s̄2+γ)·z̄ω·(c̄+γ)          ← 상수 항

    ab_factor = (
        (a_eval + beta * s_sigma1_eval + gamma)
        * (b_eval + beta * s_sigma2_eval + gamma)
    )

    # S_σ3(x) 다항식 항
    perm_s3_scalar = alpha * ab_factor * beta * z_omega_eval

    # 상수 항 (ζ에서의 값)
    perm_const = FR(0) - alpha * ab_factor * z_omega_eval * (c_eval + gamma)

    r_poly = r_poly + state.z_poly * perm_z_scalar
    r_poly = r_poly - pp.s_sigma3_poly * perm_s3_scalar
    r_poly = r_poly + Polynomial([perm_const])

    # Part 3: 경계 제약 (× α²)
    # 원래: α² · (z(x) - 1) · L₁(ζ) = α²·L₁(ζ)·z(x) - α²·L₁(ζ)
    # 다항식 항: α²·L₁(ζ)·z(x)
    # 상수 항: -α²·L₁(ζ)
    r_poly = r_poly + state.z_poly * (alpha * alpha * l1_zeta)
    r_poly = r_poly + Polynomial([FR(0) - alpha * alpha * l1_zeta])

    # ── 4. r(ζ) 평가 → Proof에 포함 ──
    # Verifier는 r(ζ)를 직접 계산할 수 없다 (q_*(ζ) 값을 모르므로).
    # 대신 Prover가 r_eval을 제공하고, W_ζ 열기 증명으로 정확성을 보장한다.
    r_eval = r_poly.evaluate(zeta)
    state.proof.r_eval = r_eval

    # ── 5. t(ζ) 계산 ──
    zeta_n = zeta ** n
    zeta_2n = zeta_n * zeta_n
    t_eval = (
        state.t_lo_poly.evaluate(zeta)
        + zeta_n * state.t_mid_poly.evaluate(zeta)
        + zeta_2n * state.t_hi_poly.evaluate(zeta)
    )

    # ── 6. 일괄 열기 증명: W_ζ(x) ──
    t_combined = (
        state.t_lo_poly
        + state.t_mid_poly * zeta_n
        + state.t_hi_poly * zeta_2n
    )

    numerator_zeta = t_combined - Polynomial([t_eval])
    numerator_zeta = numerator_zeta + (r_poly - Polynomial([r_eval])) * v

    v_power = v * v
    numerator_zeta = numerator_zeta + (state.a_poly - Polynomial([a_eval])) * v_power
    v_power = v_power * v
    numerator_zeta = numerator_zeta + (state.b_poly - Polynomial([b_eval])) * v_power
    v_power = v_power * v
    numerator_zeta = numerator_zeta + (state.c_poly - Polynomial([c_eval])) * v_power
    v_power = v_power * v
    numerator_zeta = numerator_zeta + (pp.s_sigma1_poly - Polynomial([s_sigma1_eval])) * v_power
    v_power = v_power * v
    numerator_zeta = numerator_zeta + (pp.s_sigma2_poly - Polynomial([s_sigma2_eval])) * v_power

    # (x - ζ)로 나누기
    divisor_zeta = Polynomial([FR(0) - zeta, FR(1)])
    W_zeta_poly, remainder = poly_div(numerator_zeta, divisor_zeta)

    # ── 7. z(x) 열기 증명: W_ζω(x) ──
    numerator_zeta_omega = state.z_poly - Polynomial([z_omega_eval])
    divisor_zeta_omega = Polynomial([FR(0) - zeta * omega, FR(1)])
    W_zeta_omega_poly, remainder2 = poly_div(numerator_zeta_omega, divisor_zeta_omega)

    # ── 8. KZG 커밋 ──
    state.proof.W_zeta_comm = commit(W_zeta_poly, state.srs)
    state.proof.W_zeta_omega_comm = commit(W_zeta_omega_poly, state.srs)
