"""
PLONK Verifier
================

PLONK 증명을 검증한다.

**검증 과정**:
  1. Fiat-Shamir 트랜스크립트 재생 → β, γ, α, ζ, v, u 챌린지 복원
  2. Z_H(ζ), L₁(ζ), PI(ζ) 계산
  3. 선형화 커밋먼트 [D]₁ 구성
  4. r₀ 계산 (선형화 다항식의 상수 기여분)
  5. 결합 커밋먼트 [F]₁, [E]₁ 구성
  6. 페어링 검사

**핵심 방정식**:
  Prover의 W_ζ(x) 분자:
    [t_comb(x) - t_eval]
    + v·[r(x) - r_eval]
    + v²·[a(x) - ā] + v³·[b(x) - b̄] + v⁴·[c(x) - c̄]
    + v⁵·[S_σ1(x) - s̄_σ1] + v⁶·[S_σ2(x) - s̄_σ2]

  commit(r(x)) = [D]₁ + r₀·G₁  (D는 커밋먼트 부분, r₀는 상수)
  r_eval = D(ζ) + r₀

  [F]₁ = [t_comb]₁ + v·([D]₁ + r₀·G₁) + v²·[a]₁ + ... + v⁶·[S_σ2]₁
  E = t_eval + v·r_eval + v²·ā + ... + v⁶·s̄_σ2
    = t_eval + v·(D(ζ) + r₀) + v²·ā + ...

  페어링: e(W_ζ, [τ-ζ]₂) = e([F]₁ - E·G₁, G₂)

사용 예시:
    >>> from zkp.plonk.verifier import verify
    >>> result = verify(proof, public_inputs, preprocessed, srs)
"""

from zkp.plonk.field import FR, G1, G2, ec_mul, ec_add, ec_neg, ec_pairing
from zkp.plonk.transcript import Transcript
from zkp.plonk.permutation import K1, K2
from zkp.plonk.utils import vanishing_poly_eval, lagrange_basis_eval


def verify(proof, public_inputs, preprocessed, srs):
    """PLONK 증명을 검증한다.

    Args:
        proof: Proof 객체 (prover.prove()의 결과)
        public_inputs: 공개 입력 값 리스트 (현재 구현에서는 사용하지 않음)
        preprocessed: PreprocessedData (전처리 결과)
        srs: SRS

    Returns:
        bool: 검증 성공 여부
    """
    pp = preprocessed
    n = pp.n
    omega = pp.omega

    # ── Step 1: Fiat-Shamir 트랜스크립트 재생 ──
    transcript = Transcript()

    transcript.append_point(b"a_comm", proof.a_comm)
    transcript.append_point(b"b_comm", proof.b_comm)
    transcript.append_point(b"c_comm", proof.c_comm)

    beta = transcript.challenge_scalar(b"beta")
    gamma = transcript.challenge_scalar(b"gamma")

    transcript.append_point(b"z_comm", proof.z_comm)

    alpha = transcript.challenge_scalar(b"alpha")

    transcript.append_point(b"t_lo_comm", proof.t_lo_comm)
    transcript.append_point(b"t_mid_comm", proof.t_mid_comm)
    transcript.append_point(b"t_hi_comm", proof.t_hi_comm)

    zeta = transcript.challenge_scalar(b"zeta")

    transcript.append_scalar(b"a_eval", proof.a_eval)
    transcript.append_scalar(b"b_eval", proof.b_eval)
    transcript.append_scalar(b"c_eval", proof.c_eval)
    transcript.append_scalar(b"s_sigma1_eval", proof.s_sigma1_eval)
    transcript.append_scalar(b"s_sigma2_eval", proof.s_sigma2_eval)
    transcript.append_scalar(b"z_omega_eval", proof.z_omega_eval)

    v = transcript.challenge_scalar(b"v")
    u = transcript.challenge_scalar(b"u")

    # ── Step 2: 공개 값 계산 ──
    a_eval = proof.a_eval
    b_eval = proof.b_eval
    c_eval = proof.c_eval
    s_sigma1_eval = proof.s_sigma1_eval
    s_sigma2_eval = proof.s_sigma2_eval
    z_omega_eval = proof.z_omega_eval

    zh_zeta = vanishing_poly_eval(n, zeta)
    l1_zeta = lagrange_basis_eval(0, n, omega, zeta)
    pi_zeta = FR(0)  # 현재 구현: PI(x) = 0

    # ── Step 3: 선형화 커밋먼트 [D]₁ ──
    # [D]₁ = 셀렉터·스칼라 + z·스칼라 - S_σ3·스칼라 + z·스칼라(경계)
    # D는 r(x)에서 "상수가 아닌" 다항식 부분의 커밋먼트

    # 게이트: ā·b̄·[q_M] + ā·[q_L] + b̄·[q_R] + c̄·[q_O] + [q_C]
    D = ec_mul(pp.q_m_comm, a_eval * b_eval)
    D = ec_add(D, ec_mul(pp.q_l_comm, a_eval))
    D = ec_add(D, ec_mul(pp.q_r_comm, b_eval))
    D = ec_add(D, ec_mul(pp.q_o_comm, c_eval))
    D = ec_add(D, pp.q_c_comm)

    # 순열 z(x) 항: α·(ā+βζ+γ)(b̄+βK1ζ+γ)(c̄+βK2ζ+γ)·[z]
    perm_z_scalar = (
        alpha
        * (a_eval + beta * zeta + gamma)
        * (b_eval + beta * K1 * zeta + gamma)
        * (c_eval + beta * K2 * zeta + gamma)
    )
    D = ec_add(D, ec_mul(proof.z_comm, perm_z_scalar))

    # 순열 S_σ3(x) 항: -α·(ā+βs̄1+γ)(b̄+βs̄2+γ)·β·z̄ω·[S_σ3]
    ab_factor = (
        (a_eval + beta * s_sigma1_eval + gamma)
        * (b_eval + beta * s_sigma2_eval + gamma)
    )
    perm_s3_scalar = alpha * ab_factor * beta * z_omega_eval
    D = ec_add(D, ec_neg(ec_mul(pp.s_sigma3_comm, perm_s3_scalar)))

    # 경계 z(x) 항: α²·L₁(ζ)·[z]
    D = ec_add(D, ec_mul(proof.z_comm, alpha * alpha * l1_zeta))

    # ── Step 4: r₀ (상수 기여분) ──
    # r₀ = PI(ζ) - α·(ā+βs̄1+γ)(b̄+βs̄2+γ)·(c̄+γ)·z̄ω - α²·L₁(ζ)
    r_0 = (
        pi_zeta
        - alpha * ab_factor * z_omega_eval * (c_eval + gamma)
        - alpha * alpha * l1_zeta
    )

    # ── Step 5: [F]₁ 및 [E]₁ 구성 ──

    # t_combined 커밋먼트
    zeta_n = zeta ** n
    zeta_2n = zeta_n * zeta_n
    t_comm = ec_add(
        proof.t_lo_comm,
        ec_add(
            ec_mul(proof.t_mid_comm, zeta_n),
            ec_mul(proof.t_hi_comm, zeta_2n)
        )
    )

    # [F] = [t_comb] + v·[D] + v·r₀·G₁ + v²·[a] + v³·[b] + v⁴·[c]
    #      + v⁵·[S_σ1] + v⁶·[S_σ2]
    F = t_comm
    F = ec_add(F, ec_mul(D, v))
    F = ec_add(F, ec_mul(G1, v * r_0))  # 상수항

    v_pow = v * v
    F = ec_add(F, ec_mul(proof.a_comm, v_pow))
    v_pow = v_pow * v
    F = ec_add(F, ec_mul(proof.b_comm, v_pow))
    v_pow = v_pow * v
    F = ec_add(F, ec_mul(proof.c_comm, v_pow))
    v_pow = v_pow * v
    F = ec_add(F, ec_mul(pp.s_sigma1_comm, v_pow))
    v_pow = v_pow * v
    F = ec_add(F, ec_mul(pp.s_sigma2_comm, v_pow))

    # E_scalar: 알려진 평가값의 결합
    # W_ζ 분자 = [t_comb - t_eval] + v·[r - r_eval] + v²·[a - ā] + ...
    # 따라서 E = t_eval + v·r_eval + v²·ā + v³·b̄ + ... + u·z̄_ω
    #
    # t_eval = r_eval / Z_H(ζ) (∵ r(ζ) = t(ζ)·Z_H(ζ))
    r_eval = proof.r_eval
    t_eval = r_eval / zh_zeta

    e_scalar = t_eval + v * r_eval
    v_pow = v * v
    e_scalar = e_scalar + v_pow * a_eval
    v_pow = v_pow * v
    e_scalar = e_scalar + v_pow * b_eval
    v_pow = v_pow * v
    e_scalar = e_scalar + v_pow * c_eval
    v_pow = v_pow * v
    e_scalar = e_scalar + v_pow * s_sigma1_eval
    v_pow = v_pow * v
    e_scalar = e_scalar + v_pow * s_sigma2_eval
    e_scalar = e_scalar + u * z_omega_eval

    E = ec_mul(G1, e_scalar)

    # ── Step 6: 페어링 검사 ──
    # [A] = W_ζ + u·W_ζω
    # [B] = ζ·W_ζ + u·ζω·W_ζω + F + u·[z] - E
    # e([A], [τ]₂) == e([B], G₂)

    A = ec_add(proof.W_zeta_comm, ec_mul(proof.W_zeta_omega_comm, u))

    B = ec_mul(proof.W_zeta_comm, zeta)
    B = ec_add(B, ec_mul(proof.W_zeta_omega_comm, u * zeta * omega))
    B = ec_add(B, F)
    B = ec_add(B, ec_mul(proof.z_comm, u))
    B = ec_add(B, ec_neg(E))

    lhs = ec_pairing(srs.g2_powers[1], A)
    rhs = ec_pairing(srs.g2_powers[0], B)

    return lhs == rhs
