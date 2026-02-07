"""
PLONK 순열 인자 (Permutation Argument)
========================================

배선 복사 제약(copy constraint)을 순열(permutation)로 인코딩하고,
Grand Product 논증으로 증명하는 모듈.

**배경: 왜 순열이 필요한가?**
  PLONK 게이트는 독립적으로 q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0을
  만족하지만, 서로 다른 게이트 간에 "같은 값"을 강제할 방법이 없다.
  예: 게이트 0의 출력(c₀)이 게이트 1의 입력(a₁)과 같아야 할 때.

  해결: 3n개의 배선 위치에 순열 σ를 정의하고,
  "w_{σ(i)} = wᵢ for all i"를 Grand Product로 증명한다.

**코셋 식별자 K1, K2**:
  3n개의 배선 위치를 3개의 코셋으로 분리:
  - a 배선: {ω⁰, ω¹, ..., ω^{n-1}}       (코셋 1·H)
  - b 배선: {K1·ω⁰, K1·ω¹, ..., K1·ω^{n-1}} (코셋 K1·H)
  - c 배선: {K2·ω⁰, K2·ω¹, ..., K2·ω^{n-1}} (코셋 K2·H)
  K1, K2는 서로 다른 코셋을 보장하는 상수 (K1=2, K2=3).

**Grand Product (순열 누적자 z(x))**:
  z(ω⁰) = 1
  z(ωⁱ⁺¹) = z(ωⁱ) · ∏ₖ (wₖ(ωⁱ) + β·id_k(ωⁱ) + γ) / (wₖ(ωⁱ) + β·σₖ(ωⁱ) + γ)
  z(ω^n) = 1 (순열이 올바르면)

사용 예시:
    >>> sigma = circuit.build_copy_constraints()
    >>> S1, S2, S3 = build_permutation_polynomials(sigma, n, domain)
"""

from zkp.plonk.field import FR
from zkp.plonk.polynomial import Polynomial


# 코셋 식별자 (coset identifier)
# H, K1·H, K2·H가 서로소인 코셋이 되도록 선택
# K1, K2는 H의 원소가 아닌 값이어야 함
K1 = FR(2)
K2 = FR(3)


def build_permutation_polynomials(sigma, n, domain):
    """순열 σ를 3개의 다항식 S_σ1, S_σ2, S_σ3으로 인코딩한다.

    순열 σ는 3n개의 위치를 재배열한다:
    - 위치 0..n-1:   a 배선 → 도메인 원소 {ω⁰, ω¹, ..., ω^{n-1}}
    - 위치 n..2n-1:  b 배선 → {K1·ω⁰, ..., K1·ω^{n-1}}
    - 위치 2n..3n-1: c 배선 → {K2·ω⁰, ..., K2·ω^{n-1}}

    σ(위치)가 가리키는 코셋·도메인 원소를 다항식으로 인코딩.

    Args:
        sigma: 순열 배열 (길이 3n, build_copy_constraints()의 결과)
        n: 게이트 수
        domain: 도메인 [ω⁰, ω¹, ..., ω^{n-1}]

    Returns:
        tuple: (S_sigma1_evals, S_sigma2_evals, S_sigma3_evals)
               각각 길이 n의 FR 원소 리스트 (평가값)

    예시 (x³+x+5=35, n=4):
        sigma[0]=4 이면 a₀은 b₀(위치4)과 연결 → S_σ1[0] = K1·ω⁰
    """
    # 위치 → 코셋·도메인 원소 매핑
    def position_to_value(pos):
        """순열 위치를 해당하는 코셋·도메인 값으로 변환.

        위치 i ∈ [0, n):   1 · ω^i
        위치 i ∈ [n, 2n):  K1 · ω^{i-n}
        위치 i ∈ [2n, 3n): K2 · ω^{i-2n}
        """
        if pos < n:
            return domain[pos]
        elif pos < 2 * n:
            return K1 * domain[pos - n]
        else:
            return K2 * domain[pos - 2 * n]

    # 각 배선 그룹의 순열 다항식 평가값
    s_sigma1_evals = [position_to_value(sigma[i]) for i in range(n)]
    s_sigma2_evals = [position_to_value(sigma[n + i]) for i in range(n)]
    s_sigma3_evals = [position_to_value(sigma[2 * n + i]) for i in range(n)]

    return s_sigma1_evals, s_sigma2_evals, s_sigma3_evals


def compute_accumulator(a_vals, b_vals, c_vals, sigma, n, domain, beta, gamma):
    """순열 누적자(grand product accumulator) z의 평가값을 계산한다.

    z(ωⁱ)는 다음과 같이 정의:
        z(ω⁰) = 1
        z(ωⁱ⁺¹) = z(ωⁱ) ·
            (a(ωⁱ) + β·ωⁱ + γ)(b(ωⁱ) + β·K1·ωⁱ + γ)(c(ωⁱ) + β·K2·ωⁱ + γ)
            ─────────────────────────────────────────────────────────────────
            (a(ωⁱ) + β·S_σ1(ωⁱ) + γ)(b(ωⁱ) + β·S_σ2(ωⁱ) + γ)(c(ωⁱ) + β·S_σ3(ωⁱ) + γ)

    분자: "순열 σ가 항등"이라 가정했을 때의 값
    분모: "실제 순열 σ"를 적용한 값

    σ가 올바르면 z(ω^n) = 1 (분자·분모 텔레스코핑).

    Args:
        a_vals, b_vals, c_vals: 배선 값 리스트 (길이 n)
        sigma: 순열 배열 (길이 3n)
        n: 게이트 수
        domain: [ω⁰, ..., ω^{n-1}]
        beta: β 챌린지 (FR)
        gamma: γ 챌린지 (FR)

    Returns:
        list[FR]: z의 평가값 [z(ω⁰)=1, z(ω¹), ..., z(ω^{n-1})]
    """
    # 순열 다항식 평가값
    s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

    z_evals = [FR(1)]  # z(ω⁰) = 1

    for i in range(n - 1):
        # 분자: (aᵢ + β·ωⁱ + γ)(bᵢ + β·K1·ωⁱ + γ)(cᵢ + β·K2·ωⁱ + γ)
        num = (
            (a_vals[i] + beta * domain[i] + gamma)
            * (b_vals[i] + beta * K1 * domain[i] + gamma)
            * (c_vals[i] + beta * K2 * domain[i] + gamma)
        )

        # 분모: (aᵢ + β·S_σ1(ωⁱ) + γ)(bᵢ + β·S_σ2(ωⁱ) + γ)(cᵢ + β·S_σ3(ωⁱ) + γ)
        den = (
            (a_vals[i] + beta * s1[i] + gamma)
            * (b_vals[i] + beta * s2[i] + gamma)
            * (c_vals[i] + beta * s3[i] + gamma)
        )

        z_evals.append(z_evals[-1] * num / den)

    return z_evals
