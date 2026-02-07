"""
KZG 다항식 커밋먼트 스킴
=========================

Kate-Zaverucha-Goldberg (KZG) 커밋먼트는 PLONK의 핵심 빌딩 블록이다.

**KZG 커밋먼트란?**
  다항식 p(x)에 대한 간결한 "지문"(커밋먼트)을 타원곡선 점으로 생성한다.
  - 커밋먼트: C = p(τ)·G1 (τ는 SRS의 비밀 값)
  - 바인딩(binding): 한 번 커밋하면 다른 다항식으로 바꿀 수 없음
  - 하이딩(hiding): 커밋먼트에서 원래 다항식을 복원할 수 없음

**열기 증명 (Opening Proof)**:
  "p(z) = y" 임을 증명하는 방법:
  1. 몫 다항식 q(x) = (p(x) - y) / (x - z) 계산
     (p(z) = y이면 (x-z)가 (p(x)-y)를 나누므로 q(x)는 다항식)
  2. 증명 π = q(τ)·G1
  3. 검증: e(C - y·G1, G2) == e(π, τ·G2 - z·G2)
     즉, 페어링으로 다항식 관계를 확인

사용 예시:
    >>> from zkp.plonk.kzg import commit, create_witness, verify_opening
    >>> C = commit(poly, srs)
    >>> proof = create_witness(poly, FR(7), srs)
    >>> verify_opening(C, proof, FR(7), poly.evaluate(FR(7)), srs)  # True
"""

from zkp.plonk.field import FR, G1, ec_mul, ec_add, ec_neg, ec_pairing
from zkp.plonk.polynomial import Polynomial, poly_div


def commit(poly, srs):
    """다항식을 KZG 커밋한다.

    C = Σᵢ cᵢ · [τⁱ]₁ = p(τ) · G1

    SRS의 G1 powers [G1, τG1, τ²G1, ...]에 다항식 계수를 곱하여
    선형결합한다. τ를 모르는 상태에서 p(τ)·G1을 계산하는 것이다.

    Args:
        poly: 커밋할 다항식 (Polynomial)
        srs: Structured Reference String (SRS)

    Returns:
        G1 점: 커밋먼트 C

    Raises:
        ValueError: 다항식 차수가 SRS 최대 차수를 초과할 때

    예시:
        >>> p = Polynomial([FR(1), FR(2), FR(3)])  # 1 + 2x + 3x²
        >>> C = commit(p, srs)  # (1 + 2τ + 3τ²)·G1
    """
    if poly.degree > srs.max_degree:
        raise ValueError(
            f"다항식 차수 {poly.degree}가 SRS 최대 차수 {srs.max_degree}를 초과합니다"
        )

    # C = Σ cᵢ · [τⁱ]₁
    result = None  # 무한원점 (항등원)
    for i, coeff in enumerate(poly.coeffs):
        if coeff == FR(0):
            continue
        term = ec_mul(srs.g1_powers[i], coeff)
        result = ec_add(result, term)

    return result


def create_witness(poly, point, srs):
    """열기 증명(opening proof)을 생성한다.

    p(z) = y일 때, q(x) = (p(x) - y) / (x - z)를 계산하고
    증명 π = commit(q, srs)를 반환한다.

    수학적 근거:
        p(z) = y이면 (p(x) - y)는 (x - z)로 나누어 떨어진다.
        (다항식의 인수정리: f(a) = 0 ⟺ (x-a) | f(x))

    Args:
        poly: 열어볼 다항식 p(x)
        point: 평가 점 z (FR 원소)
        srs: SRS

    Returns:
        G1 점: 열기 증명 π

    예시:
        >>> p = Polynomial([FR(1), FR(2)])  # 1 + 2x
        >>> # p(3) = 7
        >>> proof = create_witness(p, FR(3), srs)
    """
    if not isinstance(point, FR):
        point = FR(point)

    # y = p(z)
    y = poly.evaluate(point)

    # p(x) - y
    p_minus_y = poly - Polynomial([y])

    # (x - z)
    divisor = Polynomial([FR(0) - point, FR(1)])

    # q(x) = (p(x) - y) / (x - z)
    quotient, remainder = poly_div(p_minus_y, divisor)

    # 나머지가 0이어야 함 (p(z) = y이므로)
    for c in remainder.coeffs:
        if c != FR(0):
            raise ValueError("열기 증명 생성 실패: 나머지가 0이 아닙니다")

    # π = commit(q, srs)
    return commit(quotient, srs)


def verify_opening(commitment, proof, point, evaluation, srs):
    """KZG 열기 증명을 검증한다.

    검증 방정식 (페어링):
        e(C - y·G1, G2) == e(π, τ·G2 - z·G2)

    변형 (효율적인 검증):
        e(π, τ·G2) == e(π·z + C - y·G1, G2)

    Args:
        commitment: 다항식 커밋먼트 C (G1 점)
        proof: 열기 증명 π (G1 점)
        point: 평가 점 z (FR 원소)
        evaluation: 주장하는 평가값 y = p(z) (FR 원소)
        srs: SRS

    Returns:
        bool: 검증 성공 여부

    예시:
        >>> C = commit(p, srs)
        >>> pi = create_witness(p, FR(3), srs)
        >>> verify_opening(C, pi, FR(3), p.evaluate(FR(3)), srs)  # True
    """
    if not isinstance(point, FR):
        point = FR(point)
    if not isinstance(evaluation, FR):
        evaluation = FR(evaluation)

    # LHS: e(π, [τ]₂ - z·[1]₂) = e(π, [τ-z]₂)
    # τ·G2 - z·G2
    tau_g2 = srs.g2_powers[1]  # [τ]₂
    z_g2 = ec_mul(srs.g2_powers[0], point)  # z·G2
    tau_minus_z_g2 = ec_add(tau_g2, ec_neg(z_g2))  # [τ-z]₂

    # RHS: e(C - y·G1, G2)
    y_g1 = ec_mul(G1, evaluation)  # y·G1
    c_minus_y = ec_add(commitment, ec_neg(y_g1))  # C - y·G1

    # 페어링 검사: e(C - y·G1, G2) == e(π, [τ-z]₂)
    lhs = ec_pairing(srs.g2_powers[0], c_minus_y)  # e(C - y·G1, G2)
    rhs = ec_pairing(tau_minus_z_g2, proof)  # e(π, [τ-z]₂)

    return lhs == rhs
