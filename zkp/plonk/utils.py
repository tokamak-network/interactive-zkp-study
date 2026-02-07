"""
PLONK 공유 유틸리티
===================

여러 PLONK 모듈에서 공유되는 수학적 유틸리티 함수를 제공한다.

**주요 기능**:
  - vanishing_poly_eval: 소거 다항식 Z_H(ζ) = ζ^n - 1 평가
  - lagrange_basis_eval: i번째 Lagrange 기저 L_i(ζ) 평가
  - public_input_polynomial: 공개 입력(public input) 다항식 PI(x) 구성
  - coset_fft / coset_ifft: 코셋 FFT (Round 3에서 Z_H의 영점 회피)
  - pad_to_power_of_2: 리스트를 2의 거듭제곱 길이로 패딩

**코셋 FFT 설명**:
  Round 3에서 t(x) = C(x) / Z_H(x)를 계산할 때, 도메인 H 위에서
  Z_H(x) = 0이므로 직접 나눌 수 없다.
  코셋 k·H = {k, k·ω, k·ω², ...} (k ∈ FR, k ∉ H)에서 평가하면
  Z_H(k·ωⁱ) ≠ 0이 되어 나눗셈이 가능하다.
"""

from zkp.plonk.field import FR
from zkp.plonk.polynomial import Polynomial, fft, ifft


def vanishing_poly_eval(n, zeta):
    """소거 다항식 Z_H(ζ) = ζ^n - 1 을 평가한다.

    Z_H(x) = x^n - 1 은 도메인 H = {1, ω, ..., ω^(n-1)} 위에서 0이 되는 다항식이다.
    Verifier가 검증 시 효율적으로 계산할 수 있다 (O(log n) 지수 연산).

    Args:
        n: 도메인 크기
        zeta: 평가 점 (FR 원소)

    Returns:
        FR: ζ^n - 1

    예시 (x³+x+5=35 회로, n=4):
        >>> zeta = FR(17)  # 임의의 챌린지 포인트
        >>> vanishing_poly_eval(4, zeta)  # 17^4 - 1
    """
    return zeta ** n - FR(1)


def lagrange_basis_eval(i, n, omega, zeta):
    """i번째 Lagrange 기저 다항식 L_i(ζ)를 평가한다.

    공식:
        L_i(ζ) = (ω^i / n) · (ζ^n - 1) / (ζ - ω^i)

    이는 다음과 동치:
        L_i(ζ) = (1/n) · Z_H(ζ) · ω^i / (ζ - ω^i)

    성질: L_i(ω^j) = δ_{ij} (크로네커 델타)

    PLONK에서의 사용:
    - L₁(ζ) (i=0): 경계 제약 확인에 사용
    - PI(ζ): 공개 입력 다항식 평가에 사용

    Args:
        i: 기저 인덱스 (0 ≤ i < n)
        n: 도메인 크기
        omega: n차 원시 단위근
        zeta: 평가 점

    Returns:
        FR: L_i(ζ)
    """
    if not isinstance(zeta, FR):
        zeta = FR(zeta)

    omega_i = omega ** i
    zh_zeta = vanishing_poly_eval(n, zeta)

    # ζ가 도메인 위의 점인 경우 (ζ = ω^i) → L_i(ζ) = 1
    denominator = zeta - omega_i
    if denominator == FR(0):
        return FR(1)

    n_inv = FR(1) / FR(n)
    return n_inv * zh_zeta * omega_i / denominator


def public_input_polynomial(pub_inputs, n, omega):
    """공개 입력(public input) 다항식 PI(x)를 구성한다.

    PI(x) = Σᵢ wᵢ · Lᵢ(x)    (i = 0, ..., |pub_inputs|-1)

    여기서 wᵢ는 i번째 공개 입력 값, Lᵢ(x)는 i번째 Lagrange 기저 다항식이다.

    PLONK에서는 게이트 제약에 PI(x) 항이 포함된다:
        q_L·a + q_R·b + q_O·c + q_M·a·b + q_C + PI(x) = 0

    Args:
        pub_inputs: 공개 입력 값 리스트 [w₀, w₁, ...]
        n: 도메인 크기
        omega: n차 원시 단위근

    Returns:
        Polynomial: PI(x)

    예시 (x³+x+5=35 회로):
        공개 입력이 [35] (출력값)이고 게이트 3에서 사용된다면,
        PI(x) = 35 · L₃(x) — 하지만 실제 위치는 회로 설계에 따라 다르다.
    """
    if not pub_inputs:
        return Polynomial.zero()

    # 평가값 리스트 구성: 공개 입력 위치에 값, 나머지는 0
    evals = [FR(0)] * n
    for i, val in enumerate(pub_inputs):
        if not isinstance(val, FR):
            val = FR(val)
        evals[i] = val

    return Polynomial.from_evaluations(evals, omega)


def public_input_poly_eval(pub_inputs, n, omega, zeta):
    """공개 입력 다항식 PI(ζ)를 효율적으로 평가한다.

    PI(ζ) = Σᵢ wᵢ · Lᵢ(ζ)

    다항식 전체를 구성하지 않고, Lagrange 기저의 평가값만 사용하여
    효율적으로 계산한다.

    Args:
        pub_inputs: 공개 입력 값 리스트
        n: 도메인 크기
        omega: n차 원시 단위근
        zeta: 평가 점

    Returns:
        FR: PI(ζ)
    """
    result = FR(0)
    for i, val in enumerate(pub_inputs):
        if not isinstance(val, FR):
            val = FR(val)
        li = lagrange_basis_eval(i, n, omega, zeta)
        result = result + val * li
    return result


def coset_fft(coeffs, omega, k=None):
    """코셋 FFT: 다항식을 코셋 k·H에서 평가한다.

    일반 FFT가 도메인 H = {1, ω, ω², ...}에서 평가하는 반면,
    코셋 FFT는 k·H = {k, k·ω, k·ω², ...}에서 평가한다.

    방법: p(x)의 계수 [c₀, c₁, ...] 를 [c₀, k·c₁, k²·c₂, ...] 로 변환 후 FFT.
    이는 p(k·x)를 H에서 평가하는 것과 동일하다.

    Round 3에서 사용:
        Z_H(x)는 H 위에서 0이므로, t(x) = C(x)/Z_H(x)를 H에서 직접 계산할 수 없다.
        코셋 k·H에서 평가하면 Z_H(k·ωⁱ) ≠ 0이 되어 나눗셈이 가능하다.

    Args:
        coeffs: 다항식 계수 리스트 (FR 원소)
        omega: n차 원시 단위근
        k: 코셋 생성자 (기본값: FR(5))

    Returns:
        list[FR]: 코셋에서의 평가값
    """
    if k is None:
        k = FR(5)  # bn128에서의 관례적 코셋 생성자
    # 계수 변환: cᵢ → kⁱ · cᵢ
    shifted = []
    k_power = FR(1)
    for c in coeffs:
        if not isinstance(c, FR):
            c = FR(c)
        shifted.append(c * k_power)
        k_power = k_power * k
    return fft(shifted, omega)


def coset_ifft(evals, omega, k=None):
    """코셋 IFFT: 코셋 k·H에서의 평가값을 계수로 복원한다.

    코셋 FFT의 역변환.
    1. 일반 IFFT 수행
    2. 계수에서 kⁱ 인자를 제거: cᵢ / kⁱ

    Args:
        evals: 코셋에서의 평가값 리스트
        omega: n차 원시 단위근
        k: 코셋 생성자 (기본값: FR(5))

    Returns:
        list[FR]: 원래 다항식의 계수
    """
    if k is None:
        k = FR(5)
    # 먼저 일반 IFFT
    coeffs = ifft(evals, omega)
    # kⁱ 인자 제거
    k_inv = FR(1) / k
    k_inv_power = FR(1)
    result = []
    for c in coeffs:
        result.append(c * k_inv_power)
        k_inv_power = k_inv_power * k_inv
    return result


def pad_to_power_of_2(lst, fill=None):
    """리스트를 2의 거듭제곱 길이로 패딩한다.

    FFT는 입력 길이가 2의 거듭제곱이어야 한다.

    Args:
        lst: 패딩할 리스트
        fill: 채울 값 (기본값: FR(0))

    Returns:
        list: 2의 거듭제곱 길이의 리스트
    """
    if fill is None:
        fill = FR(0)
    n = len(lst)
    target = next_power_of_2(n)
    return list(lst) + [fill] * (target - n)


def next_power_of_2(n):
    """n 이상의 가장 작은 2의 거듭제곱을 반환한다.

    Args:
        n: 양의 정수

    Returns:
        int: 2의 거듭제곱

    예시:
        >>> next_power_of_2(3)  # 4
        >>> next_power_of_2(4)  # 4
        >>> next_power_of_2(5)  # 8
    """
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p
