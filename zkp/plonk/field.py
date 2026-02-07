"""
PLONK 기반 모듈: 유한체(Finite Field) 및 타원곡선 연산
========================================================

이 모듈은 PLONK 프로토콜 전체에서 사용되는 기본 대수적 도구를 정의한다.

**유한체 FR**:
  bn128 타원곡선의 스칼라 필드 (scalar field). 모든 PLONK 다항식 연산과
  증명 생성/검증에서 사용되는 기본 산술 단위이다.
  - 위수(order) p ≈ 2^254, 소수체(prime field)
  - p - 1 = 2^28 × m (m은 홀수) → 최대 2^28차 단위근(root of unity)을 지원

**타원곡선 연산**:
  KZG 다항식 커밋먼트와 검증을 위한 G1, G2 그룹 연산 및 페어링.

**단위근(Roots of Unity)**:
  FFT/IFFT와 다항식 보간에 필수적인 n차 원시 단위근.
  PLONK에서 도메인 H = {1, ω, ω², ..., ω^(n-1)}을 정의하는 데 사용된다.

사용 예시:
    >>> from zkp.plonk.field import FR, G1, ec_mul
    >>> a = FR(3)
    >>> b = FR(7)
    >>> c = a * b        # FR(21)
    >>> P = ec_mul(G1, 5)  # 5·G1
"""

from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128


# ─────────────────────────────────────────────────────────────────────
# 유한체(Finite Field) FR
# ─────────────────────────────────────────────────────────────────────

class FR(FQ):
    """bn128 스칼라 필드 위의 유한체 원소.

    bn128.curve_order (≈ 2^254) 위의 모듈러 산술을 지원한다.
    py_ecc의 FQ 클래스를 상속하여 +, -, *, /, ** 등의 필드 연산을 제공한다.

    속성:
        field_modulus: bn128 곡선 위수 (소수 p)

    예시:
        >>> x = FR(3)
        >>> x * x          # FR(9)
        >>> x ** 2          # FR(9)
        >>> FR(1) / FR(3)   # 3의 모듈러 역원
    """
    field_modulus = bn128.curve_order


# 곡선 위수 (필드 크기)
CURVE_ORDER = bn128.curve_order


# ─────────────────────────────────────────────────────────────────────
# 타원곡선 상수 및 연산
# ─────────────────────────────────────────────────────────────────────

# G1 그룹 생성자 (generator)
G1 = bn128.G1

# G2 그룹 생성자 (generator)
G2 = bn128.G2

# 영점 (point at infinity) - 항등원
Z1 = None  # bn128에서 G1의 항등원은 None으로 표현


def ec_mul(point, scalar):
    """타원곡선 스칼라 곱셈: scalar · point.

    Args:
        point: G1 또는 G2 위의 점
        scalar: 정수 또는 FR 원소

    Returns:
        scalar · point (같은 그룹의 점)

    예시:
        >>> P = ec_mul(G1, FR(5))  # 5·G1
        >>> Q = ec_mul(G2, 3)       # 3·G2
    """
    if isinstance(scalar, FR):
        scalar = int(scalar)
    return bn128.multiply(point, scalar % CURVE_ORDER)


def ec_add(p1, p2):
    """타원곡선 점 덧셈: p1 + p2.

    Args:
        p1, p2: 같은 그룹 (G1 또는 G2)의 점

    Returns:
        p1 + p2

    예시:
        >>> P = ec_add(G1, G1)  # 2·G1
    """
    return bn128.add(p1, p2)


def ec_neg(point):
    """타원곡선 점의 역원 (negation): -point.

    Args:
        point: G1 또는 G2 위의 점

    Returns:
        -point (y좌표 반전)
    """
    return bn128.neg(point)


def ec_pairing(g2_point, g1_point):
    """쌍선형 페어링 e(G1, G2) → GT.

    bn128의 optimal Ate 페어링을 수행한다.
    KZG 검증에서 다항식 열기 증명의 정당성을 확인하는 데 사용된다.

    Args:
        g2_point: G2 위의 점
        g1_point: G1 위의 점

    Returns:
        GT 원소 (페어링 결과)

    주의:
        py_ecc.bn128.pairing의 인자 순서는 (G2, G1)이다.

    예시:
        >>> e1 = ec_pairing(G2, G1)         # e(G1, G2)
        >>> e2 = ec_pairing(G2, ec_mul(G1, 5))  # e(5·G1, G2)
    """
    return bn128.pairing(g2_point, g1_point)


# ─────────────────────────────────────────────────────────────────────
# 단위근 (Roots of Unity)
# ─────────────────────────────────────────────────────────────────────

def get_root_of_unity(n):
    """n차 원시 단위근(primitive n-th root of unity) ω를 반환한다.

    ω^n = 1이고, ω^k ≠ 1 (0 < k < n)인 원소 ω를 찾는다.

    bn128 곡선의 경우:
        p - 1 = 2^28 × m (m은 홀수)
        따라서 최대 2^28차 단위근까지 지원한다.
        생성자 g = FR(5)를 사용하여 ω = g^((p-1)/n)으로 계산한다.

    Args:
        n: 단위근의 차수 (2의 거듭제곱이어야 하며, ≤ 2^28)

    Returns:
        FR: n차 원시 단위근

    Raises:
        ValueError: n이 2의 거듭제곱이 아니거나 2^28을 초과할 때

    예시:
        >>> omega = get_root_of_unity(4)
        >>> omega ** 4 == FR(1)  # True
        >>> omega ** 2 != FR(1)  # True (원시 단위근)
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n은 2의 거듭제곱이어야 합니다: {n}")
    if n > (1 << 28):
        raise ValueError(f"n은 2^28 이하여야 합니다: {n}")
    if n == 1:
        return FR(1)

    # 생성자 g = FR(5)는 FR*의 원시 원소에서 유도
    # ω = g^((p-1)/n)이면 ω^n = g^(p-1) = 1 (페르마 소정리)
    g = FR(5)
    exponent = (CURVE_ORDER - 1) // n
    omega = g ** exponent

    return omega


def get_roots_of_unity(n):
    """n개의 단위근 리스트 [1, ω, ω², ..., ω^(n-1)]을 반환한다.

    이 리스트는 PLONK의 평가 도메인 H를 정의한다.
    FFT/IFFT에서 다항식 평가/보간에 사용된다.

    Args:
        n: 도메인 크기 (2의 거듭제곱)

    Returns:
        list[FR]: [ω^0, ω^1, ..., ω^(n-1)]

    예시:
        >>> roots = get_roots_of_unity(4)
        >>> len(roots)  # 4
        >>> roots[0] == FR(1)  # True
        >>> all(r ** 4 == FR(1) for r in roots)  # True
    """
    omega = get_root_of_unity(n)
    roots = []
    current = FR(1)
    for _ in range(n):
        roots.append(current)
        current = current * omega
    return roots
