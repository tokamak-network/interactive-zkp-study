"""
PLONK 기반 모듈: 다항식(Polynomial) 클래스 및 FFT
===================================================

이 모듈은 PLONK 프로토콜에서 사용되는 모든 다항식 연산을 제공한다.

**Polynomial 클래스**:
  계수(coefficient) 표현 기반 다항식. p(x) = c₀ + c₁·x + c₂·x² + ...
  산술 연산자(+, -, *, 스칼라곱)와 평가(evaluation)를 지원한다.

**FFT/IFFT (Number Theoretic Transform)**:
  유한체 위의 다항식을 평가 표현 ↔ 계수 표현으로 변환.
  - FFT: 계수 → n개의 단위근에서의 평가값
  - IFFT: 평가값 → 계수 (보간)
  재귀적 Cooley-Tukey radix-2 알고리즘을 사용한다.

**다항식 나눗셈 (poly_div)**:
  PLONK에서 몫 다항식 t(x) 계산에 필수적이다.
  t(x) = 제약 다항식 / Z_H(x)에서 나머지가 0이 되어야 한다.

사용 예시:
    >>> from zkp.plonk.polynomial import Polynomial, fft, ifft
    >>> p = Polynomial([FR(1), FR(2), FR(3)])  # 1 + 2x + 3x²
    >>> p.evaluate(FR(2))  # 1 + 4 + 12 = FR(17)
"""

from zkp.plonk.field import FR


# ─────────────────────────────────────────────────────────────────────
# Polynomial 클래스
# ─────────────────────────────────────────────────────────────────────

class Polynomial:
    """유한체 FR 위의 다항식.

    계수 리스트로 표현: coeffs = [c₀, c₁, c₂, ...] → c₀ + c₁x + c₂x² + ...

    PLONK 프로토콜에서의 역할:
    - 배선(witness) 다항식 a(x), b(x), c(x): 회로의 배선 값을 인코딩
    - 셀렉터 다항식 q_L(x), q_R(x), ...: 게이트 유형을 인코딩
    - 순열 다항식 S_σ(x): 배선 연결 관계를 인코딩
    - 몫 다항식 t(x): 모든 제약이 만족됨을 증명

    예시:
        >>> p = Polynomial([FR(1), FR(2)])  # 1 + 2x
        >>> q = Polynomial([FR(3), FR(4)])  # 3 + 4x
        >>> r = p + q                        # 4 + 6x
        >>> r = p * q                        # 3 + 10x + 8x²
    """

    def __init__(self, coeffs=None):
        """다항식 생성.

        Args:
            coeffs: FR 원소의 리스트 [c₀, c₁, ...].
                    None이면 영 다항식(0)을 생성한다.
        """
        if coeffs is None:
            self.coeffs = [FR(0)]
        else:
            # FR 원소로 변환
            self.coeffs = [c if isinstance(c, FR) else FR(c) for c in coeffs]
        self._trim()

    def _trim(self):
        """최고차 계수가 0인 항을 제거하여 정규화한다.

        예: [1, 2, 0, 0] → [1, 2]  (1 + 2x)
        """
        while len(self.coeffs) > 1 and self.coeffs[-1] == FR(0):
            self.coeffs.pop()

    @property
    def degree(self):
        """다항식의 차수. 영 다항식의 차수는 0으로 정의한다."""
        if len(self.coeffs) == 1 and self.coeffs[0] == FR(0):
            return 0
        return len(self.coeffs) - 1

    def is_zero(self):
        """영 다항식인지 확인."""
        return len(self.coeffs) == 1 and self.coeffs[0] == FR(0)

    def evaluate(self, point):
        """다항식을 주어진 점에서 평가한다 (Horner's method).

        Horner's method: p(x) = c₀ + x(c₁ + x(c₂ + ...))
        일반적인 방법보다 곱셈 횟수가 적어 효율적이다.

        Args:
            point: 평가할 FR 원소

        Returns:
            FR: p(point) 값

        예시:
            >>> p = Polynomial([FR(1), FR(2), FR(3)])  # 1 + 2x + 3x²
            >>> p.evaluate(FR(2))  # 1 + 4 + 12 = FR(17)
        """
        if not isinstance(point, FR):
            point = FR(point)
        result = FR(0)
        for coeff in reversed(self.coeffs):
            result = result * point + coeff
        return result

    def __add__(self, other):
        """다항식 덧셈: p(x) + q(x)."""
        if isinstance(other, (int, FR)):
            other = Polynomial([other])
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else FR(0)
            b = other.coeffs[i] if i < len(other.coeffs) else FR(0)
            result.append(a + b)
        return Polynomial(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """다항식 뺄셈: p(x) - q(x)."""
        if isinstance(other, (int, FR)):
            other = Polynomial([other])
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else FR(0)
            b = other.coeffs[i] if i < len(other.coeffs) else FR(0)
            result.append(a - b)
        return Polynomial(result)

    def __rsub__(self, other):
        if isinstance(other, (int, FR)):
            other = Polynomial([other])
        return other.__sub__(self)

    def __neg__(self):
        """다항식 부호 반전: -p(x)."""
        return Polynomial([FR(0) - c for c in self.coeffs])

    def __mul__(self, other):
        """다항식 곱셈: p(x) · q(x) 또는 스칼라곱.

        다항식 × 다항식: O(n²) 나이브 곱셈 (교육적 명확성)
        다항식 × 스칼라: 각 계수에 스칼라를 곱함
        """
        if isinstance(other, (int, FR)):
            if isinstance(other, int):
                other = FR(other)
            return Polynomial([c * other for c in self.coeffs])
        # 다항식 곱셈 (convolution)
        result = [FR(0)] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = result[i + j] + a * b
        return Polynomial(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        """다항식 동등 비교."""
        if isinstance(other, (int, FR)):
            other = Polynomial([other])
        if not isinstance(other, Polynomial):
            return False
        return self.coeffs == other.coeffs

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == FR(0):
                continue
            if i == 0:
                terms.append(str(int(c)))
            elif i == 1:
                terms.append(f"{int(c)}*x")
            else:
                terms.append(f"{int(c)}*x^{i}")
        return "Poly(" + " + ".join(terms) + ")" if terms else "Poly(0)"

    def __len__(self):
        """계수 개수 반환 (차수 + 1)."""
        return len(self.coeffs)

    def scale(self, scalar):
        """스칼라곱: scalar · p(x).

        Args:
            scalar: FR 원소 또는 정수

        Returns:
            Polynomial: scalar · p(x)
        """
        return self * scalar

    def divide_by_vanishing(self, n):
        """소거 다항식 Z_H(x) = x^n - 1 로 나눈다.

        PLONK에서 몫 다항식 계산에 사용:
        t(x) = C(x) / Z_H(x)
        여기서 C(x)는 모든 제약을 결합한 다항식.

        나머지가 0이 아니면 오류 (제약이 만족되지 않음).

        Args:
            n: 도메인 크기 (Z_H(x) = x^n - 1의 n)

        Returns:
            Polynomial: 몫 다항식 t(x)

        Raises:
            ValueError: 나머지가 0이 아닌 경우
        """
        vanishing = Polynomial._vanishing_coeffs(n)
        q, r = poly_div(self, vanishing)
        # 나머지가 0인지 확인
        for c in r.coeffs:
            if c != FR(0):
                raise ValueError("소거 다항식으로 나누어 떨어지지 않습니다 (제약 불만족)")
        return q

    @staticmethod
    def _vanishing_coeffs(n):
        """Z_H(x) = x^n - 1 의 계수 표현을 반환한다.

        coeffs = [-1, 0, 0, ..., 0, 1]  (길이 n+1)
        """
        coeffs = [FR(0)] * (n + 1)
        coeffs[0] = FR(-1) if FR(-1) != FR(0) else FR(FR.field_modulus - 1)
        coeffs[0] = FR(FR.field_modulus - 1)  # -1 mod p
        coeffs[n] = FR(1)
        return Polynomial(coeffs)

    @classmethod
    def zero(cls):
        """영 다항식 p(x) = 0."""
        return cls([FR(0)])

    @classmethod
    def one(cls):
        """상수 다항식 p(x) = 1."""
        return cls([FR(1)])

    @classmethod
    def vanishing(cls, n):
        """소거 다항식 Z_H(x) = x^n - 1.

        PLONK에서 도메인 H = {1, ω, ..., ω^(n-1)} 위의 모든 점에서 0이 되는 다항식.
        Z_H(ωⁱ) = 0 for all i.

        Args:
            n: 도메인 크기

        Returns:
            Polynomial: x^n - 1
        """
        return cls._vanishing_coeffs(n)

    @classmethod
    def from_evaluations(cls, evals, omega):
        """평가값에서 다항식을 복원한다 (IFFT 사용).

        도메인 {1, ω, ω², ..., ω^(n-1)}에서의 평가값이 주어지면,
        이 값들을 보간하는 유일한 (n-1)차 이하 다항식을 반환한다.

        Args:
            evals: [p(1), p(ω), p(ω²), ...] FR 원소 리스트
            omega: n차 원시 단위근

        Returns:
            Polynomial: 보간된 다항식

        예시:
            >>> omega = get_root_of_unity(4)
            >>> # p(x) = 1 + 2x 의 평가값에서 복원
            >>> evals = [p.evaluate(omega**i) for i in range(4)]
            >>> q = Polynomial.from_evaluations(evals, omega)
            >>> q == p  # True
        """
        coeffs = ifft(evals, omega)
        return cls(coeffs)


# ─────────────────────────────────────────────────────────────────────
# FFT / IFFT (Number Theoretic Transform)
# ─────────────────────────────────────────────────────────────────────

def fft(coeffs, omega):
    """Fast Fourier Transform (NTT): 계수 → 평가값.

    재귀적 Cooley-Tukey radix-2 알고리즘.

    입력 다항식 p(x) = c₀ + c₁x + ... + c_{n-1}x^{n-1}을
    n개의 단위근 {1, ω, ω², ..., ω^{n-1}}에서 평가한다.

    알고리즘:
        1. n=1이면 계수를 그대로 반환
        2. 짝수/홀수 인덱스로 분리: even = [c₀, c₂, ...], odd = [c₁, c₃, ...]
        3. 재귀 호출: FFT(even, ω²), FFT(odd, ω²)
        4. 버터플라이 결합: y[k] = even[k] + ω^k · odd[k]
                           y[k+n/2] = even[k] - ω^k · odd[k]

    시간 복잡도: O(n log n) — 직접 평가 O(n²)보다 효율적

    Args:
        coeffs: [c₀, c₁, ..., c_{n-1}] FR 원소 리스트 (길이는 2의 거듭제곱)
        omega: n차 원시 단위근

    Returns:
        list[FR]: [p(1), p(ω), p(ω²), ..., p(ω^{n-1})]
    """
    n = len(coeffs)
    if n == 1:
        return [coeffs[0] if isinstance(coeffs[0], FR) else FR(coeffs[0])]

    # 짝수/홀수 분리 (Cooley-Tukey 분할)
    even = [coeffs[i] for i in range(0, n, 2)]
    odd = [coeffs[i] for i in range(1, n, 2)]

    # ω² = omega^2는 n/2차 단위근
    omega_sq = omega * omega

    # 재귀 호출
    even_vals = fft(even, omega_sq)
    odd_vals = fft(odd, omega_sq)

    # 버터플라이 결합
    result = [FR(0)] * n
    omega_k = FR(1)  # ω^k
    half = n // 2
    for k in range(half):
        t = omega_k * odd_vals[k]
        result[k] = even_vals[k] + t
        result[k + half] = even_vals[k] - t
        omega_k = omega_k * omega

    return result


def ifft(evals, omega):
    """Inverse FFT (INTT): 평가값 → 계수.

    FFT의 역변환. 평가값에서 원래 다항식의 계수를 복원한다.

    수학적 원리:
        FFT가 DFT 행렬 F를 곱하는 것이라면,
        IFFT는 F^{-1} = (1/n) · F(ω^{-1})을 곱하는 것이다.
        즉, 역 단위근 ω^{-1}로 FFT를 수행한 후 n으로 나눈다.

    Args:
        evals: [p(1), p(ω), ..., p(ω^{n-1})] FR 원소 리스트
        omega: n차 원시 단위근

    Returns:
        list[FR]: [c₀, c₁, ..., c_{n-1}] 계수 리스트

    예시:
        >>> omega = get_root_of_unity(4)
        >>> coeffs = [FR(1), FR(2), FR(3), FR(0)]
        >>> evals = fft(coeffs, omega)
        >>> recovered = ifft(evals, omega)
        >>> recovered == coeffs  # True
    """
    n = len(evals)
    # ω의 역원: ω^{-1} = ω^{n-2} (유한체에서 역원)
    # 또는 더 정확하게: ω^{-1} = ω^{p-2} (페르마 소정리)
    omega_inv = FR(1) / omega

    # 역 단위근으로 FFT 수행
    coeffs = fft(evals, omega_inv)

    # n으로 나누기 (1/n mod p)
    n_inv = FR(1) / FR(n)
    return [c * n_inv for c in coeffs]


# ─────────────────────────────────────────────────────────────────────
# 다항식 나눗셈 (Polynomial Long Division)
# ─────────────────────────────────────────────────────────────────────

def poly_div(a, b):
    """다항식 나눗셈: a(x) = b(x) · q(x) + r(x).

    긴 나눗셈(long division) 알고리즘으로 몫 q(x)와 나머지 r(x)를 계산한다.

    PLONK에서의 사용:
    - Round 3: 제약 다항식을 Z_H(x)로 나누어 몫 t(x) 계산
    - Round 5: 열기 증명에서 (p(x) - p(ζ)) / (x - ζ) 계산

    Args:
        a: 피제수 다항식 (Polynomial)
        b: 제수 다항식 (Polynomial)

    Returns:
        tuple: (몫 Polynomial, 나머지 Polynomial)

    Raises:
        ValueError: 제수가 영 다항식인 경우

    예시:
        >>> a = Polynomial([FR(-1), FR(0), FR(1)])  # x² - 1
        >>> b = Polynomial([FR(-1), FR(1)])          # x - 1
        >>> q, r = poly_div(a, b)
        >>> q  # x + 1
        >>> r  # 0
    """
    if b.is_zero():
        raise ValueError("0으로 나눌 수 없습니다")

    # 나머지를 복사 (수정할 것이므로)
    remainder = list(a.coeffs)
    divisor = b.coeffs
    deg_b = len(divisor) - 1
    deg_a = len(remainder) - 1

    if deg_a < deg_b:
        return Polynomial.zero(), Polynomial(remainder)

    # 몫 계수 (최고차부터 계산)
    quotient = [FR(0)] * (deg_a - deg_b + 1)
    lead_inv = FR(1) / divisor[-1]  # 제수 최고차 계수의 역원

    for i in range(deg_a - deg_b, -1, -1):
        if len(remainder) - 1 < i + deg_b:
            continue
        coeff = remainder[i + deg_b] * lead_inv
        quotient[i] = coeff
        for j in range(deg_b + 1):
            remainder[i + j] = remainder[i + j] - coeff * divisor[j]

    return Polynomial(quotient), Polynomial(remainder)


def lagrange_basis(domain, i):
    """i번째 Lagrange 기저 다항식 L_i(x)를 계수 형태로 반환한다.

    L_i(x) = ∏_{j≠i} (x - d_j) / (d_i - d_j)

    성질: L_i(d_j) = δ_{ij} (크로네커 델타)

    PLONK에서의 사용:
    - 보간(interpolation): 평가값 → 다항식 (IFFT의 대안)
    - L₁(x) 계산: 경계 제약 (z(x)-1)·L₁(x) = 0

    Args:
        domain: FR 원소 리스트 [d₀, d₁, ..., d_{n-1}]
        i: 기저 인덱스

    Returns:
        Polynomial: L_i(x)

    예시:
        >>> domain = [FR(1), FR(2), FR(3)]
        >>> L0 = lagrange_basis(domain, 0)
        >>> L0.evaluate(FR(1))  # FR(1)
        >>> L0.evaluate(FR(2))  # FR(0)
    """
    n = len(domain)
    result = Polynomial([FR(1)])
    denominator = FR(1)

    for j in range(n):
        if j == i:
            continue
        # (x - d_j) 항을 곱한다
        result = result * Polynomial([FR(0) - domain[j], FR(1)])
        denominator = denominator * (domain[i] - domain[j])

    # 분모의 역원을 곱한다
    denom_inv = FR(1) / denominator
    return result * denom_inv
