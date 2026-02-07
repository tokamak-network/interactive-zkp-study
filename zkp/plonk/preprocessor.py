"""
PLONK 전처리기 (Preprocessor)
===============================

회로 구조를 분석하여 Verifier가 사용할 공개 파라미터를 생성한다.

**전처리란?**
  PLONK의 "범용 설정(universal setup)"에서, 회로 구조가 정해지면
  셀렉터 다항식과 순열 다항식을 한 번만 계산하고 커밋한다.
  이 커밋먼트는 증명 생성/검증 시 재사용된다.

**전처리 출력물**:
  - 셀렉터 커밋먼트: [q_L]₁, [q_R]₁, [q_O]₁, [q_M]₁, [q_C]₁
  - 순열 커밋먼트: [S_σ1]₁, [S_σ2]₁, [S_σ3]₁
  - 도메인 정보: n, ω (단위근)
  - 셀렉터/순열 다항식 자체 (Prover용)

**Prover vs Verifier 사용**:
  - Prover: 다항식 원본이 필요 (연산에 사용)
  - Verifier: 커밋먼트만 필요 (페어링 검증에 사용)

사용 예시:
    >>> preprocessed = preprocess(circuit, srs)
    >>> preprocessed.q_l_comm  # [q_L]₁ 커밋먼트
"""

from zkp.plonk.field import FR, get_root_of_unity, get_roots_of_unity
from zkp.plonk.polynomial import Polynomial
from zkp.plonk.kzg import commit
from zkp.plonk.permutation import build_permutation_polynomials
from zkp.plonk.utils import next_power_of_2


class PreprocessedData:
    """전처리된 회로 데이터.

    Prover와 Verifier가 공유하는 공개 파라미터를 담는다.

    속성 (도메인):
        n: 도메인 크기 (2의 거듭제곱, ≥ 게이트 수)
        omega: n차 원시 단위근
        domain: [1, ω, ω², ..., ω^{n-1}]

    속성 (셀렉터 다항식 + 커밋먼트):
        q_l_poly, q_r_poly, q_o_poly, q_m_poly, q_c_poly: Polynomial
        q_l_comm, q_r_comm, q_o_comm, q_m_comm, q_c_comm: G1 점

    속성 (순열 다항식 + 커밋먼트):
        s_sigma1_poly, s_sigma2_poly, s_sigma3_poly: Polynomial
        s_sigma1_comm, s_sigma2_comm, s_sigma3_comm: G1 점

    속성 (회로 정보):
        sigma: 순열 배열 (길이 3n)
        num_public_inputs: 공개 입력 수
    """
    pass


def preprocess(circuit, srs):
    """회로를 전처리하여 공개 파라미터를 생성한다.

    단계:
    1. 도메인 설정: 게이트 수 → 2의 거듭제곱 n, 단위근 ω
    2. 셀렉터 다항식: 각 셀렉터 벡터를 IFFT로 다항식화 + KZG 커밋
    3. 순열 다항식: copy constraint에서 순열 생성 → 다항식화 + KZG 커밋

    Args:
        circuit: Circuit 객체
        srs: SRS (Structured Reference String)

    Returns:
        PreprocessedData: 전처리된 데이터

    예시 (x³+x+5=35 회로):
        >>> circuit, a, b, c, pub = Circuit.x3_plus_x_plus_5_eq_35()
        >>> srs = SRS.generate(max_degree=20, seed=1234)
        >>> pp = preprocess(circuit, srs)
        >>> pp.n  # 4 (게이트 수)
    """
    result = PreprocessedData()

    # ── 1단계: 도메인 설정 ──
    n = next_power_of_2(circuit.n)
    # 게이트 수가 이미 2의 거듭제곱이 아니면 패딩
    while len(circuit.gates) < n:
        # 더미 게이트 추가 (모든 셀렉터 = 0, 제약 자동 만족)
        from zkp.plonk.circuit import Gate
        circuit.gates.append(Gate(FR(0), FR(0), FR(0), FR(0), FR(0)))

    result.n = n
    result.omega = get_root_of_unity(n)
    result.domain = get_roots_of_unity(n)

    # ── 2단계: 셀렉터 다항식 ──
    q_l_evals, q_r_evals, q_o_evals, q_m_evals, q_c_evals = (
        circuit.get_selector_polynomials()
    )

    # 평가값 → 다항식 (IFFT)
    result.q_l_poly = Polynomial.from_evaluations(q_l_evals, result.omega)
    result.q_r_poly = Polynomial.from_evaluations(q_r_evals, result.omega)
    result.q_o_poly = Polynomial.from_evaluations(q_o_evals, result.omega)
    result.q_m_poly = Polynomial.from_evaluations(q_m_evals, result.omega)
    result.q_c_poly = Polynomial.from_evaluations(q_c_evals, result.omega)

    # KZG 커밋
    result.q_l_comm = commit(result.q_l_poly, srs)
    result.q_r_comm = commit(result.q_r_poly, srs)
    result.q_o_comm = commit(result.q_o_poly, srs)
    result.q_m_comm = commit(result.q_m_poly, srs)
    result.q_c_comm = commit(result.q_c_poly, srs)

    # ── 3단계: 순열 다항식 ──
    result.sigma = circuit.build_copy_constraints()
    s1_evals, s2_evals, s3_evals = build_permutation_polynomials(
        result.sigma, n, result.domain
    )

    result.s_sigma1_poly = Polynomial.from_evaluations(s1_evals, result.omega)
    result.s_sigma2_poly = Polynomial.from_evaluations(s2_evals, result.omega)
    result.s_sigma3_poly = Polynomial.from_evaluations(s3_evals, result.omega)

    result.s_sigma1_comm = commit(result.s_sigma1_poly, srs)
    result.s_sigma2_comm = commit(result.s_sigma2_poly, srs)
    result.s_sigma3_comm = commit(result.s_sigma3_poly, srs)

    # ── 회로 정보 ──
    result.num_public_inputs = circuit.num_public_inputs

    return result
