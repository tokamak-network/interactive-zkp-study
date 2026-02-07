"""
PLONK Prover — 5-라운드 프로토콜 오케스트레이터
=================================================

PLONK 증명 생성의 전체 흐름을 관리한다.

**5-라운드 구조**:
  PLONK Prover는 5개의 라운드로 구성되며, 각 라운드에서
  커밋먼트를 생성하고 Fiat-Shamir로 챌린지를 받는다.

  ┌─────────────────────────────────────────────────────┐
  │  Round 1: 배선(witness) 다항식 커밋                  │
  │  Prover → Verifier: [a]₁, [b]₁, [c]₁              │
  ├─────────────────────────────────────────────────────┤
  │  Round 2: 순열 누적자 z(x) 커밋                     │
  │  Verifier → Prover: β, γ  (Fiat-Shamir)           │
  │  Prover → Verifier: [z]₁                           │
  ├─────────────────────────────────────────────────────┤
  │  Round 3: 몫 다항식 t(x) 커밋                       │
  │  Verifier → Prover: α  (Fiat-Shamir)               │
  │  Prover → Verifier: [t_lo]₁, [t_mid]₁, [t_hi]₁    │
  ├─────────────────────────────────────────────────────┤
  │  Round 4: 다항식 평가값 산출                         │
  │  Verifier → Prover: ζ  (Fiat-Shamir)               │
  │  Prover → Verifier: ā, b̄, c̄, s̄_σ1, s̄_σ2, z̄_ω    │
  ├─────────────────────────────────────────────────────┤
  │  Round 5: 선형화 + KZG 열기 증명                     │
  │  Verifier → Prover: v  (Fiat-Shamir)               │
  │  Prover → Verifier: [W_ζ]₁, [W_ζω]₁               │
  └─────────────────────────────────────────────────────┘

사용 예시:
    >>> from zkp.plonk.prover import prove
    >>> proof = prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)
"""

from zkp.plonk.field import FR
from zkp.plonk.transcript import Transcript
from zkp.plonk.prover import round1, round2, round3, round4, round5


class Proof:
    """PLONK 증명 데이터 컨테이너.

    5개 라운드에서 생성된 모든 증명 요소를 담는다.

    Round 1 (배선 커밋먼트):
        a_comm, b_comm, c_comm: G1 점

    Round 2 (순열 누적자 커밋먼트):
        z_comm: G1 점

    Round 3 (몫 다항식 커밋먼트):
        t_lo_comm, t_mid_comm, t_hi_comm: G1 점

    Round 4 (평가값):
        a_eval, b_eval, c_eval: FR (배선 다항식 평가)
        s_sigma1_eval, s_sigma2_eval: FR (순열 다항식 평가)
        z_omega_eval: FR (z(ζ·ω) — 다음 도메인 점에서의 z값)

    Round 5 (선형화 + 열기 증명):
        r_eval: FR (선형화 다항식 r(ζ) 평가값)
        W_zeta_comm: G1 점 (ζ에서의 일괄 열기 증명)
        W_zeta_omega_comm: G1 점 (ζ·ω에서의 열기 증명)
    """

    def __init__(self):
        # Round 1
        self.a_comm = None
        self.b_comm = None
        self.c_comm = None
        # Round 2
        self.z_comm = None
        # Round 3
        self.t_lo_comm = None
        self.t_mid_comm = None
        self.t_hi_comm = None
        # Round 4
        self.a_eval = None
        self.b_eval = None
        self.c_eval = None
        self.s_sigma1_eval = None
        self.s_sigma2_eval = None
        self.z_omega_eval = None
        # Round 5
        self.r_eval = None
        self.W_zeta_comm = None
        self.W_zeta_omega_comm = None


class ProverState:
    """라운드 간 공유되는 Prover 상태.

    각 라운드 함수는 이 객체를 읽고 결과를 기록한다.
    라운드 간 데이터 의존성을 명확히 관리하기 위한 중앙 저장소.

    속성 (입력):
        a_vals, b_vals, c_vals: 배선 값 리스트 (길이 n)
        public_inputs: 공개 입력 값 리스트
        preprocessed: PreprocessedData
        srs: SRS
        transcript: Fiat-Shamir 트랜스크립트

    속성 (라운드 간 생성):
        a_poly, b_poly, c_poly: 배선 다항식 (Round 1)
        z_poly: 순열 누적자 다항식 (Round 2)
        t_lo_poly, t_mid_poly, t_hi_poly: 몫 다항식 (Round 3)
        beta, gamma, alpha, zeta, v: 챌린지 값들

    속성 (출력):
        proof: Proof 객체
    """

    def __init__(self, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs):
        # 입력
        self.a_vals = a_vals
        self.b_vals = b_vals
        self.c_vals = c_vals
        self.public_inputs = public_inputs
        self.preprocessed = preprocessed
        self.srs = srs

        # Fiat-Shamir 트랜스크립트
        self.transcript = Transcript()

        # 도메인 정보 (편의 접근)
        self.n = preprocessed.n
        self.omega = preprocessed.omega
        self.domain = preprocessed.domain

        # 라운드별 결과 (각 라운드에서 채워짐)
        self.a_poly = None
        self.b_poly = None
        self.c_poly = None
        self.z_poly = None
        self.t_lo_poly = None
        self.t_mid_poly = None
        self.t_hi_poly = None

        # 챌린지
        self.beta = None
        self.gamma = None
        self.alpha = None
        self.zeta = None
        self.v = None

        # 공개 입력 다항식
        self.pi_poly = None

        # 증명 출력
        self.proof = Proof()

    def build_proof(self):
        """최종 증명 객체를 반환한다."""
        return self.proof


def prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs):
    """PLONK 5-라운드 프로토콜을 실행하여 증명을 생성한다.

    Args:
        circuit: Circuit 객체 (copy constraint 정보 포함)
        a_vals: 왼쪽 배선 값 리스트 [a₀, a₁, ..., a_{n-1}]
        b_vals: 오른쪽 배선 값 리스트
        c_vals: 출력 배선 값 리스트
        public_inputs: 공개 입력 값 리스트
        preprocessed: PreprocessedData (전처리 결과)
        srs: SRS

    Returns:
        Proof: PLONK 증명

    예시 (x³+x+5=35):
        >>> circuit, a, b, c, pub = Circuit.x3_plus_x_plus_5_eq_35()
        >>> srs = SRS.generate(20, seed=42)
        >>> pp = preprocess(circuit, srs)
        >>> proof = prove(circuit, a, b, c, pub, pp, srs)
    """
    state = ProverState(a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)

    # ┌─────────────────────────────────────────────────────┐
    # │  Round 1: 배선(witness) 다항식 커밋                  │
    # │  a(x), b(x), c(x) → [a]₁, [b]₁, [c]₁             │
    # └─────────────────────────────────────────────────────┘
    round1.execute(state)

    # ┌─────────────────────────────────────────────────────┐
    # │  Round 2: 순열 누적자 z(x) 커밋                     │
    # │  β, γ 챌린지 → z(x) 계산 → [z]₁                    │
    # └─────────────────────────────────────────────────────┘
    round2.execute(state)

    # ┌─────────────────────────────────────────────────────┐
    # │  Round 3: 몫 다항식 t(x) 커밋                       │
    # │  α 챌린지 → t(x) = C(x)/Z_H(x) → 3분할 커밋       │
    # └─────────────────────────────────────────────────────┘
    round3.execute(state)

    # ┌─────────────────────────────────────────────────────┐
    # │  Round 4: 다항식 평가값 산출                         │
    # │  ζ 챌린지 → ā, b̄, c̄, s̄_σ1, s̄_σ2, z̄_ω 계산      │
    # └─────────────────────────────────────────────────────┘
    round4.execute(state)

    # ┌─────────────────────────────────────────────────────┐
    # │  Round 5: 선형화 + KZG 열기 증명                     │
    # │  v 챌린지 → [W_ζ]₁, [W_ζω]₁                       │
    # └─────────────────────────────────────────────────────┘
    round5.execute(state)

    return state.build_proof()
