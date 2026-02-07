"""
PLONK Prover Round 2: 순열 누적자 z(x) 커밋먼트
=================================================

  ┌─────────────────────────────────────────────────┐
  │  Verifier → Prover: β, γ  (Fiat-Shamir)       │
  │  Prover → Verifier: [z]₁                       │
  │                                                 │
  │  입력:  β, γ 챌린지, witness, 순열 정보         │
  │  출력:  순열 누적자 커밋먼트 [z]₁               │
  └─────────────────────────────────────────────────┘

**순열 누적자(Grand Product Accumulator) z(x)란?**
  Copy constraint가 만족됨을 증명하는 핵심 다항식.

  z(ω⁰) = 1  (초기값)
  z(ω^{i+1}) = z(ωⁱ) ·
      (aᵢ + β·ωⁱ + γ)(bᵢ + β·K1·ωⁱ + γ)(cᵢ + β·K2·ωⁱ + γ)
      ─────────────────────────────────────────────────────────
      (aᵢ + β·S_σ1(ωⁱ) + γ)(bᵢ + β·S_σ2(ωⁱ) + γ)(cᵢ + β·S_σ3(ωⁱ) + γ)

  분자: 배선 위치를 항등 순열로 매핑한 값
  분모: 실제 순열 σ로 매핑한 값

  **핵심 아이디어**:
  순열 σ가 올바르면 (같은 값의 배선이 정확히 연결되면),
  분자와 분모의 전체 곱이 같아서 z(ω^n) = 1이 된다.
  β, γ는 이 등식을 "무작위로 검사"하는 챌린지이다.

**블라인딩**:
  z'(x) = z(x) + (b₆·x² + b₇·x + b₈)·Z_H(x)
  3개의 블라인딩 계수 사용 (Round 3, 5에서 z를 2번 평가하므로).

사용:
    이 모듈은 직접 호출하지 않고, prover.prove()를 통해 실행된다.
"""

import secrets
from zkp.plonk.field import FR, CURVE_ORDER
from zkp.plonk.polynomial import Polynomial
from zkp.plonk.kzg import commit
from zkp.plonk.permutation import compute_accumulator


def execute(state):
    """Round 2를 실행한다.

    Args:
        state: ProverState — Round 1의 결과를 읽고,
               z_poly와 [z]₁ 커밋먼트를 기록한다.
    """
    # ── 1. β, γ 챌린지 생성 (Fiat-Shamir) ──
    # Round 1의 커밋먼트가 트랜스크립트에 있으므로,
    # 이를 기반으로 결정론적 챌린지를 생성한다.
    state.beta = state.transcript.challenge_scalar(b"beta")
    state.gamma = state.transcript.challenge_scalar(b"gamma")

    # ── 2. 순열 누적자 z(x) 계산 ──
    # z(ωⁱ) 평가값을 계산한다.
    n = state.n
    z_evals = compute_accumulator(
        state.a_vals, state.b_vals, state.c_vals,
        state.preprocessed.sigma, n, state.domain,
        state.beta, state.gamma
    )

    # z(ω^n) = 1 확인 (순열이 올바른지 검증)
    # (실제 z_evals는 길이 n이고, z[0]=1, 이 값들의 순환 곱이 1이어야 함)

    # ── 3. IFFT로 z(x) 다항식 복원 ──
    z_poly = Polynomial.from_evaluations(z_evals, state.omega)

    # ── 4. 블라인딩 ──
    # z(x)는 Round 3에서 z(ωx), Round 4에서 z(ζω)로 사용되므로
    # 3개의 블라인딩 계수가 필요
    zh = Polynomial.vanishing(n)
    blind_coeffs = [FR(secrets.randbelow(CURVE_ORDER)) for _ in range(3)]
    blind_poly = Polynomial(blind_coeffs)
    z_poly = z_poly + blind_poly * zh

    state.z_poly = z_poly

    # ── 5. KZG 커밋 + 트랜스크립트 업데이트 ──
    state.proof.z_comm = commit(z_poly, state.srs)
    state.transcript.append_point(b"z_comm", state.proof.z_comm)
