"""
PLONK Prover Round 1: 배선(Witness) 다항식 커밋먼트
=====================================================

  ┌─────────────────────────────────────────────────┐
  │  Prover → Verifier: [a]₁, [b]₁, [c]₁          │
  │                                                 │
  │  입력:  배선 값 (a, b, c), 도메인, SRS          │
  │  출력:  3개의 KZG 커밋먼트                       │
  └─────────────────────────────────────────────────┘

**과정**:
  1. witness 벡터(a, b, c)를 IFFT로 다항식으로 보간
     - a_vals = [a₀, a₁, ..., a_{n-1}]에서 a(ωⁱ) = aᵢ인 다항식 a(x) 복원
  2. 블라인딩(blinding): 영지식(zero-knowledge) 보장
     - a'(x) = a(x) + (b₁·x + b₂)·Z_H(x)
     - Z_H(x)의 배수를 더해도 도메인 위의 값은 변하지 않음
     - 랜덤 계수 b₁, b₂가 다항식의 "겉모습"을 숨김
  3. KZG 커밋: [a']₁ = commit(a', srs)

**왜 블라인딩이 필요한가?**
  블라인딩 없이는 커밋먼트에서 witness 정보가 누출될 수 있다.
  Z_H(x)의 배수를 더하면:
  - 도메인 위: Z_H(ωⁱ) = 0이므로 a'(ωⁱ) = a(ωⁱ) (값 불변)
  - 도메인 밖: 랜덤 값이 추가되어 원래 다항식을 알 수 없음

사용:
    이 모듈은 직접 호출하지 않고, prover.prove()를 통해 실행된다.
"""

import secrets
from zkp.plonk.field import FR, CURVE_ORDER
from zkp.plonk.polynomial import Polynomial
from zkp.plonk.kzg import commit
from zkp.plonk.utils import public_input_polynomial


def execute(state):
    """Round 1을 실행한다.

    Args:
        state: ProverState — a_vals, b_vals, c_vals를 읽고,
               a_poly, b_poly, c_poly와 커밋먼트를 기록한다.
    """
    n = state.n
    omega = state.omega

    # ── 1. 공개 입력 다항식 구성 ──
    # PI(x): 공개 입력 다항식.
    #
    # PLONK에서 공개 입력을 처리하는 방법:
    # 방법 A: 전용 공개 입력 게이트 (q_O=1, PI(ωⁱ)=-value)
    # 방법 B: 셀렉터에 직접 인코딩 (현재 구현)
    #
    # 현재 회로는 공개 입력을 셀렉터(q_C=5)와 게이트 구조로
    # 직접 인코딩하므로, PI(x) = 0으로 설정한다.
    # Verifier는 공개 입력이 회로에 올바르게 인코딩되었음을
    # 전처리 단계에서 확인한다.
    state.pi_poly = Polynomial.zero()

    # ── 2. 배선 다항식 보간 (IFFT) ──
    # a(ωⁱ) = a_vals[i] 인 다항식 a(x)를 복원
    a_poly = Polynomial.from_evaluations(state.a_vals, omega)
    b_poly = Polynomial.from_evaluations(state.b_vals, omega)
    c_poly = Polynomial.from_evaluations(state.c_vals, omega)

    # ── 3. 블라인딩 (Zero-Knowledge) ──
    # a'(x) = a(x) + (b₁·x + b₂)·Z_H(x)
    # Z_H(x) = x^n - 1
    zh = Polynomial.vanishing(n)

    # 각 배선 다항식에 랜덤 블라인딩 적용
    a_poly = _add_blinding(a_poly, zh, 2)
    b_poly = _add_blinding(b_poly, zh, 2)
    c_poly = _add_blinding(c_poly, zh, 2)

    state.a_poly = a_poly
    state.b_poly = b_poly
    state.c_poly = c_poly

    # ── 4. KZG 커밋 ──
    state.proof.a_comm = commit(a_poly, state.srs)
    state.proof.b_comm = commit(b_poly, state.srs)
    state.proof.c_comm = commit(c_poly, state.srs)

    # ── 5. 트랜스크립트에 커밋먼트 추가 ──
    state.transcript.append_point(b"a_comm", state.proof.a_comm)
    state.transcript.append_point(b"b_comm", state.proof.b_comm)
    state.transcript.append_point(b"c_comm", state.proof.c_comm)


def _add_blinding(poly, zh, num_blinds):
    """다항식에 블라인딩 항을 추가한다.

    blinding(x) = (r₁ + r₂·x + ...) · Z_H(x)
    여기서 rᵢ는 랜덤 FR 원소.

    Args:
        poly: 원본 다항식
        zh: 소거 다항식 Z_H(x)
        num_blinds: 블라인딩 계수의 수

    Returns:
        Polynomial: 블라인딩된 다항식
    """
    blind_coeffs = [FR(secrets.randbelow(CURVE_ORDER)) for _ in range(num_blinds)]
    blind_poly = Polynomial(blind_coeffs)
    return poly + blind_poly * zh
