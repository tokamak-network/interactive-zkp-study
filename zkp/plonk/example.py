"""
PLONK E2E 데모: x³ + x + 5 = 35 (x = 3)
============================================

이 스크립트는 PLONK 프로토콜의 전체 흐름을 시연한다.

실행:
    python -m zkp.plonk.example

흐름:
    1. 회로 구성 (x³ + x + 5 = 35)
    2. SRS 생성 (trusted setup)
    3. 회로 전처리
    4. 증명 생성 (5-라운드)
    5. 증명 검증
"""

from zkp.plonk.field import FR
from zkp.plonk.circuit import Circuit
from zkp.plonk.srs import SRS
from zkp.plonk.preprocessor import preprocess
from zkp.plonk.prover import prove
from zkp.plonk.verifier import verify


def main():
    print("=" * 60)
    print("  PLONK Zero-Knowledge Proof Demo")
    print("  회로: x³ + x + 5 = 35 (x = 3)")
    print("=" * 60)

    # ── 1. 회로 구성 ──
    print("\n[1] 회로 구성...")
    circuit, a_vals, b_vals, c_vals, public_inputs = (
        Circuit.x3_plus_x_plus_5_eq_35()
    )
    print(f"    게이트 수: {circuit.n}")
    print(f"    배선 복사 제약 수: {len(circuit.copy_constraints)}")
    print(f"    공개 입력: {[int(p) for p in public_inputs]}")

    # 게이트 제약 확인
    print("\n    게이트 제약 확인:")
    for i, gate in enumerate(circuit.gates):
        ok = gate.check(a_vals[i], b_vals[i], c_vals[i])
        print(f"      게이트 {i}: a={int(a_vals[i])}, b={int(b_vals[i])}, "
              f"c={int(c_vals[i])} → {'✓' if ok else '✗'}")

    # ── 2. SRS 생성 ──
    print("\n[2] SRS 생성 (trusted setup)...")
    # 최대 차수: 3n + 몇 개 여유 (블라인딩 등)
    max_degree = 3 * circuit.n + 10
    srs = SRS.generate(max_degree=max_degree, seed=12345)
    print(f"    최대 다항식 차수: {max_degree}")
    print(f"    G1 powers 수: {len(srs.g1_powers)}")

    # ── 3. 전처리 ──
    print("\n[3] 회로 전처리...")
    preprocessed = preprocess(circuit, srs)
    print(f"    도메인 크기 n: {preprocessed.n}")
    print(f"    단위근 ω: FR({int(preprocessed.omega)})")

    # ── 4. 증명 생성 ──
    print("\n[4] 증명 생성 (5-라운드)...")
    proof = prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)
    print("    Round 1: [a]₁, [b]₁, [c]₁ 커밋")
    print("    Round 2: [z]₁ 커밋 (순열 누적자)")
    print("    Round 3: [t_lo]₁, [t_mid]₁, [t_hi]₁ 커밋")
    print("    Round 4: 평가값 산출")
    print(f"      ā = {int(proof.a_eval)}")
    print(f"      b̄ = {int(proof.b_eval)}")
    print(f"      c̄ = {int(proof.c_eval)}")
    print("    Round 5: [W_ζ]₁, [W_ζω]₁ 열기 증명")

    # ── 5. 검증 ──
    print("\n[5] 증명 검증...")
    result = verify(proof, public_inputs, preprocessed, srs)
    print(f"    검증 결과: {'성공 ✓' if result else '실패 ✗'}")

    # ── 6. 조작된 증명 테스트 ──
    # r_eval을 조작하면 페어링 검사에서 실패해야 한다.
    print("\n[6] 조작된 증명으로 검증 (r_eval 변조)...")
    from copy import copy
    fake_proof = copy(proof)
    fake_proof.r_eval = proof.r_eval + FR(1)  # 1을 더해 조작
    wrong_result = verify(fake_proof, public_inputs, preprocessed, srs)
    print(f"    검증 결과: {'성공 ✓' if wrong_result else '실패 ✗ (예상대로 실패)'}")

    print("\n" + "=" * 60)
    if result and not wrong_result:
        print("  데모 완료: 모든 테스트 통과!")
    else:
        print("  데모 완료: 일부 테스트 실패")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
