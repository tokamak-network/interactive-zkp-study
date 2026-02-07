"""
PLONK Verifier & End-to-End Integration Tests
===============================================

Verifier 검증 로직과 전체 파이프라인(circuit -> SRS -> preprocess -> prove -> verify)을
테스트한다.

테스트 범위:
  - E2E 파이프라인: x^3 + x + 5 = 35 (x=3)
  - 단순 덧셈 회로
  - 단순 곱셈 회로
  - 건전성(soundness): 조작된 증명 요소 검증 실패
  - 잘못된 공개 입력으로 검증 (PI가 q_C에 인코딩된 현재 동작 문서화)
"""

import copy
import random
import pytest

from zkp.plonk.field import FR, G1, ec_mul, CURVE_ORDER
from zkp.plonk.circuit import Circuit, Gate
from zkp.plonk.srs import SRS
from zkp.plonk.preprocessor import preprocess
from zkp.plonk.prover import prove
from zkp.plonk.verifier import verify


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def x3_circuit_data():
    """x^3 + x + 5 = 35 (x=3) 회로의 전체 파이프라인 데이터를 반환한다."""
    circuit, a_vals, b_vals, c_vals, public_inputs = (
        Circuit.x3_plus_x_plus_5_eq_35()
    )
    max_degree = 3 * circuit.n + 10
    srs = SRS.generate(max_degree=max_degree, seed=12345)
    preprocessed = preprocess(circuit, srs)
    proof = prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)
    return {
        "circuit": circuit,
        "a_vals": a_vals,
        "b_vals": b_vals,
        "c_vals": c_vals,
        "public_inputs": public_inputs,
        "srs": srs,
        "preprocessed": preprocessed,
        "proof": proof,
    }


def _build_addition_circuit():
    """단순 덧셈 회로: a + b = c (3 + 7 = 10).

    게이트 0 (add): a=3, b=7, c=10
    """
    circuit = Circuit()
    circuit.add_addition_gate()

    a_vals = [FR(3)]
    b_vals = [FR(7)]
    c_vals = [FR(10)]
    public_inputs = []

    return circuit, a_vals, b_vals, c_vals, public_inputs


def _build_multiplication_circuit():
    """단순 곱셈 회로: a * b = c (4 * 5 = 20).

    게이트 0 (mul): a=4, b=5, c=20
    """
    circuit = Circuit()
    circuit.add_multiplication_gate()

    a_vals = [FR(4)]
    b_vals = [FR(5)]
    c_vals = [FR(20)]
    public_inputs = []

    return circuit, a_vals, b_vals, c_vals, public_inputs


@pytest.fixture(scope="module")
def addition_circuit_data():
    """덧셈 회로 파이프라인 데이터."""
    circuit, a_vals, b_vals, c_vals, public_inputs = _build_addition_circuit()
    max_degree = 3 * circuit.n + 10
    srs = SRS.generate(max_degree=max_degree, seed=9999)
    preprocessed = preprocess(circuit, srs)
    proof = prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)
    return {
        "circuit": circuit,
        "a_vals": a_vals,
        "b_vals": b_vals,
        "c_vals": c_vals,
        "public_inputs": public_inputs,
        "srs": srs,
        "preprocessed": preprocessed,
        "proof": proof,
    }


@pytest.fixture(scope="module")
def multiplication_circuit_data():
    """곱셈 회로 파이프라인 데이터."""
    circuit, a_vals, b_vals, c_vals, public_inputs = _build_multiplication_circuit()
    max_degree = 3 * circuit.n + 10
    srs = SRS.generate(max_degree=max_degree, seed=7777)
    preprocessed = preprocess(circuit, srs)
    proof = prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)
    return {
        "circuit": circuit,
        "a_vals": a_vals,
        "b_vals": b_vals,
        "c_vals": c_vals,
        "public_inputs": public_inputs,
        "srs": srs,
        "preprocessed": preprocessed,
        "proof": proof,
    }


# ─────────────────────────────────────────────────────────────────────
# E2E Pipeline Tests
# ─────────────────────────────────────────────────────────────────────

class TestE2EPipeline:
    """전체 파이프라인 E2E 테스트."""

    def test_x3_plus_x_plus_5_eq_35_passes(self, x3_circuit_data):
        """x^3 + x + 5 = 35 (x=3) 회로의 증명이 검증을 통과해야 한다."""
        d = x3_circuit_data
        result = verify(d["proof"], d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is True

    def test_addition_circuit_passes(self, addition_circuit_data):
        """단순 덧셈 회로 (3 + 7 = 10) 증명이 검증을 통과해야 한다."""
        d = addition_circuit_data
        result = verify(d["proof"], d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is True

    def test_multiplication_circuit_passes(self, multiplication_circuit_data):
        """단순 곱셈 회로 (4 * 5 = 20) 증명이 검증을 통과해야 한다."""
        d = multiplication_circuit_data
        result = verify(d["proof"], d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is True

    def test_proof_has_all_fields(self, x3_circuit_data):
        """Proof 객체에 모든 필수 필드가 존재해야 한다."""
        proof = x3_circuit_data["proof"]
        # Round 1 commitments
        assert proof.a_comm is not None
        assert proof.b_comm is not None
        assert proof.c_comm is not None
        # Round 2 commitment
        assert proof.z_comm is not None
        # Round 3 commitments
        assert proof.t_lo_comm is not None
        assert proof.t_mid_comm is not None
        assert proof.t_hi_comm is not None
        # Round 4 evaluations
        assert proof.a_eval is not None
        assert proof.b_eval is not None
        assert proof.c_eval is not None
        assert proof.s_sigma1_eval is not None
        assert proof.s_sigma2_eval is not None
        assert proof.z_omega_eval is not None
        # Round 5
        assert proof.r_eval is not None
        assert proof.W_zeta_comm is not None
        assert proof.W_zeta_omega_comm is not None

    def test_verify_is_deterministic(self, x3_circuit_data):
        """동일한 증명으로 검증을 여러 번 실행해도 같은 결과여야 한다."""
        d = x3_circuit_data
        r1 = verify(d["proof"], d["public_inputs"], d["preprocessed"], d["srs"])
        r2 = verify(d["proof"], d["public_inputs"], d["preprocessed"], d["srs"])
        assert r1 == r2 == True

    def test_different_srs_seed_still_works(self):
        """다른 SRS seed로도 동일 회로의 증명이 통과해야 한다."""
        circuit, a_vals, b_vals, c_vals, public_inputs = (
            Circuit.x3_plus_x_plus_5_eq_35()
        )
        srs = SRS.generate(max_degree=3 * circuit.n + 10, seed=99999)
        preprocessed = preprocess(circuit, srs)
        proof = prove(circuit, a_vals, b_vals, c_vals, public_inputs, preprocessed, srs)
        assert verify(proof, public_inputs, preprocessed, srs) is True


# ─────────────────────────────────────────────────────────────────────
# Soundness: Tampered Scalar Evaluations
# ─────────────────────────────────────────────────────────────────────

class TestSoundnessScalarTampering:
    """스칼라 평가값을 조작하면 검증이 실패해야 한다."""

    @pytest.mark.parametrize("field_name", [
        "a_eval",
        "b_eval",
        "c_eval",
        "s_sigma1_eval",
        "s_sigma2_eval",
        "z_omega_eval",
        "r_eval",
    ])
    def test_tampered_scalar_fails(self, x3_circuit_data, field_name):
        """증명의 스칼라 평가값을 변조하면 검증 실패해야 한다."""
        d = x3_circuit_data
        tampered = copy.deepcopy(d["proof"])
        original_val = getattr(tampered, field_name)
        setattr(tampered, field_name, original_val + FR(1))
        result = verify(tampered, d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is False, f"{field_name} 변조 후에도 검증이 통과했습니다"


# ─────────────────────────────────────────────────────────────────────
# Soundness: Tampered Commitments
# ─────────────────────────────────────────────────────────────────────

def _random_g1_point():
    """무작위 G1 점을 생성한다."""
    return ec_mul(G1, FR(random.randint(1, CURVE_ORDER - 1)))


class TestSoundnessCommitmentTampering:
    """커밋먼트(G1 점)를 조작하면 검증이 실패해야 한다."""

    @pytest.mark.parametrize("field_name", [
        "a_comm",
        "b_comm",
        "c_comm",
        "z_comm",
        "t_lo_comm",
        "t_mid_comm",
        "t_hi_comm",
        "W_zeta_comm",
        "W_zeta_omega_comm",
    ])
    def test_tampered_commitment_fails(self, x3_circuit_data, field_name):
        """증명의 커밋먼트를 변조하면 검증 실패해야 한다."""
        d = x3_circuit_data
        tampered = copy.deepcopy(d["proof"])
        fake_point = _random_g1_point()
        setattr(tampered, field_name, fake_point)
        result = verify(tampered, d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is False, f"{field_name} 변조 후에도 검증이 통과했습니다"


# ─────────────────────────────────────────────────────────────────────
# Public Input Handling
# ─────────────────────────────────────────────────────────────────────

class TestPublicInputHandling:
    """공개 입력(PI) 관련 동작 테스트.

    현재 구현에서 공개 입력은 q_C 셀렉터에 직접 인코딩되어 있으므로
    PI(x) = 0이다. 따라서 verify()의 public_inputs 파라미터를
    변경해도 검증 결과에 영향이 없다.
    이 동작을 문서화한다.
    """

    def test_wrong_public_inputs_still_passes_due_to_qc_encoding(
        self, x3_circuit_data
    ):
        """잘못된 공개 입력으로도 검증이 통과한다 (PI가 q_C에 인코딩되므로).

        현재 구현에서는 public_inputs 파라미터가 검증에 영향을 미치지 않는다.
        PI(x) = 0으로 고정되어 있기 때문이다.
        """
        d = x3_circuit_data
        wrong_public_inputs = [FR(999)]
        result = verify(d["proof"], wrong_public_inputs, d["preprocessed"], d["srs"])
        assert result is True, (
            "PI가 q_C에 인코딩된 현재 구현에서는 "
            "public_inputs 값과 무관하게 검증이 통과해야 합니다"
        )

    def test_empty_public_inputs_still_passes(self, x3_circuit_data):
        """빈 공개 입력 리스트로도 검증이 통과한다."""
        d = x3_circuit_data
        result = verify(d["proof"], [], d["preprocessed"], d["srs"])
        assert result is True


# ─────────────────────────────────────────────────────────────────────
# Soundness: Mismatched SRS / Preprocessed
# ─────────────────────────────────────────────────────────────────────

class TestCrossCircuitSoundness:
    """다른 회로의 전처리 데이터나 SRS를 교차 사용하면 실패해야 한다."""

    def test_proof_with_different_preprocessed_fails(
        self, x3_circuit_data, addition_circuit_data
    ):
        """x^3 회로의 증명을 덧셈 회로의 전처리 데이터로 검증하면 실패해야 한다."""
        result = verify(
            x3_circuit_data["proof"],
            x3_circuit_data["public_inputs"],
            addition_circuit_data["preprocessed"],
            addition_circuit_data["srs"],
        )
        assert result is False

    def test_addition_proof_with_x3_preprocessed_fails(
        self, x3_circuit_data, addition_circuit_data
    ):
        """덧셈 회로의 증명을 x^3 회로의 전처리 데이터로 검증하면 실패해야 한다."""
        result = verify(
            addition_circuit_data["proof"],
            addition_circuit_data["public_inputs"],
            x3_circuit_data["preprocessed"],
            x3_circuit_data["srs"],
        )
        assert result is False


# ─────────────────────────────────────────────────────────────────────
# Gate Constraint Validation
# ─────────────────────────────────────────────────────────────────────

class TestGateConstraints:
    """회로 게이트 제약이 witness에서 올바르게 만족되는지 검증한다."""

    def test_x3_circuit_gates_satisfied(self, x3_circuit_data):
        """x^3+x+5=35 회로의 모든 게이트 제약이 만족되어야 한다."""
        d = x3_circuit_data
        for i, gate in enumerate(d["circuit"].gates):
            assert gate.check(d["a_vals"][i], d["b_vals"][i], d["c_vals"][i]), (
                f"Gate {i} constraint not satisfied"
            )

    def test_addition_circuit_gate_satisfied(self, addition_circuit_data):
        """덧셈 회로의 게이트 제약이 만족되어야 한다."""
        d = addition_circuit_data
        assert d["circuit"].gates[0].check(FR(3), FR(7), FR(10))

    def test_multiplication_circuit_gate_satisfied(self, multiplication_circuit_data):
        """곱셈 회로의 게이트 제약이 만족되어야 한다."""
        d = multiplication_circuit_data
        assert d["circuit"].gates[0].check(FR(4), FR(5), FR(20))


# ─────────────────────────────────────────────────────────────────────
# Multiple Tampering Combinations
# ─────────────────────────────────────────────────────────────────────

class TestMultipleTampering:
    """여러 필드를 동시에 조작해도 실패해야 한다."""

    def test_tamper_a_eval_and_b_eval(self, x3_circuit_data):
        """a_eval과 b_eval을 동시에 조작하면 검증이 실패해야 한다."""
        d = x3_circuit_data
        tampered = copy.deepcopy(d["proof"])
        tampered.a_eval = tampered.a_eval + FR(1)
        tampered.b_eval = tampered.b_eval + FR(1)
        result = verify(tampered, d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is False

    def test_tamper_commitment_and_eval(self, x3_circuit_data):
        """커밋먼트와 평가값을 동시에 조작하면 검증이 실패해야 한다."""
        d = x3_circuit_data
        tampered = copy.deepcopy(d["proof"])
        tampered.a_comm = _random_g1_point()
        tampered.a_eval = tampered.a_eval + FR(1)
        result = verify(tampered, d["public_inputs"], d["preprocessed"], d["srs"])
        assert result is False
