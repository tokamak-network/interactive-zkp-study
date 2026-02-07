import pytest
from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from zkp.groth16.poly_utils import (
    FR, _multiply_vec_vec, _eval_poly, _multiply_polys, _subtract_polys,
)
from zkp.groth16.verifying import verify


class TestR1csSatisfaction:
    """R1CS 제약조건 만족성: A*r . B*r == C*r"""

    def test_all_gates(self, r1cs_data):
        r = r1cs_data["r"]
        A, B, C = r1cs_data["A"], r1cs_data["B"], r1cs_data["C"]
        for gate_idx in range(len(A)):
            a_dot = sum(A[gate_idx][j] * r[j] for j in range(len(r)))
            b_dot = sum(B[gate_idx][j] * r[j] for j in range(len(r)))
            c_dot = sum(C[gate_idx][j] * r[j] for j in range(len(r)))
            assert a_dot * b_dot == c_dot, f"Gate {gate_idx} unsatisfied"


class TestQapCancellation:
    """QAP 다항식이 게이트 점에서 소거됨"""

    def test_sol_vanishes_at_gates(self, qap_data):
        from zkp.groth16.qap_creator_lcm import eval_poly
        sol = qap_data["sol"]
        num_gates = len(qap_data["Ap"][0])  # 게이트 수
        for x in range(1, num_gates + 1):
            val = eval_poly(sol, x)
            assert abs(val) < 1e-4, f"sol({x}) = {val} != 0"


class TestPolynomialIdentity:
    """H*Z == A(x)*B(x) - C(x) (FR 필드에서)"""

    def test_hxr_remainder_zero(self, full_pipeline_data):
        remainder = full_pipeline_data["remainder"]
        for val in remainder:
            assert val == FR(0), f"Non-zero remainder: {val}"

    def test_polynomial_evaluation_identity(self, full_pipeline_data):
        """R.Ax * R.Bx - R.Cx == Hx * Zx (x_val에서)"""
        d = full_pipeline_data
        Rx = d["Rx"]
        Ax_val = d["Ax_val"]
        Bx_val = d["Bx_val"]
        Cx_val = d["Cx_val"]
        Hx_val = d["Hx_val"]
        Zx_val = d["Zx_val"]

        lhs_val = _multiply_vec_vec(Rx, Ax_val) * _multiply_vec_vec(Rx, Bx_val) - _multiply_vec_vec(Rx, Cx_val)
        rhs_val = Zx_val * Hx_val
        assert lhs_val == rhs_val


class TestE2EVerification:
    """전체 파이프라인 E2E 검증"""

    def test_verify_succeeds(self, full_pipeline_data):
        d = full_pipeline_data
        result = verify(
            d["prf_A"], d["prf_B"], d["prf_C"],
            d["s11"], d["s13"], d["s21"], d["rx_pub"]
        )
        assert result is True

    def test_wrong_input_fails(self, full_pipeline_data):
        """올바른 setup으로 잘못된 proof를 만들면 검증 실패"""
        from py_ecc import bn128 as bn
        d = full_pipeline_data
        fake_A = bn.multiply(bn.G1, 42)
        result = verify(
            fake_A, d["prf_B"], d["prf_C"],
            d["s11"], d["s13"], d["s21"], d["rx_pub"]
        )
        assert result is False
