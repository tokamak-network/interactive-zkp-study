import pytest
from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from zkp.groth16.verifying import lhs, rhs, verify

g1 = bn128.G1
g2 = bn128.G2
mult = bn128.multiply
pairing = bn128.pairing


class TestLhs:
    def test_returns_pairing(self, full_pipeline_data):
        prf_A = full_pipeline_data["prf_A"]
        prf_B = full_pipeline_data["prf_B"]
        result = lhs(prf_A, prf_B)
        assert result is not None

    def test_equals_direct_pairing(self, full_pipeline_data):
        prf_A = full_pipeline_data["prf_A"]
        prf_B = full_pipeline_data["prf_B"]
        result = lhs(prf_A, prf_B)
        expected = pairing(prf_B, prf_A)
        assert result == expected


class TestRhs:
    def test_returns_value(self, full_pipeline_data):
        d = full_pipeline_data
        result = rhs(d["prf_C"], d["s11"], d["s13"], d["s21"], d["rx_pub"])
        assert result is not None


class TestVerify:
    def test_valid_proof(self, full_pipeline_data):
        d = full_pipeline_data
        result = verify(
            d["prf_A"], d["prf_B"], d["prf_C"],
            d["s11"], d["s13"], d["s21"], d["rx_pub"]
        )
        assert result is True

    def test_tampered_proof_rejected(self, full_pipeline_data):
        """변조된 proof_A로 검증 실패"""
        d = full_pipeline_data
        fake_A = mult(g1, 9999)  # 잘못된 proof
        result = verify(
            fake_A, d["prf_B"], d["prf_C"],
            d["s11"], d["s13"], d["s21"], d["rx_pub"]
        )
        assert result is False

    def test_tampered_proof_c_rejected(self, full_pipeline_data):
        """변조된 proof_C로 검증 실패"""
        d = full_pipeline_data
        fake_C = mult(g1, 12345)
        result = verify(
            d["prf_A"], d["prf_B"], fake_C,
            d["s11"], d["s13"], d["s21"], d["rx_pub"]
        )
        assert result is False
