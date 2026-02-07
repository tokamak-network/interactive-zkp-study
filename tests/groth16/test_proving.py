import pytest
from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from zkp.groth16.proving import build_rpub_enum
from zkp.groth16.poly_utils import FR

g1 = bn128.G1
g2 = bn128.G2
mult = bn128.multiply


class TestProofA:
    def test_not_none(self, full_pipeline_data):
        prf_A = full_pipeline_data["prf_A"]
        assert prf_A is not None

    def test_is_g1_point(self, full_pipeline_data):
        """proof_A는 G1 위의 점 (2-tuple of FQ)"""
        prf_A = full_pipeline_data["prf_A"]
        assert isinstance(prf_A, tuple)
        assert len(prf_A) == 2


class TestProofB:
    def test_not_none(self, full_pipeline_data):
        prf_B = full_pipeline_data["prf_B"]
        assert prf_B is not None

    def test_is_g2_point(self, full_pipeline_data):
        """proof_B는 G2 위의 점 (2-tuple of FQ2)"""
        prf_B = full_pipeline_data["prf_B"]
        assert isinstance(prf_B, tuple)
        assert len(prf_B) == 2


class TestProofC:
    def test_not_none(self, full_pipeline_data):
        prf_C = full_pipeline_data["prf_C"]
        assert prf_C is not None

    def test_is_g1_point(self, full_pipeline_data):
        """proof_C는 G1 위의 점"""
        prf_C = full_pipeline_data["prf_C"]
        assert isinstance(prf_C, tuple)
        assert len(prf_C) == 2


class TestBuildRpubEnum:
    def test_basic(self):
        r_vec = [FR(1), FR(3), FR(35), FR(9), FR(27), FR(30)]
        pub_indexs = [0, 1]
        result = build_rpub_enum(pub_indexs, r_vec)
        assert len(result) == 2
        assert result[0] == (0, FR(1))
        assert result[1] == (1, FR(3))

    def test_different_indices(self):
        r_vec = [FR(10), FR(20), FR(30)]
        result = build_rpub_enum([0, 2], r_vec)
        assert result[0] == (0, FR(10))
        assert result[1] == (2, FR(30))
