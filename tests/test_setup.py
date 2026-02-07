import pytest
from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

g1 = bn128.G1
g2 = bn128.G2
mult = bn128.multiply


class TestSigma11:
    def test_length(self, full_pipeline_data):
        s11 = full_pipeline_data["s11"]
        assert len(s11) == 3

    def test_alpha_g1(self, full_pipeline_data):
        s11 = full_pipeline_data["s11"]
        alpha = full_pipeline_data["alpha"]
        assert s11[0] == mult(g1, int(alpha))

    def test_beta_g1(self, full_pipeline_data):
        s11 = full_pipeline_data["s11"]
        beta = full_pipeline_data["beta"]
        assert s11[1] == mult(g1, int(beta))

    def test_delta_g1(self, full_pipeline_data):
        s11 = full_pipeline_data["s11"]
        delta = full_pipeline_data["delta"]
        assert s11[2] == mult(g1, int(delta))


class TestSigma12:
    def test_length(self, full_pipeline_data):
        s12 = full_pipeline_data["s12"]
        numGates = full_pipeline_data["numGates"]
        assert len(s12) == numGates

    def test_first_element(self, full_pipeline_data):
        """첫 번째 원소는 x^0 * G1 = G1"""
        s12 = full_pipeline_data["s12"]
        assert s12[0] == mult(g1, 1)


class TestSigma13:
    def test_length(self, full_pipeline_data):
        s13 = full_pipeline_data["s13"]
        numWires = full_pipeline_data["numWires"]
        assert len(s13) == numWires

    def test_public_indices_have_values(self, full_pipeline_data):
        """공개 인덱스(0, 1)에는 실제 EC 포인트가 있어야 함"""
        s13 = full_pipeline_data["s13"]
        for idx in [0, 1]:
            assert s13[idx] is not None
            assert s13[idx] != (FQ(0), FQ(0))

    def test_private_indices_are_zero(self, full_pipeline_data):
        """비공개 인덱스에는 무한원점 (FQ(0), FQ(0))"""
        s13 = full_pipeline_data["s13"]
        numWires = full_pipeline_data["numWires"]
        for idx in range(2, numWires):
            assert s13[idx] == (FQ(0), FQ(0))


class TestSigma14:
    def test_length(self, full_pipeline_data):
        s14 = full_pipeline_data["s14"]
        numWires = full_pipeline_data["numWires"]
        assert len(s14) == numWires

    def test_public_indices_are_zero(self, full_pipeline_data):
        """공개 인덱스(0, 1)에는 무한원점"""
        s14 = full_pipeline_data["s14"]
        for idx in [0, 1]:
            assert s14[idx] == (FQ(0), FQ(0))

    def test_private_indices_have_values(self, full_pipeline_data):
        """비공개 인덱스에는 실제 EC 포인트"""
        s14 = full_pipeline_data["s14"]
        numWires = full_pipeline_data["numWires"]
        for idx in range(2, numWires):
            assert s14[idx] != (FQ(0), FQ(0))


class TestSigma15:
    def test_length(self, full_pipeline_data):
        s15 = full_pipeline_data["s15"]
        numGates = full_pipeline_data["numGates"]
        assert len(s15) == numGates - 1


class TestSigma21:
    def test_length(self, full_pipeline_data):
        s21 = full_pipeline_data["s21"]
        assert len(s21) == 3

    def test_beta_g2(self, full_pipeline_data):
        s21 = full_pipeline_data["s21"]
        beta = full_pipeline_data["beta"]
        assert s21[0] == mult(g2, int(beta))

    def test_gamma_g2(self, full_pipeline_data):
        s21 = full_pipeline_data["s21"]
        gamma = full_pipeline_data["gamma"]
        assert s21[1] == mult(g2, int(gamma))

    def test_delta_g2(self, full_pipeline_data):
        s21 = full_pipeline_data["s21"]
        delta = full_pipeline_data["delta"]
        assert s21[2] == mult(g2, int(delta))


class TestSigma22:
    def test_length(self, full_pipeline_data):
        s22 = full_pipeline_data["s22"]
        numGates = full_pipeline_data["numGates"]
        assert len(s22) == numGates

    def test_first_element(self, full_pipeline_data):
        """첫 번째 원소는 x^0 * G2 = G2"""
        s22 = full_pipeline_data["s22"]
        assert s22[0] == mult(g2, 1)
