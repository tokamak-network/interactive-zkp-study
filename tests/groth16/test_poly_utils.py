import pytest
from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from zkp.groth16.poly_utils import (
    _multiply_polys, _add_polys, _subtract_polys, _div_polys, _eval_poly,
    _multiply_vec_matrix, _multiply_vec_vec,
    getNumWires, getNumGates,
    getFRPoly1D, getFRPoly2D,
    ax_val, bx_val, cx_val, zx_val, hx_val, hxr,
    FR,
)


# ── FR 필드 다항식 연산 ──
class TestFRPolynomials:
    def test_multiply_polys(self):
        a = [FR(1), FR(1)]
        b = [FR(1), FR(1)]
        result = _multiply_polys(a, b)
        assert result[0] == FR(1)
        assert result[1] == FR(2)
        assert result[2] == FR(1)

    def test_add_polys(self):
        a = [FR(1), FR(2)]
        b = [FR(3), FR(4)]
        result = _add_polys(a, b)
        assert result[0] == FR(4)
        assert result[1] == FR(6)

    def test_subtract_polys(self):
        a = [FR(5), FR(3)]
        b = [FR(1), FR(1)]
        result = _subtract_polys(a, b)
        assert result[0] == FR(4)
        assert result[1] == FR(2)

    def test_div_polys(self):
        # (x^2 - 1) / (x - 1) = (x + 1)
        a = [FR(-1), FR(0), FR(1)]
        b = [FR(-1), FR(1)]
        q, r = _div_polys(a, b)
        assert q[0] == FR(1)
        assert q[1] == FR(1)

    def test_eval_poly(self):
        # 3 + 2x at x=4 => 11
        poly = [FR(3), FR(2)]
        assert _eval_poly(poly, FR(4)) == FR(11)


# ── 벡터 연산 ──
class TestVectorOps:
    def test_multiply_vec_vec(self):
        v1 = [FR(1), FR(2), FR(3)]
        v2 = [FR(4), FR(5), FR(6)]
        result = _multiply_vec_vec(v1, v2)
        # 1*4 + 2*5 + 3*6 = 32
        assert result == FR(32)

    def test_multiply_vec_vec_zero(self):
        v1 = [FR(0), FR(0)]
        v2 = [FR(5), FR(10)]
        assert _multiply_vec_vec(v1, v2) == FR(0)


# ── 유틸리티 함수 ──
class TestUtilFunctions:
    def test_getNumWires(self):
        Ax = [[1, 2], [3, 4], [5, 6]]
        assert getNumWires(Ax) == 3

    def test_getNumGates(self):
        Ax = [[1, 2, 3], [4, 5, 6]]
        assert getNumGates(Ax) == 3


# ── FR 변환 ──
class TestFRConversion:
    def test_getFRPoly1D(self):
        poly = [1.0, 2.0, 3.0]
        result = getFRPoly1D(poly)
        assert all(isinstance(x, FR) for x in result)
        assert result[0] == FR(1)
        assert result[2] == FR(3)

    def test_getFRPoly1D_rounding(self):
        poly = [1.7, 2.3]
        result = getFRPoly1D(poly)
        assert result[0] == FR(2)
        assert result[1] == FR(2)

    def test_getFRPoly2D(self):
        poly = [[1.0, 2.0], [3.0, 4.0]]
        result = getFRPoly2D(poly)
        assert result[0][0] == FR(1)
        assert result[1][1] == FR(4)


# ── 다항식 평가 함수 ──
class TestPolyEvalFunctions:
    @pytest.fixture
    def sample_poly_2d(self):
        """2 와이어, 3 게이트 다항식"""
        return [
            [FR(1), FR(2), FR(3)],
            [FR(4), FR(5), FR(6)],
        ]

    def test_ax_val(self, sample_poly_2d):
        result = ax_val(sample_poly_2d, FR(2))
        # wire 0: 1 + 2*2 + 3*4 = 17
        assert result[0] == FR(17)
        # wire 1: 4 + 5*2 + 6*4 = 38
        assert result[1] == FR(38)

    def test_bx_val(self, sample_poly_2d):
        result = bx_val(sample_poly_2d, FR(2))
        assert result[0] == FR(17)

    def test_cx_val(self, sample_poly_2d):
        result = cx_val(sample_poly_2d, FR(2))
        assert result[0] == FR(17)

    def test_zx_val(self):
        Zx = [FR(1), FR(-1)]  # 1 - x
        assert zx_val(Zx, FR(3)) == FR(-2)

    def test_hx_val(self):
        Hx = [FR(2), FR(3)]  # 2 + 3x
        assert hx_val(Hx, FR(4)) == FR(14)


# ── hxr (통합) ──
class TestHxr:
    def test_with_pipeline(self, qap_data):
        """QAP 파이프라인 데이터로 hxr 검증"""
        Ax = getFRPoly2D(qap_data["Ap"])
        Bx = getFRPoly2D(qap_data["Bp"])
        Cx = getFRPoly2D(qap_data["Cp"])
        Zx = getFRPoly1D(qap_data["Z"])
        R = qap_data["r"]

        Hx, remainder = hxr(Ax, Bx, Cx, Zx, R)
        # 나머지가 0에 가까워야 함 (FR 필드에서 정확히 0)
        for val in remainder:
            assert val == FR(0), f"remainder not zero: {val}"
