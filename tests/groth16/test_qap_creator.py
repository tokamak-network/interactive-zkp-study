import pytest
from zkp.groth16.qap_creator import (
    multiply_polys, add_polys, subtract_polys, div_polys, eval_poly,
    mk_singleton, lagrange_interp, transpose,
    r1cs_to_qap, create_solution_polynomials, create_divisor_polynomial,
)


# ── 다항식 연산 ──
class TestMultiplyPolys:
    def test_constants(self):
        assert multiply_polys([3], [4]) == [12]

    def test_linear(self):
        # (1 + x) * (1 + x) = 1 + 2x + x^2
        assert multiply_polys([1, 1], [1, 1]) == [1, 2, 1]

    def test_identity(self):
        # p * [1] == p
        p = [3, 2, 1]
        assert multiply_polys(p, [1]) == p


class TestAddPolys:
    def test_same_length(self):
        assert add_polys([1, 2], [3, 4]) == [4, 6]

    def test_different_length(self):
        assert add_polys([1], [1, 2, 3]) == [2, 2, 3]

    def test_subtract_mode(self):
        assert add_polys([5, 3], [1, 1], subtract=True) == [4, 2]


class TestSubtractPolys:
    def test_basic(self):
        assert subtract_polys([5, 3], [1, 1]) == [4, 2]

    def test_zero(self):
        assert subtract_polys([3, 2, 1], [3, 2, 1]) == [0, 0, 0]


class TestDivPolys:
    def test_exact_division(self):
        # (x^2 - 1) / (x - 1) = (x + 1), remainder = 0
        a = [-1, 0, 1]  # -1 + x^2
        b = [-1, 1]     # -1 + x
        q, r = div_polys(a, b)
        assert abs(q[0] - 1) < 1e-10
        assert abs(q[1] - 1) < 1e-10
        for val in r:
            assert abs(val) < 1e-10

    def test_with_remainder(self):
        # (x^2 + 1) / (x - 1) => quotient (x+1), remainder 2
        a = [1, 0, 1]
        b = [-1, 1]
        q, r = div_polys(a, b)
        # 검증: a == b*q + r (근사)
        bq = multiply_polys(b, q)
        reconstructed = add_polys(bq, r)
        for i in range(len(a)):
            assert abs(reconstructed[i] - a[i]) < 1e-10


class TestEvalPoly:
    def test_constant(self):
        assert eval_poly([5], 10) == 5

    def test_linear(self):
        # 3 + 2x at x=4 => 11
        assert eval_poly([3, 2], 4) == 11

    def test_quadratic(self):
        # 1 + 0x + 1x^2 at x=3 => 10
        assert eval_poly([1, 0, 1], 3) == 10


# ── 라그랑주 보간 ──
class TestMkSingleton:
    def test_singleton_at_point(self):
        """point_loc에서만 height 값, 다른 점에서 0"""
        total_pts = 3
        height = 5
        point_loc = 2
        poly = mk_singleton(point_loc, height, total_pts)
        assert abs(eval_poly(poly, 2) - height) < 1e-10
        assert abs(eval_poly(poly, 1)) < 1e-10
        assert abs(eval_poly(poly, 3)) < 1e-10


class TestLagrangeInterp:
    def test_basic(self):
        vec = [1, 4, 9]  # p(1)=1, p(2)=4, p(3)=9
        poly = lagrange_interp(vec)
        for i, v in enumerate(vec):
            assert abs(eval_poly(poly, i + 1) - v) < 1e-10

    def test_constant(self):
        vec = [3, 3, 3]
        poly = lagrange_interp(vec)
        for i in range(3):
            assert abs(eval_poly(poly, i + 1) - 3) < 1e-10


class TestTranspose:
    def test_basic(self):
        M = [[1, 2, 3], [4, 5, 6]]
        T = transpose(M)
        assert T == [[1, 4], [2, 5], [3, 6]]


# ── R1CS → QAP ──
class TestR1csToQap:
    def test_with_known_r1cs(self):
        """qeval(x)의 알려진 R1CS에서 QAP 변환"""
        A = [[0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [5, 0, 0, 0, 0, 1]]
        B = [[0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0]]
        C = [[0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0]]

        new_A, new_B, new_C, Z = r1cs_to_qap(A, B, C)
        # 와이어 수만큼의 다항식이 있어야 함
        assert len(new_A) == 6
        assert len(new_B) == 6
        assert len(new_C) == 6
        # Z = (x-1)(x-2)(x-3)(x-4), 차수 4
        assert len(Z) == 5


class TestCreateSolutionPolynomials:
    def test_cancellation_at_gates(self):
        """A(x)*B(x) - C(x)가 게이트 점에서 0"""
        A = [[0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [5, 0, 0, 0, 0, 1]]
        B = [[0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0]]
        C = [[0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0]]
        r = [1, 3, 35, 9, 27, 30]
        new_A, new_B, new_C, Z = r1cs_to_qap(A, B, C)
        Apoly, Bpoly, Cpoly, sol = create_solution_polynomials(r, new_A, new_B, new_C)
        # sol = A(x)*B(x) - C(x) 는 x=1,2,3,4에서 0
        for x in range(1, 5):
            assert abs(eval_poly(sol, x)) < 1e-10


class TestCreateDivisorPolynomial:
    def test_h_times_z_equals_sol(self):
        A = [[0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [5, 0, 0, 0, 0, 1]]
        B = [[0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0]]
        C = [[0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0]]
        r = [1, 3, 35, 9, 27, 30]
        new_A, new_B, new_C, Z = r1cs_to_qap(A, B, C)
        _, _, _, sol = create_solution_polynomials(r, new_A, new_B, new_C)
        H = create_divisor_polynomial(sol, Z)
        # H * Z ≈ sol
        hz = multiply_polys(H, Z)
        for i in range(len(sol)):
            assert abs(hz[i] - sol[i]) < 1e-6, f"Coeff {i}: {hz[i]} != {sol[i]}"
