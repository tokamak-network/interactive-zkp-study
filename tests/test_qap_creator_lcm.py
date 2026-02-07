import pytest
from zkp.groth16.qap_creator_lcm import (
    multiply_polys, add_polys, subtract_polys, div_polys, eval_poly,
    mk_singleton, lagrange_interp, transpose,
    k_matrix, r1cs_to_qap_times_lcm,
    create_solution_polynomials, create_divisor_polynomial,
)
from zkp.groth16.determinant import determinant_fast


# ── k_matrix ──
class TestKMatrix:
    def test_2x2(self):
        m = k_matrix(2)
        # [[1^0, 1^1], [2^0, 2^1]] = [[1,1],[1,2]]
        assert m == [[1, 1], [1, 2]]

    def test_3x3(self):
        m = k_matrix(3)
        assert m == [[1, 1, 1], [1, 2, 4], [1, 3, 9]]

    def test_4x4(self):
        m = k_matrix(4)
        assert len(m) == 4
        assert len(m[0]) == 4
        # 첫 행은 항상 [1, 1, 1, ..., 1]
        assert m[0] == [1, 1, 1, 1]

    def test_determinant(self):
        """4x4 k_matrix의 행렬식은 12.0"""
        m = k_matrix(4)
        det = determinant_fast(m)
        assert abs(det - 12.0) < 1e-10


# ── r1cs_to_qap_times_lcm ──
class TestR1csToQapTimesLcm:
    @pytest.fixture
    def known_r1cs(self):
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
        return A, B, C

    def test_output_dimensions(self, known_r1cs):
        A, B, C = known_r1cs
        new_A, new_B, new_C, Z = r1cs_to_qap_times_lcm(A, B, C)
        assert len(new_A) == 6  # 와이어 수
        assert len(new_B) == 6
        assert len(new_C) == 6
        assert len(Z) == 5  # 차수 4 다항식

    def test_lcm_scaling(self, known_r1cs):
        """LCM 스케일링이 적용되었는지 확인"""
        A, B, C = known_r1cs
        new_A, new_B, new_C, Z = r1cs_to_qap_times_lcm(A, B, C)
        det_k = determinant_fast(k_matrix(4))
        # A 다항식의 계수들이 det_k로 스케일됨
        # lagrange_interp(a) * det_k == new_A[i]
        # 정수 계수가 되어야 함 (det_k 배율)
        for poly in new_A:
            for coeff in poly:
                assert abs(coeff - round(coeff)) < 1e-6


class TestCreateSolutionPolynomialsLcm:
    def test_divisibility(self):
        """LCM 변형에서도 sol이 Z로 나누어 떨어짐"""
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

        new_A, new_B, new_C, Z = r1cs_to_qap_times_lcm(A, B, C)
        Apoly, Bpoly, Cpoly, sol = create_solution_polynomials(r, new_A, new_B, new_C)
        H = create_divisor_polynomial(sol, Z)

        # H * Z ≈ sol
        hz = multiply_polys(H, Z)
        for i in range(len(sol)):
            assert abs(hz[i] - sol[i]) < 1e-4, f"Coeff {i}: {hz[i]} != {sol[i]}"


# ── transpose ──
class TestTransposeLcm:
    def test_basic(self):
        M = [[1, 2], [3, 4], [5, 6]]
        assert transpose(M) == [[1, 3, 5], [2, 4, 6]]


# ── 다항식 함수 (LCM 모듈 내 버전도 동일하게 동작) ──
class TestPolyOpsLcm:
    def test_multiply(self):
        assert multiply_polys([1, 1], [1, -1]) == [1, 0, -1]

    def test_add(self):
        assert add_polys([1, 2], [3, 4]) == [4, 6]

    def test_subtract(self):
        assert subtract_polys([5, 3], [1, 1]) == [4, 2]

    def test_div_exact(self):
        a = [-1, 0, 1]
        b = [-1, 1]
        q, r = div_polys(a, b)
        assert abs(q[0] - 1) < 1e-10
        assert abs(q[1] - 1) < 1e-10

    def test_eval(self):
        assert eval_poly([3, 2, 1], 2) == 11  # 3 + 4 + 4
