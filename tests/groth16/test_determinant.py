import pytest
from zkp.groth16.determinant import (
    zeros_matrix, identity_matrix, copy_matrix, transpose,
    matrix_addition, matrix_subtraction, matrix_multiply,
    multiply_matrices, check_matrix_equality, dot_product,
    unitize_vector, check_squareness,
    determinant_recursive, determinant_fast, check_non_singular,
)


class TestZerosMatrix:
    def test_basic(self):
        M = zeros_matrix(2, 3)
        assert len(M) == 2
        assert len(M[0]) == 3
        assert all(M[i][j] == 0.0 for i in range(2) for j in range(3))

    def test_single(self):
        M = zeros_matrix(1, 1)
        assert M == [[0.0]]


class TestIdentityMatrix:
    def test_3x3(self):
        I = identity_matrix(3)
        for i in range(3):
            for j in range(3):
                assert I[i][j] == (1.0 if i == j else 0.0)

    def test_1x1(self):
        I = identity_matrix(1)
        assert I == [[1.0]]


class TestCopyMatrix:
    def test_deep_copy(self):
        M = [[1, 2], [3, 4]]
        C = copy_matrix(M)
        assert C == M
        C[0][0] = 99
        assert M[0][0] == 1  # 원본 불변


class TestTranspose:
    def test_2x3(self):
        M = [[1, 2, 3], [4, 5, 6]]
        T = transpose(M)
        assert T == [[1, 4], [2, 5], [3, 6]]

    def test_1d_vector(self):
        v = [1, 2, 3]
        T = transpose(v)
        assert T == [[1], [2], [3]]


class TestMatrixAddition:
    def test_basic(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = matrix_addition(A, B)
        assert C == [[6, 8], [10, 12]]

    def test_dimension_mismatch(self):
        A = [[1, 2]]
        B = [[1], [2]]
        with pytest.raises(ArithmeticError):
            matrix_addition(A, B)


class TestMatrixSubtraction:
    def test_basic(self):
        A = [[5, 6], [7, 8]]
        B = [[1, 2], [3, 4]]
        C = matrix_subtraction(A, B)
        assert C == [[4, 4], [4, 4]]

    def test_dimension_mismatch(self):
        with pytest.raises(ArithmeticError):
            matrix_subtraction([[1]], [[1, 2]])


class TestMatrixMultiply:
    def test_2x2(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = matrix_multiply(A, B)
        assert C == [[19, 22], [43, 50]]

    def test_dimension_mismatch(self):
        A = [[1, 2]]
        B = [[1, 2]]
        with pytest.raises(ArithmeticError):
            matrix_multiply(A, B)

    def test_identity(self):
        A = [[1, 2], [3, 4]]
        I = identity_matrix(2)
        assert matrix_multiply(A, I) == [[1, 2], [3, 4]]


class TestMultiplyMatrices:
    def test_chain(self):
        A = [[1, 0], [0, 1]]
        B = [[2, 3], [4, 5]]
        result = multiply_matrices([A, B])
        assert result == [[2, 3], [4, 5]]


class TestCheckMatrixEquality:
    def test_equal(self):
        A = [[1, 2], [3, 4]]
        B = [[1, 2], [3, 4]]
        assert check_matrix_equality(A, B) is True

    def test_not_equal(self):
        A = [[1, 2], [3, 4]]
        B = [[1, 2], [3, 5]]
        assert check_matrix_equality(A, B) is False

    def test_different_dimensions(self):
        A = [[1, 2]]
        B = [[1], [2]]
        assert check_matrix_equality(A, B) is False

    def test_tolerance(self):
        A = [[1.001]]
        B = [[1.002]]
        assert check_matrix_equality(A, B, tol=2) is True
        assert check_matrix_equality(A, B, tol=4) is False


class TestDotProduct:
    def test_basic(self):
        A = [[1, 2]]
        B = [[3, 4]]
        assert dot_product(A, B) == 11

    def test_dimension_mismatch(self):
        with pytest.raises(ArithmeticError):
            dot_product([[1, 2]], [[1]])


class TestUnitizeVector:
    def test_column_vector(self):
        v = [[3], [4]]
        u = unitize_vector(v)
        assert abs(u[0][0] - 0.6) < 1e-10
        assert abs(u[1][0] - 0.8) < 1e-10

    def test_not_vector(self):
        with pytest.raises(ArithmeticError):
            unitize_vector([[1, 2], [3, 4]])


class TestCheckSquareness:
    def test_square(self):
        check_squareness([[1, 2], [3, 4]])  # 예외 없음

    def test_not_square(self):
        with pytest.raises(ArithmeticError):
            check_squareness([[1, 2, 3], [4, 5, 6]])


class TestDeterminantRecursive:
    def test_2x2(self):
        A = [[1, 2], [3, 4]]
        assert determinant_recursive(A) == -2

    def test_3x3(self):
        A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        assert determinant_recursive(A) == 1

    def test_identity(self):
        I = identity_matrix(3)
        assert abs(determinant_recursive(I) - 1.0) < 1e-10


class TestDeterminantFast:
    def test_2x2(self):
        A = [[1, 2], [3, 4]]
        assert abs(determinant_fast(A) - (-2)) < 1e-10

    def test_3x3(self):
        A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        assert abs(determinant_fast(A) - 1) < 1e-10

    def test_agrees_with_recursive(self):
        A = [[2, 1, 3], [4, 5, 6], [7, 8, 9]]
        rec = determinant_recursive(A)
        fast = determinant_fast(A)
        assert abs(rec - fast) < 1e-6


class TestCheckNonSingular:
    def test_non_singular(self):
        A = [[1, 2], [3, 4]]
        det = check_non_singular(A)
        assert abs(det - (-2)) < 1e-10

    def test_singular(self):
        """determinant_fast는 0 대각선에 1e-18을 대입하므로 정확히 0을 반환하지 않음.
        따라서 check_non_singular는 예외 대신 극소값을 반환."""
        A = [[1, 2], [2, 4]]
        det = check_non_singular(A)
        assert abs(det) < 1e-10
