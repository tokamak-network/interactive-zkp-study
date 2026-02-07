"""
Foundation module tests: field.py, polynomial.py, utils.py
"""
import pytest
from zkp.plonk.field import (
    FR, CURVE_ORDER, G1, G2, Z1,
    ec_mul, ec_add, ec_neg, ec_pairing,
    get_root_of_unity, get_roots_of_unity,
)
from zkp.plonk.polynomial import Polynomial, fft, ifft, poly_div, lagrange_basis
from zkp.plonk.utils import (
    vanishing_poly_eval,
    lagrange_basis_eval,
    public_input_polynomial,
    public_input_poly_eval,
    coset_fft,
    coset_ifft,
    pad_to_power_of_2,
    next_power_of_2,
)


# =====================================================================
# FR arithmetic
# =====================================================================

class TestFR:
    def test_creation(self):
        a = FR(0)
        b = FR(1)
        c = FR(CURVE_ORDER - 1)
        assert int(a) == 0
        assert int(b) == 1
        assert int(c) == CURVE_ORDER - 1

    def test_modular_reduction(self):
        a = FR(CURVE_ORDER)
        assert a == FR(0)
        b = FR(CURVE_ORDER + 7)
        assert b == FR(7)

    def test_addition(self):
        a = FR(3)
        b = FR(5)
        assert a + b == FR(8)

    def test_addition_wrap(self):
        a = FR(CURVE_ORDER - 1)
        b = FR(2)
        assert a + b == FR(1)

    def test_subtraction(self):
        a = FR(10)
        b = FR(3)
        assert a - b == FR(7)

    def test_subtraction_wrap(self):
        a = FR(0)
        b = FR(1)
        assert a - b == FR(CURVE_ORDER - 1)

    def test_multiplication(self):
        a = FR(6)
        b = FR(7)
        assert a * b == FR(42)

    def test_multiplication_by_zero(self):
        a = FR(12345)
        assert a * FR(0) == FR(0)

    def test_division(self):
        a = FR(42)
        b = FR(7)
        assert a / b == FR(6)

    def test_division_inverse(self):
        a = FR(3)
        inv_a = FR(1) / a
        assert a * inv_a == FR(1)

    def test_power(self):
        a = FR(2)
        assert a ** 10 == FR(1024)

    def test_fermat_little_theorem(self):
        a = FR(7)
        assert a ** (CURVE_ORDER - 1) == FR(1)

    def test_negative(self):
        a = FR(5)
        neg_a = FR(0) - a
        assert a + neg_a == FR(0)

    def test_equality(self):
        assert FR(3) == FR(3)
        assert FR(3) != FR(4)

    def test_field_modulus(self):
        assert FR.field_modulus == CURVE_ORDER


# =====================================================================
# EC operations
# =====================================================================

class TestEC:
    def test_ec_mul_generator(self):
        P = ec_mul(G1, 1)
        assert P == G1

    def test_ec_mul_zero(self):
        P = ec_mul(G1, 0)
        assert P is None or P == Z1

    def test_ec_mul_fr(self):
        P = ec_mul(G1, FR(5))
        Q = ec_mul(G1, 5)
        assert P == Q

    def test_ec_add_identity(self):
        P = ec_add(G1, Z1)
        assert P == G1

    def test_ec_add_same(self):
        P = ec_add(G1, G1)
        Q = ec_mul(G1, 2)
        assert P == Q

    def test_ec_add_commutative(self):
        P = ec_mul(G1, 3)
        Q = ec_mul(G1, 7)
        assert ec_add(P, Q) == ec_add(Q, P)

    def test_ec_add_associative(self):
        A = ec_mul(G1, 2)
        B = ec_mul(G1, 3)
        C = ec_mul(G1, 5)
        lhs = ec_add(ec_add(A, B), C)
        rhs = ec_add(A, ec_add(B, C))
        assert lhs == rhs

    def test_ec_neg(self):
        P = ec_mul(G1, 5)
        neg_P = ec_neg(P)
        result = ec_add(P, neg_P)
        assert result is None or result == Z1

    def test_ec_mul_scalar_add(self):
        P = ec_mul(G1, 3)
        Q = ec_mul(G1, 7)
        R = ec_add(P, Q)
        S = ec_mul(G1, 10)
        assert R == S

    def test_ec_mul_modular(self):
        P = ec_mul(G1, CURVE_ORDER)
        assert P is None or P == Z1

    def test_ec_pairing_basic(self):
        e1 = ec_pairing(G2, G1)
        assert e1 is not None
        # Pairing of identity point gives FQ12.one()
        e_identity = ec_pairing(G2, Z1)
        assert e1 != e_identity

    def test_ec_pairing_bilinearity(self):
        a, b = 3, 5
        lhs = ec_pairing(G2, ec_mul(G1, a * b))
        rhs = ec_pairing(ec_mul(G2, b), ec_mul(G1, a))
        assert lhs == rhs

    def test_z1_is_none(self):
        assert Z1 is None


# =====================================================================
# Roots of Unity
# =====================================================================

class TestRootsOfUnity:
    def test_get_root_of_unity_1(self):
        omega = get_root_of_unity(1)
        assert omega == FR(1)

    def test_get_root_of_unity_power(self):
        for k in [2, 4, 8, 16]:
            omega = get_root_of_unity(k)
            assert omega ** k == FR(1)

    def test_primitive_root(self):
        omega = get_root_of_unity(4)
        assert omega ** 4 == FR(1)
        assert omega ** 2 != FR(1)
        assert omega ** 1 != FR(1)

    def test_get_root_non_power_of_2(self):
        with pytest.raises(ValueError):
            get_root_of_unity(3)

    def test_get_root_zero(self):
        with pytest.raises(ValueError):
            get_root_of_unity(0)

    def test_get_root_too_large(self):
        with pytest.raises(ValueError):
            get_root_of_unity(1 << 29)

    def test_get_roots_of_unity_length(self):
        roots = get_roots_of_unity(8)
        assert len(roots) == 8

    def test_get_roots_of_unity_first(self):
        roots = get_roots_of_unity(4)
        assert roots[0] == FR(1)

    def test_get_roots_of_unity_all_nth(self):
        n = 8
        roots = get_roots_of_unity(n)
        for r in roots:
            assert r ** n == FR(1)

    def test_get_roots_of_unity_distinct(self):
        roots = get_roots_of_unity(8)
        assert len(set(int(r) for r in roots)) == 8


# =====================================================================
# Polynomial creation and basic ops
# =====================================================================

class TestPolynomialBasic:
    def test_creation_from_fr(self):
        p = Polynomial([FR(1), FR(2), FR(3)])
        assert p.coeffs == [FR(1), FR(2), FR(3)]

    def test_creation_from_int(self):
        p = Polynomial([1, 2, 3])
        assert p.coeffs == [FR(1), FR(2), FR(3)]

    def test_creation_none(self):
        p = Polynomial()
        assert p.is_zero()

    def test_trim(self):
        p = Polynomial([FR(1), FR(2), FR(0), FR(0)])
        assert len(p.coeffs) == 2

    def test_degree_constant(self):
        p = Polynomial([FR(5)])
        assert p.degree == 1 - 1  # 0

    def test_degree_linear(self):
        p = Polynomial([FR(1), FR(2)])
        assert p.degree == 1

    def test_degree_quadratic(self):
        p = Polynomial([FR(1), FR(0), FR(3)])
        assert p.degree == 2

    def test_degree_zero_poly(self):
        p = Polynomial.zero()
        assert p.degree == 0

    def test_is_zero_true(self):
        assert Polynomial.zero().is_zero()
        assert Polynomial([FR(0)]).is_zero()

    def test_is_zero_false(self):
        assert not Polynomial([FR(1)]).is_zero()
        assert not Polynomial([FR(0), FR(1)]).is_zero()

    def test_len(self):
        p = Polynomial([FR(1), FR(2), FR(3)])
        assert len(p) == 3

    def test_repr(self):
        p = Polynomial([FR(1), FR(2)])
        r = repr(p)
        assert "Poly" in r

    def test_zero_class_method(self):
        p = Polynomial.zero()
        assert p.is_zero()
        assert p.coeffs == [FR(0)]

    def test_one_class_method(self):
        p = Polynomial.one()
        assert p.coeffs == [FR(1)]
        assert not p.is_zero()


# =====================================================================
# Polynomial arithmetic
# =====================================================================

class TestPolynomialArithmetic:
    def test_add(self):
        p = Polynomial([FR(1), FR(2)])
        q = Polynomial([FR(3), FR(4)])
        r = p + q
        assert r.coeffs == [FR(4), FR(6)]

    def test_add_different_degree(self):
        p = Polynomial([FR(1), FR(2), FR(3)])
        q = Polynomial([FR(4)])
        r = p + q
        assert r.coeffs == [FR(5), FR(2), FR(3)]

    def test_add_scalar(self):
        p = Polynomial([FR(1), FR(2)])
        r = p + FR(3)
        assert r.coeffs == [FR(4), FR(2)]

    def test_add_int(self):
        p = Polynomial([FR(1), FR(2)])
        r = p + 3
        assert r.coeffs == [FR(4), FR(2)]

    def test_radd(self):
        p = Polynomial([FR(1), FR(2)])
        r = 3 + p
        assert r.coeffs == [FR(4), FR(2)]

    def test_sub(self):
        p = Polynomial([FR(5), FR(7)])
        q = Polynomial([FR(2), FR(3)])
        r = p - q
        assert r.coeffs == [FR(3), FR(4)]

    def test_rsub(self):
        p = Polynomial([FR(1), FR(2)])
        r = 5 - p
        assert r.coeffs == [FR(4), FR(CURVE_ORDER - 2)]

    def test_neg(self):
        p = Polynomial([FR(1), FR(2)])
        q = -p
        assert q.coeffs == [FR(CURVE_ORDER - 1), FR(CURVE_ORDER - 2)]
        r = p + q
        assert r.is_zero()

    def test_mul_poly(self):
        # (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        p = Polynomial([FR(1), FR(2)])
        q = Polynomial([FR(3), FR(4)])
        r = p * q
        assert r.coeffs == [FR(3), FR(10), FR(8)]

    def test_mul_scalar_fr(self):
        p = Polynomial([FR(1), FR(2)])
        r = p * FR(3)
        assert r.coeffs == [FR(3), FR(6)]

    def test_mul_scalar_int(self):
        p = Polynomial([FR(1), FR(2)])
        r = p * 3
        assert r.coeffs == [FR(3), FR(6)]

    def test_rmul(self):
        p = Polynomial([FR(1), FR(2)])
        r = 3 * p
        assert r.coeffs == [FR(3), FR(6)]

    def test_eq(self):
        p = Polynomial([FR(1), FR(2)])
        q = Polynomial([FR(1), FR(2)])
        assert p == q

    def test_neq(self):
        p = Polynomial([FR(1), FR(2)])
        q = Polynomial([FR(1), FR(3)])
        assert p != q

    def test_eq_with_int(self):
        p = Polynomial([FR(5)])
        assert p == 5

    def test_scale(self):
        p = Polynomial([FR(1), FR(2)])
        r = p.scale(FR(3))
        assert r.coeffs == [FR(3), FR(6)]


# =====================================================================
# Polynomial evaluate
# =====================================================================

class TestPolynomialEvaluate:
    def test_evaluate_constant(self):
        p = Polynomial([FR(7)])
        assert p.evaluate(FR(100)) == FR(7)

    def test_evaluate_linear(self):
        # 3 + 2x at x=5 => 13
        p = Polynomial([FR(3), FR(2)])
        assert p.evaluate(FR(5)) == FR(13)

    def test_evaluate_quadratic(self):
        # 1 + 2x + 3x^2 at x=2 => 1+4+12 = 17
        p = Polynomial([FR(1), FR(2), FR(3)])
        assert p.evaluate(FR(2)) == FR(17)

    def test_evaluate_zero_poly(self):
        p = Polynomial.zero()
        assert p.evaluate(FR(42)) == FR(0)

    def test_evaluate_at_zero(self):
        p = Polynomial([FR(5), FR(3), FR(2)])
        assert p.evaluate(FR(0)) == FR(5)

    def test_evaluate_int_input(self):
        p = Polynomial([FR(1), FR(1)])  # 1 + x
        assert p.evaluate(4) == FR(5)


# =====================================================================
# Polynomial divide_by_vanishing
# =====================================================================

class TestDivideByVanishing:
    def test_exact_division(self):
        n = 4
        omega = get_root_of_unity(n)
        roots = get_roots_of_unity(n)
        # Build a polynomial that vanishes on all roots: f(x) = Z_H(x) * (1 + x)
        zh = Polynomial.vanishing(n)
        factor = Polynomial([FR(1), FR(1)])
        f = zh * factor
        q = f.divide_by_vanishing(n)
        assert q == factor

    def test_non_exact_division_raises(self):
        # x^2 + 1 is not divisible by x^4 - 1
        p = Polynomial([FR(1), FR(0), FR(1)])
        with pytest.raises(ValueError):
            p.divide_by_vanishing(4)


# =====================================================================
# Polynomial vanishing
# =====================================================================

class TestVanishing:
    def test_vanishing_evaluates_to_zero_on_roots(self):
        n = 4
        roots = get_roots_of_unity(n)
        zh = Polynomial.vanishing(n)
        for r in roots:
            assert zh.evaluate(r) == FR(0)

    def test_vanishing_degree(self):
        zh = Polynomial.vanishing(4)
        assert zh.degree == 4

    def test_vanishing_leading_coeff(self):
        zh = Polynomial.vanishing(4)
        assert zh.coeffs[-1] == FR(1)
        assert zh.coeffs[0] == FR(CURVE_ORDER - 1)  # -1 mod p


# =====================================================================
# Polynomial.from_evaluations
# =====================================================================

class TestFromEvaluations:
    def test_roundtrip(self):
        n = 4
        omega = get_root_of_unity(n)
        p = Polynomial([FR(1), FR(2), FR(3), FR(0)])
        evals = [p.evaluate(omega ** i) for i in range(n)]
        q = Polynomial.from_evaluations(evals, omega)
        assert q == p

    def test_constant_polynomial(self):
        n = 4
        omega = get_root_of_unity(n)
        evals = [FR(5)] * n
        p = Polynomial.from_evaluations(evals, omega)
        assert p == Polynomial([FR(5)])


# =====================================================================
# FFT / IFFT
# =====================================================================

class TestFFT:
    def test_fft_single(self):
        result = fft([FR(7)], FR(1))
        assert result == [FR(7)]

    def test_fft_basic(self):
        n = 4
        omega = get_root_of_unity(n)
        coeffs = [FR(1), FR(2), FR(3), FR(4)]
        evals = fft(coeffs, omega)
        assert len(evals) == n
        # Check manually: evals[0] = p(1) = 1+2+3+4 = 10
        p = Polynomial(coeffs)
        assert evals[0] == p.evaluate(FR(1))

    def test_fft_all_points(self):
        n = 8
        omega = get_root_of_unity(n)
        coeffs = [FR(i) for i in range(n)]
        evals = fft(coeffs, omega)
        p = Polynomial(coeffs)
        for i in range(n):
            assert evals[i] == p.evaluate(omega ** i)

    def test_ifft_single(self):
        result = ifft([FR(7)], FR(1))
        assert result == [FR(7)]

    def test_ifft_basic(self):
        n = 4
        omega = get_root_of_unity(n)
        evals = [FR(10), FR(5), FR(3), FR(7)]
        coeffs = ifft(evals, omega)
        # Verify roundtrip: fft(coeffs) == evals
        recovered = fft(coeffs, omega)
        for i in range(n):
            assert recovered[i] == evals[i]

    def test_fft_ifft_roundtrip(self):
        n = 8
        omega = get_root_of_unity(n)
        original = [FR(i * 3 + 1) for i in range(n)]
        evals = fft(original, omega)
        recovered = ifft(evals, omega)
        for i in range(n):
            assert recovered[i] == original[i]

    def test_ifft_fft_roundtrip(self):
        n = 4
        omega = get_root_of_unity(n)
        original = [FR(7), FR(11), FR(13), FR(17)]
        coeffs = ifft(original, omega)
        recovered = fft(coeffs, omega)
        for i in range(n):
            assert recovered[i] == original[i]


# =====================================================================
# poly_div
# =====================================================================

class TestPolyDiv:
    def test_exact_division(self):
        # (x^2 - 1) / (x - 1) = (x + 1), remainder 0
        a = Polynomial([FR(CURVE_ORDER - 1), FR(0), FR(1)])  # x^2 - 1
        b = Polynomial([FR(CURVE_ORDER - 1), FR(1)])  # x - 1
        q, r = poly_div(a, b)
        assert q == Polynomial([FR(1), FR(1)])  # x + 1
        assert r.is_zero()

    def test_with_remainder(self):
        # (x^2 + 1) / (x - 1) => quotient x+1, remainder 2
        a = Polynomial([FR(1), FR(0), FR(1)])  # x^2 + 1
        b = Polynomial([FR(CURVE_ORDER - 1), FR(1)])  # x - 1
        q, r = poly_div(a, b)
        # Verify: b * q + r == a
        check = b * q + r
        assert check == a

    def test_degree_less_than_divisor(self):
        a = Polynomial([FR(3)])
        b = Polynomial([FR(1), FR(1)])
        q, r = poly_div(a, b)
        assert q.is_zero()
        assert r == a

    def test_divide_by_zero_raises(self):
        a = Polynomial([FR(1), FR(2)])
        b = Polynomial.zero()
        with pytest.raises(ValueError):
            poly_div(a, b)

    def test_polynomial_identity(self):
        # For any a, b (b != 0): a = b*q + r
        a = Polynomial([FR(5), FR(3), FR(7), FR(2)])
        b = Polynomial([FR(1), FR(1)])
        q, r = poly_div(a, b)
        assert b * q + r == a


# =====================================================================
# lagrange_basis
# =====================================================================

class TestLagrangeBasis:
    def test_kronecker_delta(self):
        domain = [FR(1), FR(2), FR(3)]
        for i in range(3):
            L_i = lagrange_basis(domain, i)
            for j in range(3):
                val = L_i.evaluate(domain[j])
                expected = FR(1) if i == j else FR(0)
                assert val == expected

    def test_with_roots_of_unity(self):
        n = 4
        roots = get_roots_of_unity(n)
        for i in range(n):
            L_i = lagrange_basis(roots, i)
            for j in range(n):
                val = L_i.evaluate(roots[j])
                expected = FR(1) if i == j else FR(0)
                assert val == expected


# =====================================================================
# utils: vanishing_poly_eval
# =====================================================================

class TestVanishingPolyEval:
    def test_on_root(self):
        n = 4
        omega = get_root_of_unity(n)
        for i in range(n):
            assert vanishing_poly_eval(n, omega ** i) == FR(0)

    def test_off_root(self):
        zeta = FR(17)
        val = vanishing_poly_eval(4, zeta)
        assert val == FR(17) ** 4 - FR(1)

    def test_matches_polynomial(self):
        n = 4
        zeta = FR(42)
        zh = Polynomial.vanishing(n)
        assert vanishing_poly_eval(n, zeta) == zh.evaluate(zeta)


# =====================================================================
# utils: lagrange_basis_eval
# =====================================================================

class TestLagrangeBasisEval:
    def test_kronecker_delta(self):
        n = 4
        omega = get_root_of_unity(n)
        for i in range(n):
            for j in range(n):
                val = lagrange_basis_eval(i, n, omega, omega ** j)
                expected = FR(1) if i == j else FR(0)
                assert val == expected

    def test_off_domain(self):
        n = 4
        omega = get_root_of_unity(n)
        zeta = FR(17)
        roots = get_roots_of_unity(n)
        for i in range(n):
            L_i = lagrange_basis(roots, i)
            expected = L_i.evaluate(zeta)
            actual = lagrange_basis_eval(i, n, omega, zeta)
            assert actual == expected

    def test_sum_to_one(self):
        n = 4
        omega = get_root_of_unity(n)
        zeta = FR(99)
        total = FR(0)
        for i in range(n):
            total = total + lagrange_basis_eval(i, n, omega, zeta)
        assert total == FR(1)


# =====================================================================
# utils: public_input_polynomial
# =====================================================================

class TestPublicInputPolynomial:
    def test_empty_inputs(self):
        n = 4
        omega = get_root_of_unity(n)
        pi = public_input_polynomial([], n, omega)
        assert pi.is_zero()

    def test_single_input(self):
        n = 4
        omega = get_root_of_unity(n)
        pi = public_input_polynomial([FR(7)], n, omega)
        # PI(1) = 7 (at omega^0)
        assert pi.evaluate(FR(1)) == FR(7)
        # PI(omega^i) = 0 for i > 0
        for i in range(1, n):
            assert pi.evaluate(omega ** i) == FR(0)

    def test_multiple_inputs(self):
        n = 4
        omega = get_root_of_unity(n)
        vals = [FR(10), FR(20)]
        pi = public_input_polynomial(vals, n, omega)
        assert pi.evaluate(omega ** 0) == FR(10)
        assert pi.evaluate(omega ** 1) == FR(20)
        for i in range(2, n):
            assert pi.evaluate(omega ** i) == FR(0)


# =====================================================================
# utils: public_input_poly_eval
# =====================================================================

class TestPublicInputPolyEval:
    def test_matches_polynomial(self):
        n = 4
        omega = get_root_of_unity(n)
        pub = [FR(5), FR(10)]
        pi_poly = public_input_polynomial(pub, n, omega)
        zeta = FR(42)
        assert public_input_poly_eval(pub, n, omega, zeta) == pi_poly.evaluate(zeta)

    def test_empty(self):
        n = 4
        omega = get_root_of_unity(n)
        assert public_input_poly_eval([], n, omega, FR(7)) == FR(0)


# =====================================================================
# utils: coset_fft / coset_ifft
# =====================================================================

class TestCosetFFT:
    def test_coset_fft_evaluates_on_coset(self):
        n = 4
        omega = get_root_of_unity(n)
        k = FR(5)
        coeffs = [FR(1), FR(2), FR(3), FR(0)]
        p = Polynomial(coeffs)
        evals = coset_fft(coeffs, omega, k)
        for i in range(n):
            expected = p.evaluate(k * omega ** i)
            assert evals[i] == expected

    def test_coset_fft_ifft_roundtrip(self):
        n = 4
        omega = get_root_of_unity(n)
        k = FR(5)
        original = [FR(7), FR(11), FR(13), FR(17)]
        evals = coset_fft(original, omega, k)
        recovered = coset_ifft(evals, omega, k)
        for i in range(n):
            assert recovered[i] == original[i]

    def test_coset_default_k(self):
        n = 4
        omega = get_root_of_unity(n)
        coeffs = [FR(1), FR(2), FR(3), FR(4)]
        evals_explicit = coset_fft(coeffs, omega, FR(5))
        evals_default = coset_fft(coeffs, omega)
        assert evals_explicit == evals_default

    def test_coset_ifft_default_k(self):
        n = 4
        omega = get_root_of_unity(n)
        evals = [FR(1), FR(2), FR(3), FR(4)]
        r_explicit = coset_ifft(evals, omega, FR(5))
        r_default = coset_ifft(evals, omega)
        assert r_explicit == r_default


# =====================================================================
# utils: pad_to_power_of_2, next_power_of_2
# =====================================================================

class TestPadding:
    def test_next_power_of_2_exact(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(4) == 4
        assert next_power_of_2(8) == 8

    def test_next_power_of_2_non_exact(self):
        assert next_power_of_2(3) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(7) == 8
        assert next_power_of_2(9) == 16

    def test_next_power_of_2_zero(self):
        assert next_power_of_2(0) == 1

    def test_pad_already_power(self):
        lst = [FR(1), FR(2), FR(3), FR(4)]
        result = pad_to_power_of_2(lst)
        assert len(result) == 4

    def test_pad_needs_padding(self):
        lst = [FR(1), FR(2), FR(3)]
        result = pad_to_power_of_2(lst)
        assert len(result) == 4
        assert result[3] == FR(0)

    def test_pad_custom_fill(self):
        lst = [FR(1), FR(2), FR(3)]
        result = pad_to_power_of_2(lst, fill=FR(99))
        assert result[3] == FR(99)

    def test_pad_single_element(self):
        lst = [FR(42)]
        result = pad_to_power_of_2(lst)
        assert len(result) == 1
        assert result[0] == FR(42)

    def test_pad_empty(self):
        # next_power_of_2(0) = 1
        lst = []
        result = pad_to_power_of_2(lst)
        assert len(result) == 1
        assert result[0] == FR(0)
