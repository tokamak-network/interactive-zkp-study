"""
Tests for PLONK cryptographic modules: SRS, KZG, Preprocessor.

Covers:
- SRS generation (deterministic, correct lengths, max_degree)
- KZG commit (known polynomial, zero polynomial, degree overflow)
- KZG create_witness + verify_opening (valid/invalid proofs)
- KZG linearity: commit(a+b) == commit(a) + commit(b)
- Preprocessor: domain setup, selector/permutation commitments
"""

import pytest
from zkp.plonk.field import FR, G1, G2, Z1, ec_mul, ec_add, ec_neg, CURVE_ORDER
from zkp.plonk.polynomial import Polynomial
from zkp.plonk.srs import SRS
from zkp.plonk.kzg import commit, create_witness, verify_opening
from zkp.plonk.preprocessor import preprocess, PreprocessedData
from zkp.plonk.circuit import Circuit


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def srs_small():
    """Small SRS for fast tests (max_degree=8)."""
    return SRS.generate(max_degree=8, seed=42)


@pytest.fixture
def srs_medium():
    """Medium SRS for circuit tests (max_degree=32)."""
    return SRS.generate(max_degree=32, seed=1234)


# ─────────────────────────────────────────────────────────────────────
# SRS Tests
# ─────────────────────────────────────────────────────────────────────

class TestSRS:
    """SRS.generate 테스트."""

    def test_generate_g1_powers_length(self, srs_small):
        """g1_powers length == max_degree + 1."""
        assert len(srs_small.g1_powers) == 9  # 8 + 1

    def test_generate_g2_powers_length(self, srs_small):
        """g2_powers length == 2 ([G2, tau*G2])."""
        assert len(srs_small.g2_powers) == 2

    def test_generate_max_degree(self, srs_small):
        """max_degree is stored correctly."""
        assert srs_small.max_degree == 8

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical SRS."""
        srs1 = SRS.generate(max_degree=4, seed=99)
        srs2 = SRS.generate(max_degree=4, seed=99)
        assert srs1.g1_powers == srs2.g1_powers
        assert srs1.g2_powers == srs2.g2_powers

    def test_different_seeds_produce_different_srs(self):
        """Different seeds produce different SRS."""
        srs1 = SRS.generate(max_degree=4, seed=1)
        srs2 = SRS.generate(max_degree=4, seed=2)
        assert srs1.g1_powers != srs2.g1_powers

    def test_g1_first_element_is_generator(self, srs_small):
        """g1_powers[0] == G1 (tau^0 * G1 = G1)."""
        assert srs_small.g1_powers[0] == G1

    def test_g2_first_element_is_generator(self, srs_small):
        """g2_powers[0] == G2."""
        assert srs_small.g2_powers[0] == G2

    def test_g1_powers_not_at_infinity(self, srs_small):
        """All g1_powers should be valid curve points (not infinity)."""
        for i, pt in enumerate(srs_small.g1_powers):
            assert pt is not None, f"g1_powers[{i}] is at infinity"

    def test_g2_powers_not_at_infinity(self, srs_small):
        """Both g2_powers should be valid curve points."""
        for i, pt in enumerate(srs_small.g2_powers):
            assert pt is not None, f"g2_powers[{i}] is at infinity"

    def test_consecutive_g1_powers_are_distinct(self, srs_small):
        """Consecutive g1_powers should be distinct (tau != 1)."""
        for i in range(len(srs_small.g1_powers) - 1):
            assert srs_small.g1_powers[i] != srs_small.g1_powers[i + 1]

    def test_generate_without_seed(self):
        """SRS generation without seed should work (random tau)."""
        srs = SRS.generate(max_degree=2)
        assert len(srs.g1_powers) == 3
        assert len(srs.g2_powers) == 2

    def test_max_degree_zero(self):
        """SRS with max_degree=0 should have just [G1] and [G2, tau*G2]."""
        srs = SRS.generate(max_degree=0, seed=42)
        assert len(srs.g1_powers) == 1
        assert srs.g1_powers[0] == G1
        assert srs.max_degree == 0


# ─────────────────────────────────────────────────────────────────────
# KZG Commit Tests
# ─────────────────────────────────────────────────────────────────────

class TestKZGCommit:
    """KZG commit 함수 테스트."""

    def test_commit_constant_polynomial(self, srs_small):
        """commit(c) == c * G1 for constant polynomial."""
        c = FR(7)
        poly = Polynomial([c])
        C = commit(poly, srs_small)
        expected = ec_mul(G1, c)
        assert C == expected

    def test_commit_linear_polynomial(self, srs_small):
        """commit(a + bx) == a*G1 + b*tau*G1."""
        a, b = FR(3), FR(5)
        poly = Polynomial([a, b])
        C = commit(poly, srs_small)
        expected = ec_add(
            ec_mul(srs_small.g1_powers[0], a),
            ec_mul(srs_small.g1_powers[1], b),
        )
        assert C == expected

    def test_commit_zero_polynomial(self, srs_small):
        """commit(0) should return the point at infinity (Z1/None)."""
        poly = Polynomial([FR(0)])
        C = commit(poly, srs_small)
        assert C is None or C == Z1

    def test_commit_degree_exceeds_max_degree(self, srs_small):
        """commit should raise ValueError if degree > max_degree."""
        # srs_small has max_degree=8, make a degree-9 polynomial
        coeffs = [FR(1)] * 10  # degree 9
        poly = Polynomial(coeffs)
        with pytest.raises(ValueError):
            commit(poly, srs_small)

    def test_commit_at_max_degree(self, srs_small):
        """commit should work for polynomial at exactly max_degree."""
        coeffs = [FR(0)] * 8 + [FR(1)]  # degree 8
        poly = Polynomial(coeffs)
        C = commit(poly, srs_small)
        assert C is not None

    def test_commit_scalar_multiplication(self, srs_small):
        """commit(s * p) == s * commit(p) for scalar s."""
        poly = Polynomial([FR(2), FR(3)])
        s = FR(5)
        C1 = commit(poly * s, srs_small)
        C2 = ec_mul(commit(poly, srs_small), s)
        assert C1 == C2


# ─────────────────────────────────────────────────────────────────────
# KZG Linearity Tests
# ─────────────────────────────────────────────────────────────────────

class TestKZGLinearity:
    """KZG commitment linearity: commit(a+b) == commit(a) + commit(b)."""

    def test_linearity_simple(self, srs_small):
        """commit(p+q) == commit(p) + commit(q)."""
        p = Polynomial([FR(1), FR(2)])
        q = Polynomial([FR(3), FR(4)])
        C_sum = commit(p + q, srs_small)
        C_separate = ec_add(commit(p, srs_small), commit(q, srs_small))
        assert C_sum == C_separate

    def test_linearity_different_degrees(self, srs_small):
        """Linearity holds for polynomials of different degrees."""
        p = Polynomial([FR(1)])                    # degree 0
        q = Polynomial([FR(0), FR(0), FR(5)])      # degree 2
        C_sum = commit(p + q, srs_small)
        C_separate = ec_add(commit(p, srs_small), commit(q, srs_small))
        assert C_sum == C_separate

    def test_linearity_with_zero(self, srs_small):
        """commit(p + 0) == commit(p)."""
        p = Polynomial([FR(3), FR(7)])
        zero = Polynomial([FR(0)])
        C1 = commit(p, srs_small)
        C2 = commit(p + zero, srs_small)
        assert C1 == C2


# ─────────────────────────────────────────────────────────────────────
# KZG Opening Proof Tests
# ─────────────────────────────────────────────────────────────────────

class TestKZGOpening:
    """KZG create_witness + verify_opening 테스트."""

    def test_valid_opening_constant(self, srs_small):
        """Valid opening for constant polynomial p(x) = 5."""
        poly = Polynomial([FR(5)])
        point = FR(7)
        evaluation = poly.evaluate(point)
        C = commit(poly, srs_small)
        proof = create_witness(poly, point, srs_small)
        assert verify_opening(C, proof, point, evaluation, srs_small)

    def test_valid_opening_linear(self, srs_small):
        """Valid opening for p(x) = 1 + 2x at x=3."""
        poly = Polynomial([FR(1), FR(2)])
        point = FR(3)
        evaluation = poly.evaluate(point)  # 1 + 6 = 7
        assert evaluation == FR(7)
        C = commit(poly, srs_small)
        proof = create_witness(poly, point, srs_small)
        assert verify_opening(C, proof, point, evaluation, srs_small)

    def test_valid_opening_quadratic(self, srs_small):
        """Valid opening for p(x) = 1 + x + x^2 at x=2."""
        poly = Polynomial([FR(1), FR(1), FR(1)])
        point = FR(2)
        evaluation = poly.evaluate(point)  # 1 + 2 + 4 = 7
        assert evaluation == FR(7)
        C = commit(poly, srs_small)
        proof = create_witness(poly, point, srs_small)
        assert verify_opening(C, proof, point, evaluation, srs_small)

    def test_valid_opening_at_zero(self, srs_small):
        """Valid opening at x=0 (returns constant term)."""
        poly = Polynomial([FR(42), FR(3), FR(5)])
        point = FR(0)
        evaluation = poly.evaluate(point)
        assert evaluation == FR(42)
        C = commit(poly, srs_small)
        proof = create_witness(poly, point, srs_small)
        assert verify_opening(C, proof, point, evaluation, srs_small)

    def test_invalid_opening_wrong_evaluation(self, srs_small):
        """Verification fails if claimed evaluation is wrong."""
        poly = Polynomial([FR(1), FR(2)])
        point = FR(3)
        correct_eval = poly.evaluate(point)  # 7
        wrong_eval = correct_eval + FR(1)    # 8
        C = commit(poly, srs_small)
        proof = create_witness(poly, point, srs_small)
        assert not verify_opening(C, proof, point, wrong_eval, srs_small)

    def test_invalid_opening_wrong_point(self, srs_small):
        """Verification fails if proof was for point1 but we claim evaluation at point2.

        We use point1's evaluation with point2, so the pairing check should fail.
        """
        poly = Polynomial([FR(1), FR(2)])
        point1 = FR(3)
        point2 = FR(5)
        C = commit(poly, srs_small)
        proof = create_witness(poly, point1, srs_small)
        eval_at_1 = poly.evaluate(point1)  # correct for point1, wrong for point2
        assert not verify_opening(C, proof, point2, eval_at_1, srs_small)

    def test_invalid_opening_wrong_commitment(self, srs_small):
        """Verification fails if commitment is for a different polynomial."""
        poly1 = Polynomial([FR(1), FR(2)])
        poly2 = Polynomial([FR(3), FR(4)])
        point = FR(5)
        C_wrong = commit(poly2, srs_small)
        proof = create_witness(poly1, point, srs_small)
        eval1 = poly1.evaluate(point)
        assert not verify_opening(C_wrong, proof, point, eval1, srs_small)

    def test_opening_high_degree(self, srs_small):
        """Valid opening for a higher degree polynomial."""
        # degree 5: 1 + x + x^2 + x^3 + x^4 + x^5
        coeffs = [FR(1)] * 6
        poly = Polynomial(coeffs)
        point = FR(2)
        evaluation = poly.evaluate(point)  # 1+2+4+8+16+32=63
        assert evaluation == FR(63)
        C = commit(poly, srs_small)
        proof = create_witness(poly, point, srs_small)
        assert verify_opening(C, proof, point, evaluation, srs_small)

    def test_opening_with_integer_point(self, srs_small):
        """create_witness and verify_opening accept int for point."""
        poly = Polynomial([FR(1), FR(2)])
        C = commit(poly, srs_small)
        proof = create_witness(poly, 3, srs_small)
        evaluation = poly.evaluate(FR(3))
        assert verify_opening(C, proof, 3, evaluation, srs_small)

    def test_opening_multiple_points(self, srs_small):
        """Verify opening at multiple different points for same polynomial."""
        poly = Polynomial([FR(2), FR(3), FR(1)])  # 2 + 3x + x^2
        C = commit(poly, srs_small)
        for z in [FR(0), FR(1), FR(5), FR(100)]:
            evaluation = poly.evaluate(z)
            proof = create_witness(poly, z, srs_small)
            assert verify_opening(C, proof, z, evaluation, srs_small), \
                f"Opening failed at point {z}"


# ─────────────────────────────────────────────────────────────────────
# Preprocessor Tests
# ─────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    """preprocess 함수 테스트."""

    @pytest.fixture
    def circuit_and_srs(self):
        """x^3 + x + 5 = 35 circuit with SRS."""
        circuit, a, b, c, pub = Circuit.x3_plus_x_plus_5_eq_35()
        srs = SRS.generate(max_degree=32, seed=1234)
        return circuit, srs, a, b, c, pub

    def test_returns_preprocessed_data(self, circuit_and_srs):
        """preprocess returns PreprocessedData instance."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert isinstance(pp, PreprocessedData)

    def test_domain_size_power_of_2(self, circuit_and_srs):
        """n should be a power of 2."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.n > 0
        assert (pp.n & (pp.n - 1)) == 0  # power of 2 check

    def test_domain_size_at_least_gate_count(self, circuit_and_srs):
        """n >= original gate count (4 gates)."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.n >= 4

    def test_omega_nth_power_is_one(self, circuit_and_srs):
        """omega^n == 1 (primitive n-th root of unity)."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.omega ** pp.n == FR(1)

    def test_omega_is_primitive(self, circuit_and_srs):
        """omega^k != 1 for 0 < k < n."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        power = FR(1)
        for k in range(1, pp.n):
            power = power * pp.omega
            assert power != FR(1), f"omega^{k} == 1, not primitive"

    def test_domain_length(self, circuit_and_srs):
        """domain has n elements."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert len(pp.domain) == pp.n

    def test_domain_elements_are_powers_of_omega(self, circuit_and_srs):
        """domain[i] == omega^i."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        for i in range(pp.n):
            assert pp.domain[i] == pp.omega ** i

    def test_selector_polynomials_exist(self, circuit_and_srs):
        """All 5 selector polynomials should be present."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert isinstance(pp.q_l_poly, Polynomial)
        assert isinstance(pp.q_r_poly, Polynomial)
        assert isinstance(pp.q_o_poly, Polynomial)
        assert isinstance(pp.q_m_poly, Polynomial)
        assert isinstance(pp.q_c_poly, Polynomial)

    def test_selector_commitments_exist(self, circuit_and_srs):
        """All 5 selector commitments should be G1 points (not None for non-zero selectors)."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        # q_l, q_o, q_m have non-zero entries so commitments should be non-None
        assert pp.q_l_comm is not None
        assert pp.q_o_comm is not None
        assert pp.q_m_comm is not None

    def test_selector_commitments_match_polynomials(self, circuit_and_srs):
        """Each selector commitment == commit(selector_poly, srs)."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.q_l_comm == commit(pp.q_l_poly, srs)
        assert pp.q_r_comm == commit(pp.q_r_poly, srs)
        assert pp.q_o_comm == commit(pp.q_o_poly, srs)
        assert pp.q_m_comm == commit(pp.q_m_poly, srs)
        assert pp.q_c_comm == commit(pp.q_c_poly, srs)

    def test_permutation_polynomials_exist(self, circuit_and_srs):
        """All 3 permutation polynomials should be present."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert isinstance(pp.s_sigma1_poly, Polynomial)
        assert isinstance(pp.s_sigma2_poly, Polynomial)
        assert isinstance(pp.s_sigma3_poly, Polynomial)

    def test_permutation_commitments_exist(self, circuit_and_srs):
        """All 3 permutation commitments should be non-None."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.s_sigma1_comm is not None
        assert pp.s_sigma2_comm is not None
        assert pp.s_sigma3_comm is not None

    def test_permutation_commitments_match_polynomials(self, circuit_and_srs):
        """Each permutation commitment == commit(perm_poly, srs)."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.s_sigma1_comm == commit(pp.s_sigma1_poly, srs)
        assert pp.s_sigma2_comm == commit(pp.s_sigma2_poly, srs)
        assert pp.s_sigma3_comm == commit(pp.s_sigma3_poly, srs)

    def test_sigma_length(self, circuit_and_srs):
        """sigma permutation array has length 3n."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert len(pp.sigma) == 3 * pp.n

    def test_sigma_is_valid_permutation(self, circuit_and_srs):
        """sigma should be a valid permutation of [0, 3n)."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert sorted(pp.sigma) == list(range(3 * pp.n))

    def test_num_public_inputs(self, circuit_and_srs):
        """num_public_inputs should match circuit."""
        circuit, srs, *_ = circuit_and_srs
        pp = preprocess(circuit, srs)
        assert pp.num_public_inputs == 1

    def test_selector_polynomials_evaluate_correctly(self, circuit_and_srs):
        """Selector polynomials should evaluate to the correct values on domain."""
        circuit, srs, a, b, c, pub = circuit_and_srs
        pp = preprocess(circuit, srs)
        # Gate 0 is multiplication: q_L=0, q_R=0, q_O=-1, q_M=1, q_C=0
        assert pp.q_l_poly.evaluate(pp.domain[0]) == FR(0)
        assert pp.q_m_poly.evaluate(pp.domain[0]) == FR(1)
        # Gate 2 is addition: q_L=1, q_R=1, q_O=-1, q_M=0
        assert pp.q_l_poly.evaluate(pp.domain[2]) == FR(1)
        assert pp.q_r_poly.evaluate(pp.domain[2]) == FR(1)
        assert pp.q_m_poly.evaluate(pp.domain[2]) == FR(0)

    def test_preprocess_idempotent_commitment(self, circuit_and_srs):
        """Running preprocess twice gives the same commitments."""
        circuit1, a1, b1, c1, pub1 = Circuit.x3_plus_x_plus_5_eq_35()
        circuit2, a2, b2, c2, pub2 = Circuit.x3_plus_x_plus_5_eq_35()
        srs = SRS.generate(max_degree=32, seed=1234)
        pp1 = preprocess(circuit1, srs)
        pp2 = preprocess(circuit2, srs)
        assert pp1.q_l_comm == pp2.q_l_comm
        assert pp1.s_sigma1_comm == pp2.s_sigma1_comm


# ─────────────────────────────────────────────────────────────────────
# Cross-module integration: SRS + KZG + Preprocessor
# ─────────────────────────────────────────────────────────────────────

class TestCryptoIntegration:
    """SRS, KZG, Preprocessor 간 통합 테스트."""

    def test_preprocessed_selector_poly_opening(self):
        """Can create and verify opening proof for preprocessed selector polynomials."""
        circuit, a, b, c, pub = Circuit.x3_plus_x_plus_5_eq_35()
        srs = SRS.generate(max_degree=32, seed=1234)
        pp = preprocess(circuit, srs)

        # Open q_l_poly at a random point
        zeta = FR(17)
        evaluation = pp.q_l_poly.evaluate(zeta)
        proof = create_witness(pp.q_l_poly, zeta, srs)
        assert verify_opening(pp.q_l_comm, proof, zeta, evaluation, srs)

    def test_preprocessed_permutation_poly_opening(self):
        """Can create and verify opening proof for permutation polynomials."""
        circuit, a, b, c, pub = Circuit.x3_plus_x_plus_5_eq_35()
        srs = SRS.generate(max_degree=32, seed=1234)
        pp = preprocess(circuit, srs)

        zeta = FR(23)
        evaluation = pp.s_sigma1_poly.evaluate(zeta)
        proof = create_witness(pp.s_sigma1_poly, zeta, srs)
        assert verify_opening(pp.s_sigma1_comm, proof, zeta, evaluation, srs)

    def test_commit_from_evaluations_polynomial(self):
        """Polynomial built from evaluations can be committed and opened."""
        from zkp.plonk.field import get_root_of_unity
        n = 4
        omega = get_root_of_unity(n)
        evals = [FR(1), FR(2), FR(3), FR(4)]
        poly = Polynomial.from_evaluations(evals, omega)

        srs = SRS.generate(max_degree=8, seed=42)
        C = commit(poly, srs)
        assert C is not None

        zeta = FR(11)
        evaluation = poly.evaluate(zeta)
        proof = create_witness(poly, zeta, srs)
        assert verify_opening(C, proof, zeta, evaluation, srs)
