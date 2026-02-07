"""
PLONK circuit, permutation, transcript 모듈 테스트.

테스트 대상:
  - Gate: 생성, check 메서드 (유효/무효)
  - Circuit: 게이트 추가, copy constraint, selector polynomials,
             build_copy_constraints, compute_witness, x3_plus_x_plus_5_eq_35
  - permutation: build_permutation_polynomials, compute_accumulator
  - Transcript: 결정론성, append_scalar, append_point, challenge_scalar
"""

import pytest
from zkp.plonk.field import FR, CURVE_ORDER, G1, ec_mul, get_root_of_unity, get_roots_of_unity
from zkp.plonk.circuit import Gate, Circuit
from zkp.plonk.permutation import (
    K1, K2,
    build_permutation_polynomials,
    compute_accumulator,
)
from zkp.plonk.transcript import Transcript


# ─────────────────────────────────────────────────────────────────────
# Gate 테스트
# ─────────────────────────────────────────────────────────────────────

class TestGate:
    """Gate 클래스 테스트."""

    def test_gate_creation_with_fr(self):
        """FR 원소로 게이트 생성."""
        g = Gate(FR(1), FR(2), FR(3), FR(4), FR(5))
        assert g.q_l == FR(1)
        assert g.q_r == FR(2)
        assert g.q_o == FR(3)
        assert g.q_m == FR(4)
        assert g.q_c == FR(5)

    def test_gate_creation_with_int(self):
        """정수로 게이트 생성 (자동 FR 변환)."""
        g = Gate(1, 2, 3, 4, 5)
        assert g.q_l == FR(1)
        assert g.q_r == FR(2)
        assert g.q_o == FR(3)
        assert g.q_m == FR(4)
        assert g.q_c == FR(5)

    def test_multiplication_gate_check_valid(self):
        """곱셈 게이트: a * b = c 만족."""
        g = Gate(FR(0), FR(0), FR(CURVE_ORDER - 1), FR(1), FR(0))
        assert g.check(FR(3), FR(7), FR(21)) is True

    def test_multiplication_gate_check_invalid(self):
        """곱셈 게이트: a * b != c 불만족."""
        g = Gate(FR(0), FR(0), FR(CURVE_ORDER - 1), FR(1), FR(0))
        assert g.check(FR(3), FR(7), FR(20)) is False

    def test_addition_gate_check_valid(self):
        """덧셈 게이트: a + b = c 만족."""
        g = Gate(FR(1), FR(1), FR(CURVE_ORDER - 1), FR(0), FR(0))
        assert g.check(FR(10), FR(20), FR(30)) is True

    def test_addition_gate_check_invalid(self):
        """덧셈 게이트: a + b != c 불만족."""
        g = Gate(FR(1), FR(1), FR(CURVE_ORDER - 1), FR(0), FR(0))
        assert g.check(FR(10), FR(20), FR(31)) is False

    def test_constant_gate_check_valid(self):
        """상수 덧셈 게이트: a + 5 = c 만족."""
        g = Gate(FR(1), FR(0), FR(CURVE_ORDER - 1), FR(0), FR(5))
        assert g.check(FR(30), FR(0), FR(35)) is True

    def test_constant_gate_check_invalid(self):
        """상수 덧셈 게이트: a + 5 != c 불만족."""
        g = Gate(FR(1), FR(0), FR(CURVE_ORDER - 1), FR(0), FR(5))
        assert g.check(FR(30), FR(0), FR(36)) is False

    def test_public_input_gate_check(self):
        """공개 입력 게이트: q_O * c = 0 (c=0일 때)."""
        g = Gate(FR(0), FR(0), FR(1), FR(0), FR(0))
        # q_O * c = 0 이 되려면 c = 0
        assert g.check(FR(0), FR(0), FR(0)) is True
        # c != 0이면 실패
        assert g.check(FR(0), FR(0), FR(1)) is False

    def test_check_with_int_inputs(self):
        """check에 정수를 넘겨도 자동 변환."""
        g = Gate(FR(0), FR(0), FR(CURVE_ORDER - 1), FR(1), FR(0))
        assert g.check(3, 7, 21) is True

    def test_zero_gate(self):
        """모든 셀렉터가 0인 게이트: 항상 만족."""
        g = Gate(0, 0, 0, 0, 0)
        assert g.check(FR(999), FR(888), FR(777)) is True

    def test_general_gate_equation(self):
        """일반 게이트 방정식: 2a + 3b - c + 4ab + 7 = 0."""
        # 2*1 + 3*2 - c + 4*1*2 + 7 = 0
        # 2 + 6 - c + 8 + 7 = 0  => c = 23
        g = Gate(FR(2), FR(3), FR(CURVE_ORDER - 1), FR(4), FR(7))
        assert g.check(FR(1), FR(2), FR(23)) is True
        assert g.check(FR(1), FR(2), FR(22)) is False


# ─────────────────────────────────────────────────────────────────────
# Circuit 테스트
# ─────────────────────────────────────────────────────────────────────

class TestCircuit:
    """Circuit 클래스 테스트."""

    def test_empty_circuit(self):
        """빈 회로 초기화."""
        c = Circuit()
        assert c.n == 0
        assert c.gates == []
        assert c.copy_constraints == []
        assert c.num_public_inputs == 0

    def test_add_multiplication_gate(self):
        """곱셈 게이트 추가."""
        c = Circuit()
        idx = c.add_multiplication_gate()
        assert idx == 0
        assert c.n == 1
        gate = c.gates[0]
        # a * b = c  =>  q_M*ab + q_O*c = 0  => q_M=1, q_O=-1
        assert gate.q_l == FR(0)
        assert gate.q_r == FR(0)
        assert gate.q_o == FR(CURVE_ORDER - 1)
        assert gate.q_m == FR(1)
        assert gate.q_c == FR(0)
        # 실제 확인
        assert gate.check(FR(3), FR(5), FR(15)) is True

    def test_add_addition_gate(self):
        """덧셈 게이트 추가."""
        c = Circuit()
        idx = c.add_addition_gate()
        assert idx == 0
        assert c.n == 1
        gate = c.gates[0]
        assert gate.q_l == FR(1)
        assert gate.q_r == FR(1)
        assert gate.q_o == FR(CURVE_ORDER - 1)
        assert gate.q_m == FR(0)
        assert gate.q_c == FR(0)
        assert gate.check(FR(10), FR(20), FR(30)) is True

    def test_add_constant_gate(self):
        """상수 덧셈 게이트 추가."""
        c = Circuit()
        idx = c.add_constant_gate(5)
        assert idx == 0
        gate = c.gates[0]
        assert gate.q_l == FR(1)
        assert gate.q_r == FR(0)
        assert gate.q_o == FR(CURVE_ORDER - 1)
        assert gate.q_m == FR(0)
        assert gate.q_c == FR(5)
        assert gate.check(FR(30), FR(0), FR(35)) is True

    def test_add_constant_gate_with_fr(self):
        """FR 값으로 상수 게이트 추가."""
        c = Circuit()
        c.add_constant_gate(FR(42))
        assert c.gates[0].q_c == FR(42)

    def test_add_public_input_gate(self):
        """공개 입력 게이트 추가."""
        c = Circuit()
        idx = c.add_public_input_gate()
        assert idx == 0
        assert c.num_public_inputs == 1
        gate = c.gates[0]
        assert gate.q_l == FR(0)
        assert gate.q_r == FR(0)
        assert gate.q_o == FR(1)
        assert gate.q_m == FR(0)
        assert gate.q_c == FR(0)

    def test_multiple_public_input_gates(self):
        """공개 입력 게이트 여러 개 추가 시 카운터 증가."""
        c = Circuit()
        c.add_public_input_gate()
        c.add_public_input_gate()
        assert c.num_public_inputs == 2

    def test_gate_index_increments(self):
        """게이트 추가 시 인덱스 순차 증가."""
        c = Circuit()
        assert c.add_multiplication_gate() == 0
        assert c.add_addition_gate() == 1
        assert c.add_constant_gate(5) == 2
        assert c.add_public_input_gate() == 3
        assert c.n == 4

    def test_add_copy_constraint(self):
        """복사 제약 추가."""
        c = Circuit()
        c.add_multiplication_gate()
        c.add_multiplication_gate()
        c.add_copy_constraint(0, 2, 1, 0)  # gate0.c == gate1.a
        assert len(c.copy_constraints) == 1
        assert c.copy_constraints[0] == (0, 2, 1, 0)

    def test_add_multiple_copy_constraints(self):
        """여러 복사 제약 추가."""
        c = Circuit()
        for _ in range(3):
            c.add_multiplication_gate()
        c.add_copy_constraint(0, 2, 1, 0)
        c.add_copy_constraint(1, 2, 2, 0)
        assert len(c.copy_constraints) == 2

    def test_get_selector_polynomials(self):
        """셀렉터 다항식 벡터 반환."""
        c = Circuit()
        c.add_multiplication_gate()
        c.add_addition_gate()
        c.add_constant_gate(5)

        q_l, q_r, q_o, q_m, q_c = c.get_selector_polynomials()

        assert len(q_l) == 3
        # 곱셈 게이트: q_l=0, q_r=0, q_o=-1, q_m=1, q_c=0
        assert q_l[0] == FR(0)
        assert q_m[0] == FR(1)
        assert q_o[0] == FR(CURVE_ORDER - 1)

        # 덧셈 게이트: q_l=1, q_r=1, q_o=-1, q_m=0
        assert q_l[1] == FR(1)
        assert q_r[1] == FR(1)
        assert q_m[1] == FR(0)

        # 상수 게이트: q_c=5
        assert q_c[2] == FR(5)

    def test_get_selector_polynomials_empty(self):
        """빈 회로의 셀렉터 다항식."""
        c = Circuit()
        q_l, q_r, q_o, q_m, q_c = c.get_selector_polynomials()
        assert all(len(q) == 0 for q in [q_l, q_r, q_o, q_m, q_c])

    def test_build_copy_constraints_identity(self):
        """복사 제약이 없을 때 항등 순열."""
        c = Circuit()
        c.add_multiplication_gate()
        c.add_addition_gate()
        sigma = c.build_copy_constraints()
        n = c.n
        assert len(sigma) == 3 * n
        # 항등 순열: sigma[i] = i
        for i in range(3 * n):
            assert sigma[i] == i

    def test_build_copy_constraints_single(self):
        """단일 복사 제약: gate0.c == gate1.a."""
        c = Circuit()
        c.add_multiplication_gate()
        c.add_multiplication_gate()
        c.add_copy_constraint(0, 2, 1, 0)  # gate0.c(pos=2*2+0=4) == gate1.a(pos=0*2+1=1)
        sigma = c.build_copy_constraints()
        n = c.n  # 2
        # pos1 = w1*n + g1 = 2*2 + 0 = 4 (c wire, gate 0)
        # pos2 = w2*n + g2 = 0*2 + 1 = 1 (a wire, gate 1)
        # swap: sigma[4], sigma[1] = sigma[1], sigma[4]
        # 원래 sigma[1]=1, sigma[4]=4 => sigma[1]=4, sigma[4]=1
        assert sigma[1] == 4
        assert sigma[4] == 1
        # 나머지는 항등
        assert sigma[0] == 0
        assert sigma[2] == 2
        assert sigma[3] == 3
        assert sigma[5] == 5

    def test_build_copy_constraints_cycle(self):
        """여러 복사 제약으로 순환 구성."""
        c = Circuit()
        c.add_multiplication_gate()  # gate 0
        c.add_multiplication_gate()  # gate 1
        c.add_addition_gate()        # gate 2
        n = 3
        # gate0.a == gate0.b (둘 다 x)
        c.add_copy_constraint(0, 0, 0, 1)
        sigma = c.build_copy_constraints()
        # pos1 = 0*3 + 0 = 0, pos2 = 1*3 + 0 = 3
        # swap: sigma[0]=3, sigma[3]=0
        assert sigma[0] == 3
        assert sigma[3] == 0

    def test_compute_witness_raises(self):
        """기본 Circuit.compute_witness는 NotImplementedError."""
        c = Circuit()
        c.add_multiplication_gate()
        with pytest.raises(NotImplementedError):
            c.compute_witness({"x": FR(3)})


# ─────────────────────────────────────────────────────────────────────
# x3_plus_x_plus_5_eq_35 테스트
# ─────────────────────────────────────────────────────────────────────

class TestX3PlusXPlus5Eq35:
    """x^3 + x + 5 = 35 예제 회로 테스트."""

    @pytest.fixture
    def example_circuit(self):
        return Circuit.x3_plus_x_plus_5_eq_35()

    def test_returns_five_elements(self, example_circuit):
        """반환 값: (circuit, a_vals, b_vals, c_vals, public_inputs)."""
        assert len(example_circuit) == 5

    def test_circuit_has_4_gates(self, example_circuit):
        """4개의 게이트."""
        circuit = example_circuit[0]
        assert circuit.n == 4

    def test_witness_values(self, example_circuit):
        """witness 값 확인: x=3."""
        _, a_vals, b_vals, c_vals, _ = example_circuit
        # gate 0: 3*3 = 9
        assert a_vals[0] == FR(3)
        assert b_vals[0] == FR(3)
        assert c_vals[0] == FR(9)
        # gate 1: 9*3 = 27
        assert a_vals[1] == FR(9)
        assert b_vals[1] == FR(3)
        assert c_vals[1] == FR(27)
        # gate 2: 27+3 = 30
        assert a_vals[2] == FR(27)
        assert b_vals[2] == FR(3)
        assert c_vals[2] == FR(30)
        # gate 3: 30+5 = 35
        assert a_vals[3] == FR(30)
        assert b_vals[3] == FR(0)
        assert c_vals[3] == FR(35)

    def test_all_gates_satisfied(self, example_circuit):
        """모든 게이트 제약이 만족."""
        circuit, a_vals, b_vals, c_vals, _ = example_circuit
        for i, gate in enumerate(circuit.gates):
            assert gate.check(a_vals[i], b_vals[i], c_vals[i]), \
                f"Gate {i} not satisfied"

    def test_public_inputs(self, example_circuit):
        """공개 입력은 [FR(35)]."""
        _, _, _, _, public_inputs = example_circuit
        assert len(public_inputs) == 1
        assert public_inputs[0] == FR(35)

    def test_num_public_inputs(self, example_circuit):
        """num_public_inputs == 1."""
        circuit = example_circuit[0]
        assert circuit.num_public_inputs == 1

    def test_copy_constraints_count(self, example_circuit):
        """6개의 copy constraints."""
        circuit = example_circuit[0]
        assert len(circuit.copy_constraints) == 6

    def test_copy_constraints_enforce_x_sharing(self, example_circuit):
        """x 값 공유 제약: gate0.a, gate0.b, gate1.b, gate2.b."""
        circuit, a_vals, b_vals, _, _ = example_circuit
        # 모두 x=3
        assert a_vals[0] == FR(3)  # gate0.a
        assert b_vals[0] == FR(3)  # gate0.b
        assert b_vals[1] == FR(3)  # gate1.b
        assert b_vals[2] == FR(3)  # gate2.b

    def test_copy_constraints_enforce_x2_sharing(self, example_circuit):
        """x^2 공유: gate0.c == gate1.a."""
        _, a_vals, _, c_vals, _ = example_circuit
        assert c_vals[0] == a_vals[1] == FR(9)

    def test_copy_constraints_enforce_x3_sharing(self, example_circuit):
        """x^3 공유: gate1.c == gate2.a."""
        _, a_vals, _, c_vals, _ = example_circuit
        assert c_vals[1] == a_vals[2] == FR(27)

    def test_copy_constraints_enforce_sum_sharing(self, example_circuit):
        """x^3+x 공유: gate2.c == gate3.a."""
        _, a_vals, _, c_vals, _ = example_circuit
        assert c_vals[2] == a_vals[3] == FR(30)

    def test_gate_types(self, example_circuit):
        """게이트 유형 확인: mul, mul, add, const."""
        circuit = example_circuit[0]
        gates = circuit.gates
        # gate 0, 1: multiplication (q_m=1, q_o=-1)
        for i in [0, 1]:
            assert gates[i].q_m == FR(1)
            assert gates[i].q_o == FR(CURVE_ORDER - 1)
            assert gates[i].q_l == FR(0)
        # gate 2: addition (q_l=1, q_r=1, q_o=-1)
        assert gates[2].q_l == FR(1)
        assert gates[2].q_r == FR(1)
        assert gates[2].q_o == FR(CURVE_ORDER - 1)
        assert gates[2].q_m == FR(0)
        # gate 3: constant (q_l=1, q_o=-1, q_c=5)
        assert gates[3].q_l == FR(1)
        assert gates[3].q_c == FR(5)
        assert gates[3].q_o == FR(CURVE_ORDER - 1)


# ─────────────────────────────────────────────────────────────────────
# Permutation 테스트
# ─────────────────────────────────────────────────────────────────────

class TestBuildPermutationPolynomials:
    """build_permutation_polynomials 테스트."""

    def test_identity_permutation(self):
        """항등 순열에서 S_sigma는 원래 코셋 값."""
        n = 4
        domain = get_roots_of_unity(n)
        sigma = list(range(3 * n))  # 항등 순열

        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

        # S_sigma1: a 배선 → 1·domain[i]
        for i in range(n):
            assert s1[i] == domain[i]
        # S_sigma2: b 배선 → K1·domain[i]
        for i in range(n):
            assert s2[i] == K1 * domain[i]
        # S_sigma3: c 배선 → K2·domain[i]
        for i in range(n):
            assert s3[i] == K2 * domain[i]

    def test_simple_swap(self):
        """단순 스왑: a[0] <-> b[0]."""
        n = 2
        domain = get_roots_of_unity(n)
        sigma = list(range(6))
        # swap pos 0 (a[0]) and pos 2 (b[0])
        sigma[0], sigma[2] = sigma[2], sigma[0]

        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

        # a[0]은 이제 pos 2 → b 배선의 index 0 → K1 * domain[0]
        assert s1[0] == K1 * domain[0]
        # b[0]은 이제 pos 0 → a 배선의 index 0 → domain[0]
        assert s2[0] == domain[0]
        # 나머지는 변경 없음
        assert s1[1] == domain[1]
        assert s2[1] == K1 * domain[1]

    def test_x3_circuit_permutation(self):
        """x^3+x+5=35 회로의 순열 다항식."""
        circuit, _, _, _, _ = Circuit.x3_plus_x_plus_5_eq_35()
        n = circuit.n  # 4
        domain = get_roots_of_unity(n)
        sigma = circuit.build_copy_constraints()

        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

        assert len(s1) == n
        assert len(s2) == n
        assert len(s3) == n

    def test_output_length(self):
        """출력 리스트 길이는 n."""
        n = 4
        domain = get_roots_of_unity(n)
        sigma = list(range(3 * n))
        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)
        assert len(s1) == n
        assert len(s2) == n
        assert len(s3) == n

    def test_all_values_in_coset(self):
        """모든 S_sigma 값은 코셋 원소."""
        n = 4
        domain = get_roots_of_unity(n)
        sigma = list(range(3 * n))
        # 임의의 순열 만들기
        sigma[0], sigma[4] = sigma[4], sigma[0]
        sigma[2], sigma[8] = sigma[8], sigma[2]

        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

        # 모든 값은 domain, K1*domain, K2*domain 중 하나
        all_coset_values = set()
        for d in domain:
            all_coset_values.add(int(d))
            all_coset_values.add(int(K1 * d))
            all_coset_values.add(int(K2 * d))

        for s in [s1, s2, s3]:
            for v in s:
                assert int(v) in all_coset_values


class TestComputeAccumulator:
    """compute_accumulator 테스트."""

    @pytest.fixture
    def x3_circuit_data(self):
        circuit, a_vals, b_vals, c_vals, _ = Circuit.x3_plus_x_plus_5_eq_35()
        n = circuit.n
        domain = get_roots_of_unity(n)
        sigma = circuit.build_copy_constraints()
        beta = FR(31)
        gamma = FR(47)
        return a_vals, b_vals, c_vals, sigma, n, domain, beta, gamma

    def test_z_starts_at_one(self, x3_circuit_data):
        """z[0] = 1."""
        z = compute_accumulator(*x3_circuit_data)
        assert z[0] == FR(1)

    def test_z_length(self, x3_circuit_data):
        """z의 길이는 n."""
        a_vals, b_vals, c_vals, sigma, n, domain, beta, gamma = x3_circuit_data
        z = compute_accumulator(*x3_circuit_data)
        assert len(z) == n

    def test_product_check_with_identity(self):
        """항등 순열에서 전체 곱 = 1."""
        n = 4
        domain = get_roots_of_unity(n)
        # 임의의 witness 값
        a_vals = [FR(1), FR(2), FR(3), FR(4)]
        b_vals = [FR(5), FR(6), FR(7), FR(8)]
        c_vals = [FR(9), FR(10), FR(11), FR(12)]
        sigma = list(range(3 * n))  # 항등 순열
        beta = FR(13)
        gamma = FR(17)

        z = compute_accumulator(a_vals, b_vals, c_vals, sigma, n, domain, beta, gamma)

        assert z[0] == FR(1)
        # 항등 순열에서는 분자 == 분모이므로 모든 z[i] = 1
        for val in z:
            assert val == FR(1)

    def test_z_last_product_equals_one(self, x3_circuit_data):
        """올바른 순열/witness에서 마지막 누적 곱까지 계산하면 1이 되어야 함."""
        a_vals, b_vals, c_vals, sigma, n, domain, beta, gamma = x3_circuit_data
        # compute_accumulator는 z[0]..z[n-1]까지 반환
        # 전체 곱 확인: z[n-1]에 마지막 분수를 곱하면 1이 되어야 함
        z = compute_accumulator(*x3_circuit_data)

        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

        # 마지막 인덱스 i = n-1
        i = n - 1
        num = (
            (a_vals[i] + beta * domain[i] + gamma)
            * (b_vals[i] + beta * K1 * domain[i] + gamma)
            * (c_vals[i] + beta * K2 * domain[i] + gamma)
        )
        den = (
            (a_vals[i] + beta * s1[i] + gamma)
            * (b_vals[i] + beta * s2[i] + gamma)
            * (c_vals[i] + beta * s3[i] + gamma)
        )
        final_product = z[n - 1] * num / den
        assert final_product == FR(1)

    def test_accumulator_with_different_challenges(self, x3_circuit_data):
        """다른 챌린지 값으로도 z[0]=1, 곱=1 성립."""
        a_vals, b_vals, c_vals, sigma, n, domain, _, _ = x3_circuit_data
        beta2 = FR(100)
        gamma2 = FR(200)
        z = compute_accumulator(a_vals, b_vals, c_vals, sigma, n, domain, beta2, gamma2)
        assert z[0] == FR(1)

        # 전체 곱 확인
        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)
        i = n - 1
        num = (
            (a_vals[i] + beta2 * domain[i] + gamma2)
            * (b_vals[i] + beta2 * K1 * domain[i] + gamma2)
            * (c_vals[i] + beta2 * K2 * domain[i] + gamma2)
        )
        den = (
            (a_vals[i] + beta2 * s1[i] + gamma2)
            * (b_vals[i] + beta2 * s2[i] + gamma2)
            * (c_vals[i] + beta2 * s3[i] + gamma2)
        )
        final = z[n - 1] * num / den
        assert final == FR(1)

    def test_wrong_witness_breaks_product(self):
        """잘못된 witness에서는 전체 곱 != 1."""
        circuit, a_vals, b_vals, c_vals, _ = Circuit.x3_plus_x_plus_5_eq_35()
        n = circuit.n
        domain = get_roots_of_unity(n)
        sigma = circuit.build_copy_constraints()
        beta = FR(7)
        gamma = FR(11)

        # a_vals를 변조: gate0.a를 3 -> 999로 변경 (copy constraint 깨짐)
        bad_a = list(a_vals)
        bad_a[0] = FR(999)

        z = compute_accumulator(bad_a, b_vals, c_vals, sigma, n, domain, beta, gamma)
        s1, s2, s3 = build_permutation_polynomials(sigma, n, domain)

        i = n - 1
        num = (
            (bad_a[i] + beta * domain[i] + gamma)
            * (b_vals[i] + beta * K1 * domain[i] + gamma)
            * (c_vals[i] + beta * K2 * domain[i] + gamma)
        )
        den = (
            (bad_a[i] + beta * s1[i] + gamma)
            * (b_vals[i] + beta * s2[i] + gamma)
            * (c_vals[i] + beta * s3[i] + gamma)
        )
        final = z[n - 1] * num / den
        assert final != FR(1)


# ─────────────────────────────────────────────────────────────────────
# Transcript 테스트
# ─────────────────────────────────────────────────────────────────────

class TestTranscript:
    """Transcript 클래스 테스트."""

    def test_determinism(self):
        """동일한 입력 → 동일한 챌린지."""
        t1 = Transcript()
        t1.append_scalar(b"val", FR(42))
        c1 = t1.challenge_scalar(b"c")

        t2 = Transcript()
        t2.append_scalar(b"val", FR(42))
        c2 = t2.challenge_scalar(b"c")

        assert c1 == c2

    def test_different_inputs_different_challenges(self):
        """다른 입력 → 다른 챌린지."""
        t1 = Transcript()
        t1.append_scalar(b"val", FR(42))
        c1 = t1.challenge_scalar(b"c")

        t2 = Transcript()
        t2.append_scalar(b"val", FR(43))
        c2 = t2.challenge_scalar(b"c")

        assert c1 != c2

    def test_different_labels_different_challenges(self):
        """다른 레이블 → 다른 챌린지."""
        t1 = Transcript()
        t1.append_scalar(b"val_a", FR(42))
        c1 = t1.challenge_scalar(b"c")

        t2 = Transcript()
        t2.append_scalar(b"val_b", FR(42))
        c2 = t2.challenge_scalar(b"c")

        assert c1 != c2

    def test_challenge_chaining(self):
        """연속 챌린지는 서로 다름."""
        t = Transcript()
        t.append_scalar(b"val", FR(1))
        c1 = t.challenge_scalar(b"first")
        c2 = t.challenge_scalar(b"second")
        assert c1 != c2

    def test_challenge_is_fr(self):
        """챌린지는 FR 타입."""
        t = Transcript()
        c = t.challenge_scalar(b"c")
        assert isinstance(c, FR)

    def test_challenge_nonzero(self):
        """챌린지는 일반적으로 0이 아님."""
        t = Transcript()
        t.append_scalar(b"x", FR(12345))
        c = t.challenge_scalar(b"c")
        assert c != FR(0)

    def test_append_scalar(self):
        """append_scalar 후 상태 변경 확인."""
        t = Transcript()
        initial_len = len(t.state)
        t.append_scalar(b"test", FR(100))
        # 레이블(4 bytes) + 스칼라(32 bytes)
        assert len(t.state) == initial_len + 4 + 32

    def test_append_point(self):
        """append_point 후 상태 변경 확인."""
        t = Transcript()
        initial_len = len(t.state)
        point = ec_mul(G1, 5)
        t.append_point(b"pt", point)
        # 레이블(2 bytes) + x좌표(32) + y좌표(32)
        assert len(t.state) == initial_len + 2 + 64

    def test_append_point_none(self):
        """무한원점(None) 추가."""
        t = Transcript()
        initial_len = len(t.state)
        t.append_point(b"pt", None)
        # 레이블(2 bytes) + 64바이트 0
        assert len(t.state) == initial_len + 2 + 64

    def test_append_point_none_deterministic(self):
        """무한원점 추가도 결정론적."""
        t1 = Transcript()
        t1.append_point(b"p", None)
        c1 = t1.challenge_scalar(b"c")

        t2 = Transcript()
        t2.append_point(b"p", None)
        c2 = t2.challenge_scalar(b"c")

        assert c1 == c2

    def test_append_point_vs_none_different(self):
        """실제 점과 무한원점은 다른 챌린지."""
        t1 = Transcript()
        t1.append_point(b"p", G1)
        c1 = t1.challenge_scalar(b"c")

        t2 = Transcript()
        t2.append_point(b"p", None)
        c2 = t2.challenge_scalar(b"c")

        assert c1 != c2

    def test_custom_label(self):
        """커스텀 레이블로 초기화."""
        t1 = Transcript(label=b"plonk")
        t2 = Transcript(label=b"other")
        c1 = t1.challenge_scalar(b"c")
        c2 = t2.challenge_scalar(b"c")
        assert c1 != c2

    def test_default_label(self):
        """기본 레이블은 b'plonk'."""
        t = Transcript()
        assert bytes(t.state) == b"plonk"

    def test_order_matters(self):
        """데이터 추가 순서가 다르면 다른 챌린지."""
        t1 = Transcript()
        t1.append_scalar(b"a", FR(1))
        t1.append_scalar(b"b", FR(2))
        c1 = t1.challenge_scalar(b"c")

        t2 = Transcript()
        t2.append_scalar(b"b", FR(2))
        t2.append_scalar(b"a", FR(1))
        c2 = t2.challenge_scalar(b"c")

        assert c1 != c2

    def test_challenge_scalar_updates_state(self):
        """challenge_scalar가 상태를 업데이트."""
        t = Transcript()
        len_before = len(t.state)
        t.challenge_scalar(b"beta")
        # 레이블 + SHA-256 해시(32 bytes) 추가
        assert len(t.state) > len_before

    def test_plonk_round_simulation(self):
        """PLONK 라운드 시뮬레이션: 여러 append 후 challenge."""
        t = Transcript()
        # Round 1: 커밋먼트 추가
        t.append_point(b"a_comm", ec_mul(G1, 3))
        t.append_point(b"b_comm", ec_mul(G1, 5))
        t.append_point(b"c_comm", ec_mul(G1, 7))
        beta = t.challenge_scalar(b"beta")
        gamma = t.challenge_scalar(b"gamma")

        assert isinstance(beta, FR)
        assert isinstance(gamma, FR)
        assert beta != gamma

    def test_scalar_value_in_field(self):
        """챌린지 값은 필드 범위 내."""
        t = Transcript()
        for i in range(10):
            c = t.challenge_scalar(f"c{i}".encode())
            assert int(c) < CURVE_ORDER
            assert int(c) >= 0
