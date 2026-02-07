"""
PLONK 회로 표현 (Circuit Representation)
==========================================

PLONK 산술화(arithmetization) 시스템의 핵심: 계산을 게이트와 배선으로 표현.

**PLONK 게이트 구조**:
  각 게이트는 3개의 배선(wire) a, b, c와 5개의 셀렉터(selector)로 구성:

    q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C = 0

  - q_L: 왼쪽 입력(a) 계수
  - q_R: 오른쪽 입력(b) 계수
  - q_O: 출력(c) 계수
  - q_M: 곱셈 계수
  - q_C: 상수 항

**게이트 유형별 셀렉터 설정**:
  | 유형    | q_L | q_R | q_O | q_M | q_C | 의미              |
  |---------|-----|-----|-----|-----|-----|-------------------|
  | 곱셈    |  0  |  0  | -1  |  1  |  0  | a·b = c           |
  | 덧셈    |  1  |  1  | -1  |  0  |  0  | a + b = c         |
  | 상수덧셈|  1  |  0  | -1  |  0  |  k  | a + k = c         |
  | 공개입력|  0  |  0  |  0  |  0  |  0  | (PI로 처리)       |

**배선(Copy) 제약**:
  서로 다른 게이트의 배선이 같은 값을 가져야 함을 표현.
  예: 게이트 0의 출력(c)이 게이트 1의 입력(b)으로 사용될 때,
  "게이트0.c == 게이트1.b"라는 제약이 필요하다.
  이는 순열(permutation)로 인코딩된다.

**예제 회로**: x³ + x + 5 = 35 (x = 3)
  | 게이트 | 유형 | a   | b   | c   | 의미               |
  |--------|------|-----|-----|-----|--------------------|
  | 0      | mul  | x=3 | x=3 | x²=9 | x·x = x²         |
  | 1      | mul  | x²=9| x=3 | x³=27| x²·x = x³        |
  | 2      | add  | x³=27| x=3| 30  | x³ + x = 30       |
  | 3      | add+c| 30  | 0   | 35  | 30 + 5 = 35       |

사용 예시:
    >>> circuit = Circuit.x3_plus_x_plus_5_eq_35()
    >>> a, b, c = circuit.compute_witness({"x": FR(3)})
"""

from zkp.plonk.field import FR


class Gate:
    """PLONK 산술 게이트.

    게이트 방정식: q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C = 0

    각 게이트는 회로의 한 "행"에 해당하며,
    셀렉터 값에 따라 덧셈, 곱셈 등 다양한 연산을 표현한다.
    """

    def __init__(self, q_l, q_r, q_o, q_m, q_c):
        """게이트를 생성한다.

        Args:
            q_l: 왼쪽 입력 셀렉터 (FR 원소)
            q_r: 오른쪽 입력 셀렉터
            q_o: 출력 셀렉터
            q_m: 곱셈 셀렉터
            q_c: 상수 셀렉터
        """
        self.q_l = q_l if isinstance(q_l, FR) else FR(q_l)
        self.q_r = q_r if isinstance(q_r, FR) else FR(q_r)
        self.q_o = q_o if isinstance(q_o, FR) else FR(q_o)
        self.q_m = q_m if isinstance(q_m, FR) else FR(q_m)
        self.q_c = q_c if isinstance(q_c, FR) else FR(q_c)

    def check(self, a, b, c):
        """게이트 제약이 만족되는지 확인한다.

        q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C == 0 ?

        Args:
            a, b, c: 배선 값 (FR 원소)

        Returns:
            bool: 제약 만족 여부
        """
        if not isinstance(a, FR):
            a = FR(a)
        if not isinstance(b, FR):
            b = FR(b)
        if not isinstance(c, FR):
            c = FR(c)
        result = (
            self.q_l * a
            + self.q_r * b
            + self.q_o * c
            + self.q_m * (a * b)
            + self.q_c
        )
        return result == FR(0)


class Circuit:
    """PLONK 산술 회로.

    게이트들의 리스트와 배선 연결(copy constraint) 정보를 관리한다.

    속성:
        gates: Gate 객체 리스트
        n: 게이트 수 (2의 거듭제곱으로 패딩됨)
        copy_constraints: (i1, j1, i2, j2) 튜플 리스트
            - 게이트 i1의 j1번째 배선 == 게이트 i2의 j2번째 배선
            - j=0: a(왼쪽), j=1: b(오른쪽), j=2: c(출력)
        num_public_inputs: 공개 입력의 수
    """

    def __init__(self):
        self.gates = []
        self.copy_constraints = []
        self.num_public_inputs = 0

    @property
    def n(self):
        """게이트 수 (패딩 전)."""
        return len(self.gates)

    def add_multiplication_gate(self):
        """곱셈 게이트 추가: a · b = c.

        셀렉터: q_L=0, q_R=0, q_O=-1, q_M=1, q_C=0
        → 0·a + 0·b + (-1)·c + 1·(a·b) + 0 = 0
        → a·b - c = 0  →  a·b = c

        Returns:
            int: 추가된 게이트의 인덱스
        """
        gate = Gate(FR(0), FR(0), FR(CURVE_ORDER - 1), FR(1), FR(0))
        self.gates.append(gate)
        return len(self.gates) - 1

    def add_addition_gate(self):
        """덧셈 게이트 추가: a + b = c.

        셀렉터: q_L=1, q_R=1, q_O=-1, q_M=0, q_C=0
        → 1·a + 1·b + (-1)·c + 0 + 0 = 0
        → a + b - c = 0  →  a + b = c

        Returns:
            int: 추가된 게이트의 인덱스
        """
        gate = Gate(FR(1), FR(1), FR(CURVE_ORDER - 1), FR(0), FR(0))
        self.gates.append(gate)
        return len(self.gates) - 1

    def add_constant_gate(self, constant):
        """상수 덧셈 게이트 추가: a + constant = c.

        셀렉터: q_L=1, q_R=0, q_O=-1, q_M=0, q_C=constant
        → 1·a + 0·b + (-1)·c + 0 + constant = 0
        → a + constant = c

        Args:
            constant: 상수 값 (정수 또는 FR)

        Returns:
            int: 추가된 게이트의 인덱스
        """
        if not isinstance(constant, FR):
            constant = FR(constant)
        gate = Gate(FR(1), FR(0), FR(CURVE_ORDER - 1), FR(0), constant)
        self.gates.append(gate)
        return len(self.gates) - 1

    def add_public_input_gate(self):
        """공개 입력 게이트 추가: c = public_input (PI로 처리).

        PLONK에서 공개 입력은 PI(x) 다항식으로 처리된다.
        게이트 제약: q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C + PI(x) = 0

        공개 입력 게이트: q_L=0, q_R=0, q_O=1, q_M=0, q_C=0
        → c + PI(ωⁱ) = 0  →  c = -PI(ωⁱ)

        PI(ωⁱ)에 -public_value를 넣으면 c = public_value가 된다.

        Returns:
            int: 추가된 게이트의 인덱스
        """
        gate = Gate(FR(0), FR(0), FR(1), FR(0), FR(0))
        self.gates.append(gate)
        self.num_public_inputs += 1
        return len(self.gates) - 1

    def add_copy_constraint(self, gate1, wire1, gate2, wire2):
        """배선 복사 제약 추가: 게이트1.wire1 == 게이트2.wire2.

        PLONK에서 서로 다른 게이트 간에 같은 값을 공유해야 할 때 사용한다.
        순열(permutation)로 인코딩되어 Round 2에서 증명된다.

        Args:
            gate1: 첫 번째 게이트 인덱스
            wire1: 첫 번째 배선 (0=a, 1=b, 2=c)
            gate2: 두 번째 게이트 인덱스
            wire2: 두 번째 배선 (0=a, 1=b, 2=c)

        예시:
            >>> circuit.add_copy_constraint(0, 2, 1, 0)  # 게이트0.c == 게이트1.a
        """
        self.copy_constraints.append((gate1, wire1, gate2, wire2))

    def get_selector_polynomials(self):
        """셀렉터 벡터를 반환한다.

        각 셀렉터의 i번째 원소는 i번째 게이트의 해당 셀렉터 값이다.

        Returns:
            tuple: (q_L, q_R, q_O, q_M, q_C) — 각각 FR 원소 리스트
        """
        q_l = [g.q_l for g in self.gates]
        q_r = [g.q_r for g in self.gates]
        q_o = [g.q_o for g in self.gates]
        q_m = [g.q_m for g in self.gates]
        q_c = [g.q_c for g in self.gates]
        return q_l, q_r, q_o, q_m, q_c

    def build_copy_constraints(self):
        """배선 순열(permutation) σ를 구성한다.

        3n개의 배선 위치 (a₀, a₁, ..., a_{n-1}, b₀, ..., b_{n-1}, c₀, ..., c_{n-1})에
        대해 같은 값을 가져야 하는 위치들을 순환(cycle)으로 연결한다.

        초기 순열: 항등 순열 σ(i) = i
        각 copy constraint마다 관련 위치들을 같은 순환으로 병합한다.

        Returns:
            list[int]: 길이 3n의 순열 배열.
                       sigma[i] = i번째 위치와 연결된 다음 위치
        """
        n = self.n
        # 인덱스 규칙: a의 i번째 = i, b의 i번째 = n+i, c의 i번째 = 2n+i
        sigma = list(range(3 * n))

        for g1, w1, g2, w2 in self.copy_constraints:
            # 위치 계산
            pos1 = w1 * n + g1
            pos2 = w2 * n + g2
            # 순환(cycle) 병합: pos1과 pos2를 같은 순환에 넣는다
            # pos1 → ... → X → pos1 을 pos1 → ... → X → pos2 → ... → Y → pos1으로 변환
            sigma[pos1], sigma[pos2] = sigma[pos2], sigma[pos1]

        return sigma

    def compute_witness(self, assignments):
        """주어진 변수 할당(assignment)에서 배선 값(witness)을 계산한다.

        사용자가 변수 값을 딕셔너리로 제공하면,
        각 게이트의 a, b, c 배선 값을 자동으로 계산한다.

        이 메서드는 서브클래스나 정적 팩토리에서 재정의된다.

        Args:
            assignments: 변수 이름 → FR 값 딕셔너리

        Returns:
            tuple: (a_vals, b_vals, c_vals) — 각각 FR 원소 리스트
        """
        raise NotImplementedError("서브클래스에서 구현하거나 팩토리 메서드를 사용하세요")

    @staticmethod
    def x3_plus_x_plus_5_eq_35():
        """예제 회로: x³ + x + 5 = 35 (x = 3).

        회로 구조:
          게이트 0 (mul): x · x = x²        → 3·3 = 9
          게이트 1 (mul): x² · x = x³       → 9·3 = 27
          게이트 2 (add): x³ + x = x³+x     → 27+3 = 30
          게이트 3 (add+5): (x³+x) + 5 = 35 → 30+5 = 35

        배선 복사 제약 (같은 값을 가져야 하는 배선들):
          - x: 게이트0.a, 게이트0.b, 게이트1.b, 게이트2.b  (모두 x=3)
          - x²: 게이트0.c, 게이트1.a  (모두 9)
          - x³: 게이트1.c, 게이트2.a  (모두 27)
          - x³+x: 게이트2.c, 게이트3.a  (모두 30)

        공개 입력: 35 (게이트 3의 출력값)

        Returns:
            tuple: (circuit, a_vals, b_vals, c_vals, public_inputs)
        """
        circuit = Circuit()

        # 게이트 0: x · x = x²
        circuit.add_multiplication_gate()
        # 게이트 1: x² · x = x³
        circuit.add_multiplication_gate()
        # 게이트 2: x³ + x = 30
        circuit.add_addition_gate()
        # 게이트 3: 30 + 5 = 35
        circuit.add_constant_gate(5)

        # 배선 복사 제약
        # x 값들을 연결: 게이트0.a == 게이트0.b (둘 다 x)
        circuit.add_copy_constraint(0, 0, 0, 1)
        # 게이트0.a == 게이트1.b (둘 다 x)
        circuit.add_copy_constraint(0, 0, 1, 1)
        # 게이트0.a == 게이트2.b (둘 다 x)
        circuit.add_copy_constraint(0, 0, 2, 1)
        # x² 연결: 게이트0.c == 게이트1.a
        circuit.add_copy_constraint(0, 2, 1, 0)
        # x³ 연결: 게이트1.c == 게이트2.a
        circuit.add_copy_constraint(1, 2, 2, 0)
        # x³+x 연결: 게이트2.c == 게이트3.a
        circuit.add_copy_constraint(2, 2, 3, 0)

        # Witness 계산 (x = 3)
        x = FR(3)
        x2 = x * x           # 9
        x3 = x2 * x          # 27
        x3_plus_x = x3 + x   # 30
        result = x3_plus_x + FR(5)  # 35

        a_vals = [x, x2, x3, x3_plus_x]
        b_vals = [x, x, x, FR(0)]
        c_vals = [x2, x3, x3_plus_x, result]

        # 공개 입력: 결과값 35
        # PI(x) 다항식에서 게이트 3 위치에 -35를 넣어
        # 게이트 3 제약: q_L·a + q_C + PI(ω³) = 0
        # → 30 + 5 + (-35) = 0 이 아닌,
        # 게이트 3: 1·30 + 0·0 + (-1)·35 + 0 + 5 = 30 - 35 + 5 = 0 ✓
        # 하지만 공개 입력은 별도 처리 — 여기서는 간단히 35를 공개 입력으로 설정
        public_inputs = [FR(35)]
        circuit.num_public_inputs = 1

        return circuit, a_vals, b_vals, c_vals, public_inputs


# 상수: curve_order (FR에서 -1을 표현하기 위해)
from zkp.plonk.field import CURVE_ORDER
