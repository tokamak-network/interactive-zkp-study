import pytest
from zkp.groth16.code_to_r1cs import (
    parse, extract_inputs_and_body, flatten_body, flatten_stmt,
    flatten_expr, mksymbol, initialize_symbol,
    insert_var, get_var_placement, flatcode_to_r1cs,
    grab_var, assign_variables, code_to_r1cs_with_inputs,
)


@pytest.fixture(autouse=True)
def reset_symbol():
    """매 테스트 전 전역 심볼 카운터 초기화."""
    initialize_symbol()


# ── parse ──
class TestParse:
    def test_returns_ast_body(self):
        code = "x = 1"
        result = parse(code)
        assert len(result) == 1

    def test_function_def(self):
        import ast
        code = "def foo(x):\n    return x"
        result = parse(code)
        assert isinstance(result[0], ast.FunctionDef)


# ── extract_inputs_and_body ──
class TestExtractInputsAndBody:
    def test_single_input(self):
        code = parse("def foo(x):\n    y = x\n    return y")
        inputs, body = extract_inputs_and_body(code)
        assert inputs == ['x']
        assert len(body) == 2

    def test_multiple_inputs(self):
        code = parse("def foo(x, y):\n    return x")
        inputs, body = extract_inputs_and_body(code)
        assert inputs == ['x', 'y']

    def test_not_function(self):
        code = parse("x = 1")
        with pytest.raises(Exception, match="Expecting function"):
            extract_inputs_and_body(code)


# ── flatten_body / flatten_stmt / flatten_expr ──
class TestFlatten:
    def test_simple_assignment(self):
        code = parse("def f(x):\n    y = x\n    return y")
        _, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        assert flat[0] == ['set', 'y', 'x']
        assert flat[1] == ['set', '~out', 'y']

    def test_addition(self):
        code = parse("def f(x):\n    return x + 5")
        _, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        assert flat[0][0] == '+'
        assert flat[0][1] == '~out'

    def test_multiplication(self):
        code = parse("def f(x):\n    return x * x")
        _, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        assert flat[0][0] == '*'

    def test_power_expansion(self):
        """x**3은 두 번의 곱셈으로 전개"""
        code = parse("def f(x):\n    y = x**3\n    return y")
        _, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        mult_ops = [f for f in flat if f[0] == '*']
        assert len(mult_ops) == 2

    def test_power_zero(self):
        code = parse("def f(x):\n    y = x**0\n    return y")
        _, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        assert flat[0] == ['set', 'y', 1]

    def test_power_one(self):
        code = parse("def f(x):\n    y = x**1\n    return y")
        _, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        assert flat[0] == ['set', 'y', 'x']


# ── mksymbol / initialize_symbol ──
class TestSymbol:
    def test_increments(self):
        s1 = mksymbol()
        s2 = mksymbol()
        assert s1 == 'sym_1'
        assert s2 == 'sym_2'

    def test_reset(self):
        mksymbol()
        initialize_symbol()
        s = mksymbol()
        assert s == 'sym_1'


# ── insert_var ──
class TestInsertVar:
    def test_string_var(self):
        varz = ['~one', 'x', '~out']
        arr = [0, 0, 0]
        insert_var(arr, varz, 'x', {'x': True})
        assert arr == [0, 1, 0]

    def test_string_var_reverse(self):
        varz = ['~one', 'x', '~out']
        arr = [0, 0, 0]
        insert_var(arr, varz, 'x', {'x': True}, reverse=True)
        assert arr == [0, -1, 0]

    def test_int_var(self):
        varz = ['~one', 'x']
        arr = [0, 0]
        insert_var(arr, varz, 5, {})
        assert arr == [5, 0]

    def test_undefined_variable(self):
        varz = ['~one', 'x']
        arr = [0, 0]
        with pytest.raises(Exception, match="before it is set"):
            insert_var(arr, varz, 'x', {})


# ── get_var_placement ──
class TestGetVarPlacement:
    def test_qeval(self):
        """qeval(x): y = x**3; return y + x + 5"""
        code = parse("def qeval(x):\n    y = x**3\n    return y + x + 5")
        inputs, body = extract_inputs_and_body(code)
        flat = flatten_body(body)
        varz = get_var_placement(inputs, flat)
        assert varz[0] == '~one'
        assert 'x' in varz
        assert '~out' in varz


# ── flatcode_to_r1cs ──
class TestFlatcodeToR1cs:
    def test_qeval_dimensions(self):
        code = """
def qeval(x):
    y = x**3
    return y + x + 5
"""
        r, A, B, C = code_to_r1cs_with_inputs(code, [3])
        assert len(A) == len(B) == len(C)  # 같은 수의 게이트
        assert len(A[0]) == len(r)  # 와이어 수 == r 벡터 길이


# ── grab_var ──
class TestGrabVar:
    def test_string_var(self):
        varz = ['~one', 'x', '~out']
        assignment = [1, 3, 35]
        assert grab_var(varz, assignment, 'x') == 3

    def test_int_var(self):
        assert grab_var([], [], 5) == 5

    def test_invalid_var(self):
        with pytest.raises(Exception, match="What kind"):
            grab_var([], [], 3.14)


# ── assign_variables ──
class TestAssignVariables:
    def test_qeval(self):
        code = """
def qeval(x):
    y = x**3
    return y + x + 5
"""
        inputs, body = extract_inputs_and_body(parse(code))
        flat = flatten_body(body)
        r = assign_variables(inputs, [3], flat)
        assert r[0] == 1  # ~one
        assert r[1] == 3  # x
        assert r[2] == 35  # ~out = 27 + 3 + 5


# ── code_to_r1cs_with_inputs (통합) ──
class TestCodeToR1csWithInputs:
    def test_known_values(self):
        code = """
def qeval(x):
    y = x**3
    return y + x + 5
"""
        r, A, B, C = code_to_r1cs_with_inputs(code, [3])
        assert r == [1, 3, 35, 9, 27, 30]

    def test_r1cs_satisfaction(self):
        """A*r . B*r == C*r 확인 (각 게이트)"""
        code = """
def qeval(x):
    y = x**3
    return y + x + 5
"""
        r, A, B, C = code_to_r1cs_with_inputs(code, [3])
        for i in range(len(A)):
            a_dot = sum(A[i][j] * r[j] for j in range(len(r)))
            b_dot = sum(B[i][j] * r[j] for j in range(len(r)))
            c_dot = sum(C[i][j] * r[j] for j in range(len(r)))
            assert a_dot * b_dot == c_dot, f"Gate {i}: {a_dot}*{b_dot} != {c_dot}"
