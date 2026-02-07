import sys
import os
import pytest

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from zkp.groth16.code_to_r1cs import (
    code_to_r1cs_with_inputs, initialize_symbol
)
from zkp.groth16.qap_creator_lcm import (
    r1cs_to_qap_times_lcm,
    create_solution_polynomials,
    create_divisor_polynomial,
)
from zkp.groth16.poly_utils import (
    getNumWires, getNumGates,
    getFRPoly1D, getFRPoly2D,
    ax_val, bx_val, cx_val, zx_val, hx_val, hxr,
)
from zkp.groth16.setup import (
    sigma11, sigma12, sigma13, sigma14, sigma15,
    sigma21, sigma22,
)
from zkp.groth16.proving import proof_a, proof_b, proof_c, build_rpub_enum
from zkp.groth16.verifying import verify


class FR(FQ):
    field_modulus = bn128.curve_order


# ── 테스트 상수 ──
TEST_CODE = """
def qeval(x):
    y = x**3
    return y + x + 5
"""
TEST_INPUT_VARS = [3]
EXPECTED_R = [1, 3, 35, 9, 27, 30]

TOXIC_ALPHA = 3926
TOXIC_BETA = 3604
TOXIC_GAMMA = 2971
TOXIC_DELTA = 1357
TOXIC_X_VAL = 3721

PROVER_R = 4106
PROVER_S = 4565

PUB_R_INDEXS = [0, 1]


@pytest.fixture(scope="session")
def r1cs_data():
    """R1CS 변환 결과를 반환하는 fixture."""
    initialize_symbol()
    r, A, B, C = code_to_r1cs_with_inputs(TEST_CODE, TEST_INPUT_VARS)
    return {"r": r, "A": A, "B": B, "C": C}


@pytest.fixture(scope="session")
def qap_data(r1cs_data):
    """QAP 변환 결과를 반환하는 fixture."""
    A, B, C = r1cs_data["A"], r1cs_data["B"], r1cs_data["C"]
    r = r1cs_data["r"]
    Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
    Apoly, Bpoly, Cpoly, sol = create_solution_polynomials(r, Ap, Bp, Cp)
    H = create_divisor_polynomial(sol, Z)
    return {
        "Ap": Ap, "Bp": Bp, "Cp": Cp, "Z": Z,
        "Apoly": Apoly, "Bpoly": Bpoly, "Cpoly": Cpoly,
        "sol": sol, "H": H, "r": r,
    }


@pytest.fixture(scope="session")
def full_pipeline_data(qap_data):
    """전체 파이프라인 데이터 (setup → proving → verifying)."""
    alpha = FR(TOXIC_ALPHA)
    beta = FR(TOXIC_BETA)
    gamma = FR(TOXIC_GAMMA)
    delta = FR(TOXIC_DELTA)
    x_val = FR(TOXIC_X_VAL)

    Ap = qap_data["Ap"]
    Bp = qap_data["Bp"]
    Cp = qap_data["Cp"]
    Z = qap_data["Z"]
    R = qap_data["r"]

    Ax = getFRPoly2D(Ap)
    Bx = getFRPoly2D(Bp)
    Cx = getFRPoly2D(Cp)
    Zx = getFRPoly1D(Z)
    Rx = getFRPoly1D(R)

    Hx, remainder = hxr(Ax, Bx, Cx, Zx, R)
    Hx_val_result = hx_val(Hx, x_val)

    numGates = getNumGates(Ax)
    numWires = getNumWires(Ax)

    Ax_val_result = ax_val(Ax, x_val)
    Bx_val_result = bx_val(Bx, x_val)
    Cx_val_result = cx_val(Cx, x_val)
    Zx_val_result = zx_val(Zx, x_val)

    s11 = sigma11(alpha, beta, delta)
    s12 = sigma12(numGates, x_val)
    s13, VAL = sigma13(numWires, alpha, beta, gamma,
                       Ax_val_result, Bx_val_result, Cx_val_result,
                       pub_r_indexs=PUB_R_INDEXS)
    s14 = sigma14(numWires, alpha, beta, delta,
                  Ax_val_result, Bx_val_result, Cx_val_result,
                  pub_r_indexs=PUB_R_INDEXS)
    s15 = sigma15(numGates, delta, x_val, Zx_val_result)
    s21 = sigma21(beta, delta, gamma)
    s22 = sigma22(numGates, x_val)

    r_prover = FR(PROVER_R)
    s_prover = FR(PROVER_S)

    prf_A = proof_a(s11, s12, Ax, Rx, r_prover)
    prf_B = proof_b(s21, s22, Bx, Rx, s_prover)
    prf_C = proof_c(s11, s12, s14, s15, Bx, Rx, Hx, s_prover, r_prover, prf_A,
                    pub_r_indexs=PUB_R_INDEXS)

    rx_pub = build_rpub_enum(PUB_R_INDEXS, Rx)

    return {
        # toxic waste
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "delta": delta, "x_val": x_val,
        # polynomials
        "Ax": Ax, "Bx": Bx, "Cx": Cx, "Zx": Zx, "Rx": Rx,
        "Hx": Hx, "remainder": remainder,
        # evaluated values
        "Ax_val": Ax_val_result, "Bx_val": Bx_val_result,
        "Cx_val": Cx_val_result, "Zx_val": Zx_val_result,
        "Hx_val": Hx_val_result,
        # dimensions
        "numGates": numGates, "numWires": numWires,
        # sigma
        "s11": s11, "s12": s12, "s13": s13, "s14": s14, "s15": s15,
        "s21": s21, "s22": s22, "VAL": VAL,
        # prover
        "r_prover": r_prover, "s_prover": s_prover,
        "prf_A": prf_A, "prf_B": prf_B, "prf_C": prf_C,
        "rx_pub": rx_pub,
        # raw R1CS r
        "R": R,
    }
