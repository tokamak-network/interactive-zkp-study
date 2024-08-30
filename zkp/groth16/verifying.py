from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

# from random import randint

from .poly_utils import (
    # _multiply_polys,
    _add_polys,
    # _subtract_polys,
    # _div_polys,
    # _eval_poly,
    # _multiply_vec_matrix,
    _multiply_vec_vec,
    getNumWires,
    getNumGates,
    getFRPoly1D,
    getFRPoly2D,
    ax_val,
    bx_val,
    cx_val,
    zx_val,
    hxr,
    hx_val
)

from .setup import (
    sigma11,
    sigma12,
    sigma13,
    sigma14,
    sigma15,
    sigma21,
    sigma22
)

from .proving import (
    proof_a,
    proof_b,
    proof_c
)

g1 = bn128.G1
g2 = bn128.G2

class FR(FQ):
    field_modulus = bn128.curve_order

# Elliptic Curve operations
mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg

def verify(numWires, prf_A, proof_B, proof_C, sigma1_1, sigma1_3, sigma2_1, Rx):
    LHS = pairing(proof_B, prf_A)
    RHS = pairing(sigma2_1[0], sigma1_1[0])

    temp = None

    for i in [0, numWires-1]:
        temp = add(temp, mult(sigma1_3[i], int(Rx[i])))

    RHS = (RHS * pairing(sigma2_1[1], temp)) * pairing(sigma2_1[2], proof_C)

    return LHS == RHS

def test():
    # EXAMPLE TOXIC
    alpha = FR(3926)
    beta = FR(3604)
    gamma = FR(2971)
    delta = FR(1357)
    x_val = FR(3721)

    # EXAMPLE POLYNOMIAL
    Ap = [
        [-60.0, 110.0, -60.0, 10.0],
        [96.0, -136.0, 60.0, -8.0],
        [0.0, 0.0, 0.0, 0.0],
        [-72.0, 114.0, -48.0, 6.0],
        [48.0, -84.0, 42.0, -6.0],
        [-12.0, 22.0, -12.0, 2.0]
    ]
    Bp = [
        [36.0, -62.0, 30.0, -4.0],
        [-24.0, 62.0, -30.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]
    Cp = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-144.0, 264.0, -144.0, 24.0],
        [576.0, -624.0, 216.0, -24.0],
        [-864.0, 1368.0, -576.0, 72.0],
        [576.0, -1008.0, 504.0, -72.0]
    ]
    Z = [3456.0, -7200.0, 5040.0, -1440.0, 144.0]
    R = [1, 3, 35, 9, 27, 30]

    Ax = getFRPoly2D(Ap)
    Bx = getFRPoly2D(Bp)
    Cx = getFRPoly2D(Cp)
    Zx = getFRPoly1D(Z)
    Rx = getFRPoly1D(R)

    Hx, remain = hxr(Ax, Bx, Cx, Zx, R)

    Hx_val = hx_val(Hx, x_val)

    numGates = getNumGates(Ax)
    numWires = getNumWires(Ax)
    
    Ax_val = ax_val(Ax, x_val)
    Bx_val = bx_val(Bx, x_val)
    Cx_val = cx_val(Cx, x_val)
    Zx_val = zx_val(Zx, x_val)

    sigma1_1 = sigma11(alpha, beta, delta)
    sigma1_2 = sigma12(numGates, x_val)
    sigma1_3, VAL = sigma13(numWires, alpha, beta, gamma, Ax_val, Bx_val, Cx_val)
    sigma1_4 = sigma14(numWires, alpha, beta, delta, Ax_val, Bx_val, Cx_val)
    sigma1_5 = sigma15(numGates, delta, x_val, Zx_val)
    sigma2_1 = sigma21(beta, delta, gamma)
    sigma2_2 = sigma22(numGates, x_val)

    # EXAMPLE r, s
    r = FR(4106)
    s = FR(4565)

    proof_A = proof_a(numWires, numGates, sigma1_1, sigma1_2, Ax, Rx, r)
    proof_B = proof_b(numWires, numGates, sigma2_1, sigma2_2, Bx, Rx, s)
    proof_C = proof_c(numWires, numGates, sigma1_1, sigma1_2, sigma1_4, sigma1_5, Bx, Rx, Hx, s, r, proof_A)

    result = verify(numWires, proof_A, proof_B, proof_C, sigma1_1, sigma1_3, sigma2_1, Rx)

    print("Verify LHS == RHS ? {}".format(result))