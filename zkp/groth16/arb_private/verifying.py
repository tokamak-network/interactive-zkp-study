from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

# from random import randint

from poly_utils import (
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

from setup import (
    sigma11,
    sigma12,
    sigma13,
    sigma14,
    sigma15,
    sigma21,
    sigma22
)

from proving import (
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

def verify(numWires, prf_A, proof_B, proof_C, sigma1_1, sigma1_3, sigma2_1, Rx, pub_r_indexs=None):
    if pub_r_indexs == None:
        pub_r_indexs = [0, numWires-1]

    LHS = pairing(proof_B, prf_A)
    RHS = pairing(sigma2_1[0], sigma1_1[0])

    temp = None

    for i in pub_r_indexs:
        temp = add(temp, mult(sigma1_3[i], int(Rx[i])))

    RHS = (RHS * pairing(sigma2_1[1], temp)) * pairing(sigma2_1[2], proof_C)

    return LHS == RHS