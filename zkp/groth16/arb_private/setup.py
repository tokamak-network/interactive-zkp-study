from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from poly_utils import (
    # _multiply_polys,
    # _add_polys,
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

g1 = bn128.G1
g2 = bn128.G2

mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg

class FR(FQ):
    field_modulus = bn128.curve_order

def sigma11(alpha, beta, delta):
    return [mult(g1, int(alpha)), mult(g1, int(beta)), mult(g1, int(delta))]

def sigma12(numGates, x_val):
    sigma1_2 = []
    for i in range(numGates):
        val = x_val ** i
        sigma1_2.append(mult(g1, int(val)))
    return sigma1_2

def sigma13(numWires, alpha, beta, gamma, Ax_val, Bx_val, Cx_val, pub_r_indexs=None):
    if pub_r_indexs == None:
        pub_r_indexs = [0, numWires-1]

    print("in sig13 func : {}".format(pub_r_indexs))
    sigma1_3 = []
    VAL = [FR(0)]*numWires
    for i in range(numWires):
        if i in pub_r_indexs:
            val = (beta*Ax_val[i] + alpha*Bx_val[i] + Cx_val[i]) / gamma
            VAL[i] = val
            sigma1_3.append(mult(g1, int(val)))
        else:
            sigma1_3.append((FQ(0), FQ(0)))
    return sigma1_3, VAL

def sigma14(numWires, alpha, beta, delta, Ax_val, Bx_val, Cx_val, pub_r_indexs=None):
    if pub_r_indexs == None:
        pub_r_indexs = [0, numWires-1]

    print("in sig14 func : {}".format(pub_r_indexs))
    sigma1_4 = []
    for i in range(numWires):
        if i in pub_r_indexs:
            sigma1_4.append((FQ(0), FQ(0)))
        else:
            val = (beta*Ax_val[i] + alpha*Bx_val[i] + Cx_val[i]) / delta
            sigma1_4.append(mult(g1, int(val)))
    return sigma1_4

def sigma15(numGates, delta, x_val, Zx_val):
    sigma1_5 = []
    for i in range(numGates-1):
        sigma1_5.append(mult(g1, int((x_val**i * Zx_val) / delta)))
    return sigma1_5

def sigma21(beta, delta, gamma):
    return [mult(g2, int(beta)), mult(g2, int(gamma)), mult(g2, int(delta))]

def sigma22(numGates, x_val):
    sigma2_2 = []
    for i in range(numGates):
        sigma2_2.append(mult(g2, int(x_val**i)))
    return sigma2_2