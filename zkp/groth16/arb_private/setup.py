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
        #TODO : apply auxiliary input size l
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
        #TODO : apply auxiliary input size l
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

def test():
    # EXAMPLE TOXIC
    alpha = FR(3926)
    beta = FR(3604)
    gamma = FR(2971)
    delta = FR(1357)
    x_val = FR(3721)

    # EXAMPLE POLYNOMIAL
    # numGates : 4
    # numWires : 6
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

    Hx, r = hxr(Ax, Bx, Cx, Zx, R)

    Hx_val = hx_val(Hx, x_val)

    numGates = getNumGates(Ax)
    numWires = getNumWires(Ax)
    
    Ax_val = ax_val(Ax, x_val)
    Bx_val = bx_val(Bx, x_val)
    Cx_val = cx_val(Cx, x_val)
    Zx_val = zx_val(Zx, x_val)

    sigma11(alpha, beta, delta)
    sigma12(numGates, x_val)
    sigma13(numWires, alpha, beta, gamma, Ax_val, Bx_val, Cx_val)
    sigma14(numWires, alpha, beta, delta, Ax_val, Bx_val, Cx_val)
    sigma15(numGates, delta, x_val, Zx_val)
    sigma21(beta, delta, gamma)
    sigma22(numGates, x_val)

    #TEST1 : r should be zero
    t1 = (r == [0,0,0,0])

    lhs = _multiply_vec_vec(Rx, Ax_val) * _multiply_vec_vec(Rx, Bx_val) - _multiply_vec_vec(Rx, Cx_val)
    rhs = Zx_val * Hx_val

    #TEST2 : lhs == rhs
    t2 = (lhs == rhs)

    print("TEST1 {}".format(t1))
    print("TEST2 {}".format(t2))