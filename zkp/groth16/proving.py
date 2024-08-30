from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

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

g1 = bn128.G1
g2 = bn128.G2

mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg

pointInf1 = mult(g1, bn128.curve_order) # None
pointInf2 = mult(g2, bn128.curve_order) # None

class FR(FQ):
    field_modulus = bn128.curve_order

# EXAMPLE r, s
# r = FR(4106)
# s = FR(4565)

def proof_a(numWires, numGates, sigma1_1, sigma1_2, Ax, Rx, r):
    proof_A = sigma1_1[0]
    for i in range(numWires):
        temp = pointInf1
        for j in range(numGates):
            temp = add(temp, mult(sigma1_2[j], int(Ax[i][j])))
        proof_A = add(proof_A, mult(temp, int(Rx[i])))
    proof_A = add(proof_A, mult(sigma1_1[2], int(r)))
    return proof_A

def proof_b(numWires, numGates, sigma2_1, sigma2_2, Bx, Rx, s):
    proof_B = sigma2_1[0]
    for i in range(numWires):
        temp = pointInf2
        for j in range(numGates):
            temp = add(temp, mult(sigma2_2[j], int(Bx[i][j])))
        proof_B = add(proof_B, mult(temp, int(Rx[i])))
    proof_B = add(proof_B, mult(sigma2_1[2], int(s)))
    return proof_B

def proof_c(numWires, numGates, sigma1_1, sigma1_2, sigma1_4, sigma1_5, Bx, Rx, Hx, s, r, prf_A):
    print("proof A : {}".format(prf_A))
    #Build temp_proof_B
    temp_proof_B = sigma1_1[1]
    for i in range(numWires):
        temp = pointInf1
        for j in range(numGates):
            temp = add(temp, mult(sigma1_2[j], int(Bx[i][j])))
        temp_proof_B = add(temp_proof_B, mult(temp, int(Rx[i])))
    temp_proof_B = add(temp_proof_B, mult(sigma1_1[2], int(s)))

    #Build proof_C, g1_based
    ## TODO : FIX ERROR HERE
    proof_C = add(add(mult(prf_A, int(s)), mult(temp_proof_B, int(r))), neg(mult(mult(sigma1_1[2], int(s)), int(r))))

    # proof_C = add(add(mult(proof_A, int(s)), mult(temp_proof_B, int(r))), neg(mult(mult(sigma1_1[2], int(s)), int(r))))

    for i in range(1, numWires-1):
        proof_C = add(proof_C, mult(sigma1_4[i], int(Rx[i])))

    for i in range(numGates-1):
        proof_C = add(proof_C, mult(sigma1_5[i], int(Hx[i])))

    return proof_C

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

    def scalar_vec(scalar, vec):
        return [scalar*num for num in vec]
    
    ### PROOF COMPLETENESS CHECK ###
    VAL = [FR(0)]*numWires

    A = alpha + _multiply_vec_vec(Rx, Ax_val) + r*delta
    B = beta + _multiply_vec_vec(Rx, Bx_val) + s*delta

    C0 = 1/delta 
    C1 = Rx[1:numWires-1] #vec
    C1_1 = scalar_vec(beta, Ax_val[1:numWires-1])
    C1_2 = scalar_vec(alpha, Bx_val[1:numWires-1])
    C1_3 = Cx_val[1:numWires-1]
    C2 = Hx_val*Zx_val
    C3 = A*s + B*r - r*s*delta

    C1112 = _add_polys(C1_1, C1_2) # vec
    C111213 = _add_polys(C1112, C1_3) # vec
    C1111213 = _multiply_vec_vec(C1, C111213) #num

    C = C0 * (C1111213 + C2) + C3

    lhs = A*B #21888242871839275222246405745257275088548364400416033032405666501928354297837

    rhs = alpha*beta #14149304

    rpub = [Rx[0], Rx[-1]] #[1, 30]
    valpub = [VAL[0], VAL[-1]]

    rhs = rhs + gamma*_multiply_vec_vec(rpub,valpub) #12058091336480024
    rhs = rhs + C*delta #21888242871839275222246405745257275088548364400414254943408259015785828911597

    print("#PROOF COMPLETENESS CHECK#")
    print("rhs : {}".format(rhs))
    print("lhs : {}".format(lhs))
    print("rhs == lhs ? : {}".format(rhs == lhs))
    print("proof A check : {}".format(proof_A == mult(g1, int(A))))
    print("proof B check : {}".format(proof_B == mult(g2, int(B))))
    print("proof C check : {}".format(proof_C == mult(g1, int(C))))
    print("")