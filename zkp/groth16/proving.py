from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from poly_utils import (
    getNumWires,
    getNumGates
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

def proof_a(sigma1_1, sigma1_2, Ax, Rx, r):
    numGates = getNumGates(Ax)
    numWires = getNumWires(Ax)
    proof_A = sigma1_1[0]
    for i in range(numWires):
        temp = pointInf1
        for j in range(numGates):
            temp = add(temp, mult(sigma1_2[j], int(Ax[i][j])))
        proof_A = add(proof_A, mult(temp, int(Rx[i])))
    proof_A = add(proof_A, mult(sigma1_1[2], int(r)))
    return proof_A

def proof_b(sigma2_1, sigma2_2, Bx, Rx, s):
    numGates = getNumGates(Bx)
    numWires = getNumWires(Bx)
    proof_B = sigma2_1[0]
    for i in range(numWires):
        temp = pointInf2
        for j in range(numGates):
            temp = add(temp, mult(sigma2_2[j], int(Bx[i][j])))
        proof_B = add(proof_B, mult(temp, int(Rx[i])))
    proof_B = add(proof_B, mult(sigma2_1[2], int(s)))
    return proof_B

def proof_c(sigma1_1, sigma1_2, sigma1_4, sigma1_5, Bx, Rx, Hx, s, r, prf_A, pub_r_indexs=None):

    if pub_r_indexs == None:
        pub_r_indexs = [0, 1]
    
    numGates = getNumGates(Bx)
    numWires = getNumWires(Bx)
    #Build temp_proof_B
    temp_proof_B = sigma1_1[1]
    for i in range(numWires):
        temp = pointInf1
        for j in range(numGates):
            temp = add(temp, mult(sigma1_2[j], int(Bx[i][j])))
        temp_proof_B = add(temp_proof_B, mult(temp, int(Rx[i])))
    temp_proof_B = add(temp_proof_B, mult(sigma1_1[2], int(s)))

    #Build proof_C, g1_based
    proof_C = add(add(mult(prf_A, int(s)), mult(temp_proof_B, int(r))), neg(mult(mult(sigma1_1[2], int(s)), int(r))))

    for i in range(numWires):
        if i in pub_r_indexs:
            # print("proof_c i {}".format(i))
            continue
        proof_C = add(proof_C, mult(sigma1_4[i], int(Rx[i])))

    for i in range(numGates-1):
        proof_C = add(proof_C, mult(sigma1_5[i], int(Hx[i])))

    return proof_C

def build_rpub_enum(pub_r_indexs, r_vec):
    o = []
    for i in pub_r_indexs:
        o.append((i, r_vec[i]))
    return o