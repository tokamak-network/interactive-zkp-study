from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

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

def proof_c(numWires, numGates, sigma1_1, sigma1_2, sigma1_4, sigma1_5, Bx, Rx, Hx, s, r, proof_A):
    #Build temp_proof_B
    temp_proof_B = sigma1_1[1]
    for i in range(numWires):
        temp = pointInf1
        for j in range(numGates):
            temp = add(temp, mult(sigma1_2[j], int(Bx[i][j])))
        temp_proof_B = add(temp_proof_B, mult(temp, int(Rx[i])))
    temp_proof_B = add(temp_proof_B, mult(sigma1_1[2], int(s)))

    #Build proof_C, g1_based
    proof_C = add(add(mult(proof_A, int(s)), mult(temp_proof_B, int(r))), neg(mult(mult(sigma1_1[2], int(s)), int(r))))

    for i in range(1, numWires-1):
        proof_C = add(proof_C, mult(sigma1_4[i], int(Rx[i])))

    for i in range(numGates-1):
        proof_C = add(proof_C, mult(sigma1_5[i], int(Hx[i])))

    return proof_C

def test():
    return