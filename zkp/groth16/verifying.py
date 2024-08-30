from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from random import randint

g1 = bn128.G1
g2 = bn128.G2

# FR = FQ
# FR.field_modulus = bn128.curve_order

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