from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

g1 = bn128.G1
g2 = bn128.G2

class FR(FQ):
    field_modulus = bn128.curve_order

# Elliptic Curve operations
mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg


def lhs(prf_A, prf_B):
    return pairing(prf_B, prf_A)

def rhs(prf_C, sigma1_1, sigma1_3, sigma2_1, rx_pub):
    RHS = pairing(sigma2_1[0], sigma1_1[0])
    temp = None
    for i, ri in rx_pub:
        temp = add(temp, mult(sigma1_3[i], int(ri)))
    RHS = (RHS * pairing(sigma2_1[1], temp)) * pairing(sigma2_1[2], prf_C)
    return RHS

#(rx_pub) = [(index_i, ri), ... ]
def verify(prf_A, prf_B, prf_C, sigma1_1, sigma1_3, sigma2_1, rx_pub):

    LHS = pairing(prf_B, prf_A)
    RHS = pairing(sigma2_1[0], sigma1_1[0])

    temp = None

    for i, ri in rx_pub:
        temp = add(temp, mult(sigma1_3[i], int(ri)))

    RHS = (RHS * pairing(sigma2_1[1], temp)) * pairing(sigma2_1[2], prf_C)

    return LHS == RHS