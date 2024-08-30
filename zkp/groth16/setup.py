from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

g1 = bn128.G1
g2 = bn128.G2

mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg

class FR(FQ):
    field_modulus = bn128.curve_order

# Multiply two polynomials
def _multiply_polys(a, b):
    o = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            o[i + j] += a[i] * b[j]
    return o

# Add two polynomials
def _add_polys(a, b, subtract=False):
    o = [0] * max(len(a), len(b))
    for i in range(len(a)):
        o[i] += a[i]
    for i in range(len(b)):
        o[i] += b[i] * (-1 if subtract else 1) # Reuse the function structure for subtraction
    return o

def _subtract_polys(a, b):
    return _add_polys(a, b, subtract=True)

# Divide a/b, return quotient and remainder
def _div_polys(a, b):
    o = [0] * (len(a) - len(b) + 1)
    remainder = a
    while len(remainder) >= len(b):
        leading_fac = remainder[-1] / b[-1]
        pos = len(remainder) - len(b)
        o[pos] = leading_fac
        remainder = _subtract_polys(remainder, _multiply_polys(b, [0] * pos + [leading_fac]))[:-1]
    return o, remainder

# Evaluate a polynomial at a point
def _eval_poly(poly, x):
    return sum([poly[i] * x**i for i in range(len(poly))])

# Multiply Vector * Matrix
def _multiply_vec_matrix(vec, matrix):
    # len(vec) == len(matrix.row)
    assert not len(vec) == len(matrix[0])
    target = [FR(0)]*len(vec)
    for i in range(len(matrix)): #loop num of rows == size of vec, 0-5
        for j in range(len(matrix[0])): #loop num of columns, 0-3
            target[j] = target[j] + vec[i] * matrix[i][j]
    return target

def _multiply_vec_vec(vec1, vec2):
    assert len(vec1) == len(vec2)
    target = 0
    size = len(vec1)
    for i in range(size):
        target += vec1[i]*vec2[i]
    return target

def getNumWires(Ax):
    return len(Ax)

def getNumGates(Ax):
    return len(Ax[0])

def getFRPoly1D(poly):
    return [ FR(int(num)) for num in poly ]

def getFRPoly2D(poly):
    return [ [FR(int(num)) for num in vec] for vec in poly ]

def ax_val(Ax, x_val):
    Ax_val = []
    for i in range(len(Ax)):
        ax_single = _eval_poly(Ax[i], x_val)
        Ax_val.append(ax_single)
    return Ax_val

def bx_val(Bx, x_val):
    Bx_val = []
    for i in range(len(Bx)):
        bx_single = _eval_poly(Bx[i], x_val)
        Bx_val.append(bx_single)
    return Bx_val

def cx_val(Cx, x_val):
    Cx_val = []
    for i in range(len(Cx)):
        cx_single = _eval_poly(Cx[i], x_val)
        Cx_val.append(cx_single)
    return Cx_val

def zx_val(Zx, x_val):
    return _eval_poly(Zx, x_val)

def hxr(Ax, Bx, Cx, Zx, R):
    Rax = _multiply_vec_matrix(R, Ax)
    Rbx = _multiply_vec_matrix(R, Bx)
    Rcx = _multiply_vec_matrix(R, Cx)
    Px = _subtract_polys(_multiply_polys(Rax, Rbx), Rcx)

    q, r = _div_polys(Px, Zx)
    Hx = q

    return Hx, r

def hx_val(Hx, x_val):
    return _eval_poly(Hx, x_val)

def sigma11(alpha, beta, delta):
    return [mult(g1, int(alpha)), mult(g1, int(beta)), mult(g1, int(delta))]

def sigma12(numGates, x_val):
    sigma1_2 = []
    for i in range(numGates):
        val = x_val ** i
        sigma1_2.append(mult(g1, int(val)))
    return sigma1_2

def sigma13(numWires, alpha, beta, gamma, Ax_val, Bx_val, Cx_val):
    sigma1_3 = []
    VAL = [FR(0)]*numWires
    for i in range(numWires):
        if i in [0, numWires-1]:
            val = (beta*Ax_val[i] + alpha*Bx_val[i] + Cx_val[i]) / gamma
            VAL[i] = val
            sigma1_3.append(mult(g1, int(val)))
        else:
            sigma1_3.append((FQ(0), FQ(0)))

def sigma14(numWires, alpha, beta, delta, Ax_val, Bx_val, Cx_val):
    sigma1_4 = []
    for i in range(numWires):
        if i in [0, numWires-1]:
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

    # Ax = [ [FR(int(num)) for num in vec] for vec in Ap ]
    # Bx = [ [FR(int(num)) for num in vec] for vec in Bp ]
    # Cx = [ [FR(int(num)) for num in vec] for vec in Cp ]
    # Zx = [ FR(int(num)) for num in Z ]
    # Rx = [ FR(int(num)) for num in R ]

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