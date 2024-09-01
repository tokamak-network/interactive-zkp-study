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
    return [ FR(round(num)) for num in poly ]

def getFRPoly2D(poly):
    return [ [FR(round(num)) for num in vec] for vec in poly ]

# def eval_2d_poly(poly, x_val):
#     o = []
#     for i in range(len(poly)):
#         ax_single = _eval_poly(poly[i], x_val)
#         o.append(ax_single)
#     return o

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

def hx_val(Hx, x_val):
    return _eval_poly(Hx, x_val)

# (Ax.R * Bx.R - Cx.R) / Zx = Hx .... r
def hxr(Ax, Bx, Cx, Zx, R):
    Rax = _multiply_vec_matrix(R, Ax)
    Rbx = _multiply_vec_matrix(R, Bx)
    Rcx = _multiply_vec_matrix(R, Cx)
    Px = _subtract_polys(_multiply_polys(Rax, Rbx), Rcx)

    q, r = _div_polys(Px, Zx)
    Hx = q

    return Hx, r

