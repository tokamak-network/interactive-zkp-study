from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

class FR(FQ):
    field_modulus = bn128.curve_order


def multiply_polys(a, b):
    o = [FR(0)] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            o[i + j] += a[i] * b[j]
    return o

def add_polys(a, b, subtract=False):
    o = [FR(0)] * max(len(a), len(b))
    for i in range(len(a)):
        o[i] += a[i]
    for i in range(len(b)):
        o[i] += b[i] * (FR(-1) if subtract else FR(1))
    return o

def subtract_polys(a, b):
    return add_polys(a, b, subtract=True)

def div_polys(a, b):
    o = [FR(0)] * (len(a) - len(b) + 1)
    remainder = a
    while len(remainder) >= len(b):
        if b[-1] == 0:
            raise ZeroDivisionError("Division by zero polynomial")
        leading_fac = remainder[-1] / b[-1]
        pos = len(remainder) - len(b)
        o[pos] = leading_fac
        remainder = subtract_polys(remainder, multiply_polys(b, [FR(0)] * pos + [leading_fac]))[:-1]
    return o, remainder

def eval_poly(poly, x):
    return sum([poly[i] * x**i for i in range(len(poly))])

def mk_singleton(point_loc, height, total_pts):
    fac = FR(1)
    for i in range(1, total_pts + 1):
        if i != point_loc:
            fac *= FR(point_loc - i)
    o = [FR(height) / fac]
    for i in range(1, total_pts + 1):
        if i != point_loc:
            o = multiply_polys(o, [FR(-i), FR(1)])
    return o

def lagrange_interp(vec):
    o = []
    for i in range(len(vec)):
        o = add_polys(o, mk_singleton(i + 1, FR(vec[i]), len(vec)))
    for i in range(len(vec)):
        assert eval_poly(o, i + 1) == vec[i], \
            (o, eval_poly(o, i + 1), i+1)
    return o

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def r1cs_to_qap(A, B, C):
    A, B, C = transpose(A), transpose(B), transpose(C)
    new_A = [lagrange_interp([FR(a) for a in row]) for row in A]
    new_B = [lagrange_interp([FR(b) for b in row]) for row in B]
    new_C = [lagrange_interp([FR(c) for c in row]) for row in C]
    Z = [FR(1)]
    for i in range(1, len(A[0]) + 1):
        Z = multiply_polys(Z, [FR(-i), FR(1)])
    return (new_A, new_B, new_C, Z)

def create_solution_polynomials(r, new_A, new_B, new_C):
    Apoly = []
    for rval, a in zip(r, new_A):
        Apoly = add_polys(Apoly, multiply_polys([FR(rval)], a))

    Bpoly = []
    for rval, b in zip(r, new_B):
        Bpoly = add_polys(Bpoly, multiply_polys([FR(rval)], b))

    Cpoly = []
    for rval, c in zip(r, new_C):
        Cpoly = add_polys(Cpoly, multiply_polys([FR(rval)], c))

    sol = subtract_polys(multiply_polys(Apoly, Bpoly), Cpoly)
    #return Apoly, Bpoly, Cpoly, o
    return sol

def create_divisor_polynomial(sol, Z):
    quot, rem = div_polys(sol, Z)
    return quot