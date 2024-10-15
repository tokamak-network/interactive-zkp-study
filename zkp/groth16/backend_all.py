##Pure python Groth16 backend implementation
##It is porting code from https://codeocean.com/capsule/8850121/tree/v3

##INSTALL
# $pip install galois py_ecc

##RUN
# python groth16FR.py
import os,sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

from random import randint
from zkp.groth16.qap_creator_lcm_1 import r1cs_to_qap_times_lcm
# from zkp.groth16.qap_creator_lcm_1 import create_solution_polynomials
# from zkp.groth16.qap_creator_lcm_1 import create_divisor_polynomial

g1 = bn128.G1
g2 = bn128.G2

# FR = FQ
# FR.field_modulus = bn128.curve_order

class FR(FQ):
    field_modulus = bn128.curve_order

#TEST pairing
mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg

pointInf1 = mult(g1, bn128.curve_order) # None
pointInf2 = mult(g2, bn128.curve_order) # None


# a = pairing(mult(g2,10),g1)
# b = pairing(g2, mult(g1,10))
# print(a == b)

# Multiply two polynomials
def multiply_polys(a, b):
    o = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            o[i + j] += a[i] * b[j]
    return o

# Add two polynomials
def add_polys(a, b, subtract=False):
    o = [0] * max(len(a), len(b))
    for i in range(len(a)):
        o[i] += a[i]
    for i in range(len(b)):
        o[i] += b[i] * (-1 if subtract else 1) # Reuse the function structure for subtraction
    return o

def subtract_polys(a, b):
    return add_polys(a, b, subtract=True)

# Divide a/b, return quotient and remainder
def div_polys(a, b):
    o = [0] * (len(a) - len(b) + 1)
    remainder = a
    while len(remainder) >= len(b):
        leading_fac = remainder[-1] / b[-1]
        pos = len(remainder) - len(b)
        o[pos] = leading_fac
        remainder = subtract_polys(remainder, multiply_polys(b, [0] * pos + [leading_fac]))[:-1]
    return o, remainder

# Evaluate a polynomial at a point
def eval_poly(poly, x):
    return sum([poly[i] * x**i for i in range(len(poly))])

# Mod q of all matrix elements
def to_matrix_mod(matrix, q):
    target = []
    for row in matrix:
        temp_row = []
        for val in row:
            temp_row.append(int(val) % q)
        target.append(temp_row)
    return target

# #TARGET POLYNOMIAL - TEST 1
# # f(x) = x^3 + x^2 + x + 5
# r = [1, 4, 89, 16, 64, 16, 80, 84]
# A = [[0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 0],
#     [0, 1, 0, 0, 0, 0, 1, 0],
#     [5, 0, 0, 0, 0, 0, 0, 1]]
# B = [[0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0]]
# C = [[0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0, 0, 0]]

#TARGET POLYNOMIAL - TEST 2
# f(x) = x^3 + x + 5

# r = [1, 3, 35, 9, 27, 30]
# A = [   [0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 1, 0, 0, 1, 0],
#         [5, 0, 0, 0, 0, 1]]
# B = [
#     [0, 1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0]]
# C = [
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0]]


#TARGET POLYNOMIAL - TEST 3
# f(x) = 2 * x^3 + 2 * x^2 + x + 5
r = [1, 2, 31, 4, 8, 16, 4, 8, 24, 26]
A = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
[5, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
B = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
C = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]


Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
print('Ap')
for x in Ap: print(x)
print('Bp')
for x in Bp: print(x)
print('Cp')
for x in Cp: print(x)
print('Z')
print(Z)
R = r

Ax = [ [FR(round(num)) for num in vec] for vec in Ap ]
Bx = [ [FR(round(num)) for num in vec] for vec in Bp ]
Cx = [ [FR(round(num)) for num in vec] for vec in Cp ]
Zx = [ FR(round(num)) for num in Z ]
Rx = [ FR(round(num)) for num in R ]

print("Ax : {}".format(Ax))
print("Bx : {}".format(Bx))
print("Cx : {}".format(Cx))
print("Zx : {}".format(Zx))

# Rax = [multiply_polys(Rx, vec)for vec in Ax]
# Rbx = [multiply_polys(Rx, vec)for vec in Bx]
# Rcx = [multiply_polys(Rx, vec)for vec in Cx]

def multiply_vec_matrix(vec, matrix):
    # len(vec) == len(matrix.row)
    assert not len(vec) == len(matrix[0])
    target = [FR(0)]*len(vec)
    for i in range(len(matrix)): #loop num of rows == size of vec, 0-5
        for j in range(len(matrix[0])): #loop num of columns, 0-3
            target[j] = target[j] + vec[i] * matrix[i][j]
    return target


Rax = multiply_vec_matrix(R, Ax)
Rbx = multiply_vec_matrix(R, Bx)
Rcx = multiply_vec_matrix(R, Cx)

print('Rax', Rax)
print('Rbx', Rbx)
print('Rcx', Rcx)


#Px = Rax * Rbx - Rcx
Px = subtract_polys(multiply_polys(Rax, Rbx), Rcx)
print('Px', Px)

# Px = Hx.Zx
q, r = div_polys(Px, Zx)
print('q', q)
print('r', r)
Hx = q
print('r', r)
# r should be zero

#q = [14592161914559516814830937163504850059130874104865215775126025263096817472385,
#  20672229378959315487677160981631870917102071648559055681428535789387158085901,
#  9728107943039677876553958109003233372753916069910143850084016842064544981589,
#  0,
#  0,
#  0,
#  0]

#r = [0, 0, 0, 0]
print('//---------------')

# alpha = FR(3926)
# beta = FR(3604)
# gamma = FR(2971)
# delta = FR(1357)
# x_val = FR(3721)

# alpha = FR(718946396703)
# beta = FR(238324186104)
# gamma = FR(968267494313)
# delta = FR(730816178932)
# x_val = FR(688296554963)

alpha = FR(942489720369)
beta = FR(249472381224)
delta = FR(699936090831)
gamma = FR(288797827954)
x_val = FR(482323349322)

# alpha = FR(randint(0, bn128.curve_order))
# beta = FR(randint(0, bn128.curve_order))
# gamma = FR(randint(0, bn128.curve_order))
# delta = FR(randint(0, bn128.curve_order))
# x_val = FR(randint(0, bn128.curve_order))

tau = [alpha, beta, gamma, delta, x_val]

print('alpha', alpha)
print('beta', beta)
print('gamma', gamma)
print('delta', delta)
print('x_val', x_val)

Ax_val = []
Bx_val = []
Cx_val = []

for i in range(len(Ax)):
    ax_single = eval_poly(Ax[i], x_val)
    Ax_val.append(ax_single)

for i in range(len(Bx)):
    bx_single = eval_poly(Bx[i], x_val)
    Bx_val.append(bx_single)

for i in range(len(Cx)):
    cx_single = eval_poly(Cx[i], x_val)
    Cx_val.append(cx_single)

Zx_val = eval_poly(Zx, x_val)
Hx_val = eval_poly(Hx, x_val)

print('Ax_val',Ax_val)
print('Bx_val',Bx_val)
print('Cx_val',Cx_val)
print('Zx_val',Zx_val)


#numGates = len(Ax.columns())
#numWires = len(Ax.rows())

numGates = len(Ax[0])
numWires = len(Ax)

sigma1_1 = [mult(g1, int(alpha)), mult(g1, int(beta)), mult(g1, int(delta))]

print('sigma1_1',sigma1_1)

sigma1_2 = []
sigma1_3 = []
sigma1_4 = []
sigma1_5 = []

sigma2_1 = [mult(g2, int(beta)), mult(g2, int(gamma)), mult(g2, int(delta))]
sigma2_2 = []

#sigma1_2
for i in range(numGates):
    val = x_val ** i
    sigma1_2.append(mult(g1, int(val)))

print('sigma1_2',sigma1_2)

#sigma1_3
VAL = [FR(0)]*numWires
for i in range(numWires):
    if i in [0, numWires-1]:
        val = (beta*Ax_val[i] + alpha*Bx_val[i] + Cx_val[i]) / gamma
        VAL[i] = val
        sigma1_3.append(mult(g1, int(val)))
    else:
        sigma1_3.append((FQ(0), FQ(0)))

print('sigma1_3',sigma1_3)

#sigma1_4
for i in range(numWires):
    if i in [0, numWires-1]:
        sigma1_4.append((FQ(0), FQ(0)))
    else:
        val = (beta*Ax_val[i] + alpha*Bx_val[i] + Cx_val[i]) / delta
        sigma1_4.append(mult(g1, int(val)))

print('sigma1_4',sigma1_4)

#sigma1_5
for i in range(numGates-1):
    sigma1_5.append(mult(g1, int((x_val**i * Zx_val) / delta)))

print('sigma1_5',sigma1_5)

#sigma2-2
for i in range(numGates):
    # sigma2_2.append(h*(Z(x_val^i)))
    sigma2_2.append(mult(g2, int(x_val**i)))

print('sigma2-2',sigma2_2)

##CRS validity check

# Ax_val = vector(Z, Ax_val)
# Bx_val = vector(Z, Bx_val)
# Cx_val = vector(Z, Cx_val)

# Zx_val = Z(Zx_val)
# Hx_val = Z(Hx_val)

def multiply_vec_vec(vec1, vec2):
    assert len(vec1) == len(vec2)
    target = 0
    size = len(vec1)
    for i in range(size):
        target += vec1[i]*vec2[i]
    return target

lhs = multiply_vec_vec(Rx, Ax_val) * multiply_vec_vec(Rx, Bx_val) - multiply_vec_vec(Rx, Cx_val)
rhs = Zx_val * Hx_val

# lhs = Z((Rx*Ax_val)*(Rx*Bx_val)-(Rx*Cx_val))
# rhs = Zx_val*Hx_val

print("polynomial verification")
print(lhs == rhs)


### 2. PROVING ###

# r = FR(4106)
# s = FR(4565)

r = FR(697899094451)
s = FR(294858740669)

# r = FR(randint(0, bn128.curve_order))
# s = FR(randint(0, bn128.curve_order))

print("//-----Build Proof_A, g1 based  ")
print("pointInf1", pointInf1)
print("sigma1_1", sigma1_1)
print("sigma1_2", sigma1_2)
print("numGates", numGates)
print("Ax", Ax)
print("Rx", Rx)
print("r", r)

#Build Proof_A, g1 based
print('numGates',numGates )
print('numWires',numWires )
print("pointInf1", pointInf1)

proof_A = sigma1_1[0]
for i in range(numWires):
    temp = pointInf1
    for j in range(numGates):
        temp = add(temp, mult(sigma1_2[j], int(Ax[i][j])))
    proof_A = add(proof_A, mult(temp, int(Rx[i])))

proof_A = add(proof_A, mult(sigma1_1[2], int(r)))
print("** proof_A", proof_A)
print("//-------------------  ")
#Build proof_B, g2 based
proof_B = sigma2_1[0]
for i in range(numWires):
    temp = pointInf2
    for j in range(numGates):
        temp = add(temp, mult(sigma2_2[j], int(Bx[i][j])))
    proof_B = add(proof_B, mult(temp, int(Rx[i])))
proof_B = add(proof_B, mult(sigma2_1[2], int(s)))


#Build temp_proof_B
temp_proof_B = sigma1_1[1]
for i in range(numWires):
    temp = pointInf1
    for j in range(numGates):
        temp = add(temp, mult(sigma1_2[j], int(Bx[i][j])))
    temp_proof_B = add(temp_proof_B, mult(temp, int(Rx[i])))
temp_proof_B = add(temp_proof_B, mult(sigma1_1[2], int(s)))

#Build proof_C, g1_based
# proof_C = add(add(mult(proof_A, int(s)), mult(temp_proof_B, int(r))), neg(mult(sigma1_1[2], int(r*s))))
proof_C = add(add(mult(proof_A, int(s)), mult(temp_proof_B, int(r))), neg(mult(mult(sigma1_1[2], int(s)), int(r))))

for i in range(1, numWires-1):
    proof_C = add(proof_C, mult(sigma1_4[i], int(Rx[i])))

for i in range(numGates-1):
    proof_C = add(proof_C, mult(sigma1_5[i], int(Hx[i])))


proof = [proof_A, proof_B, proof_C]

print("proofs : ", proof)
print("")

### 2.1 PROOF COMPLETENESS CHECK ###

def scalar_vec(scalar, vec):
    return [scalar*num for num in vec]

A = alpha + multiply_vec_vec(Rx, Ax_val) + r*delta
B = beta + multiply_vec_vec(Rx, Bx_val) + s*delta

# C1 = scalar_vec(1/delta, Rx[1:numWires-1])
# C2_1 = scalar_vec(beta, Ax_val[1:numWires-1])
# C2_2 = scalar_vec(alpha, Bx_val[1:numWires-1])
# C2_3 = Cx_val[1:numWires-1]
# C3 = Hx_val*Zx_val
# C4 = A*s + B*r - r*s*delta

# C = multiply_vec_vec(C1, (add_polys(add_polys(C2_1, C2_2), C2_3))) + C3 + C4


# C0 * {C1*(C1_1+C1_2+C1_3) + C2} + C3
C0 = 1/delta
C1 = Rx[1:numWires-1] #vec
C1_1 = scalar_vec(beta, Ax_val[1:numWires-1])
C1_2 = scalar_vec(alpha, Bx_val[1:numWires-1])
C1_3 = Cx_val[1:numWires-1]
C2 = Hx_val*Zx_val
C3 = A*s + B*r - r*s*delta

C1112 = add_polys(C1_1, C1_2) # vec
C111213 = add_polys(C1112, C1_3) # vec
C1111213 = multiply_vec_vec(C1, C111213) #num

C = C0 * (C1111213 + C2) + C3

# C = multiply_vec_vec(C1, (add_polys(add_polys(C2_1, C2_2), C2_3))) + C3 + C4

lhs = A*B #21888242871839275222246405745257275088548364400416033032405666501928354297837

rhs = alpha*beta #14149304

rpub = [Rx[0], Rx[-1]] #[1, 30]
valpub = [VAL[0], VAL[-1]]
print("valpub : {}".format(valpub))

#[17858330771234736835653075572704017103548042849750409710240473560856989375368,
# 3057428741774924004453806255227791707084339019243572619533744442206670609805]

print("VAL : {}".format(VAL))
#VAL : [17858330771234736835653075572704017103548042849750409710240473560856989375368, 0, 0, 0, 0, 3057428741774924004453806255227791707084339019243572619533744442206670609805]

rhs = rhs + gamma*multiply_vec_vec(rpub,valpub) #12058091336480024
rhs = rhs + C*delta #21888242871839275222246405745257275088548364400414254943408259015785828911597

print("#PROOF COMPLETENESS CHECK#")
print("rhs : {}".format(rhs))
print("lhs : {}".format(lhs))
print("rhs == lhs ? : {}".format(rhs == lhs))
print("proof A check : {}".format(proof_A == mult(g1, int(A))))
print("proof B check : {}".format(proof_B == mult(g2, int(B))))
print("proof C check : {}".format(proof_C == mult(g1, int(C))))
print("")

# A = alpha + Rx*Ax_val + r*delta
# B = beta + Rx*Bx_val + s*delta
# C = 1/delta*Rx[1:numWires-1]*(beta*Ax_val[1:numWires-1] + alpha*Bx_val[1:numWires-1] + Cx_val[1:numWires-1]) + Hx_val*Zx_val + A*s + B*r + (-r*s*delta)

# lhs = A*B

# rhs = alpha*beta #2024

# rpub = [Rx[0], Rx[-1]]
# valpub = [VAL[0], VAL[-1]]

# rhs = rhs + gamma*vector(rpub)*vector(valpub)  #4040
# rhs = rhs + C*delta #984

# result = (proof_A == g*A) and (proof_B == B*h) and (proof_C == C*g)

# print("proof completeness check : {}".format(result and lhs==rhs))

##### 3. VERIFY ######

LHS = pairing(proof_B, proof_A)
RHS = pairing(sigma2_1[0], sigma1_1[0])

temp = None

for i in [0, numWires-1]:
  temp = add(temp, mult(sigma1_3[i], int(Rx[i])))

RHS = (RHS * pairing(sigma2_1[1], temp)) * pairing(sigma2_1[2], proof_C)

print("LHS", LHS)
print("")
print("RHS", RHS)
print("")
print("Verification result (RHS == LHS)? : {}".format(RHS == LHS))