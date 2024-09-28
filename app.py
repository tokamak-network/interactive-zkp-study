from flask import Flask, session, redirect, url_for
from flask import render_template
from flask import request

from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

import ast

from zkp.groth16.code_to_r1cs import (
    parse,
    extract_inputs_and_body,
    flatten_body,
    initialize_symbol,
    get_var_placement,
    flatcode_to_r1cs,
    assign_variables
)

from zkp.groth16.qap_creator import (
    r1cs_to_qap,
    create_solution_polynomials
)

from zkp.groth16.qap_creator_lcm import (
    r1cs_to_qap_times_lcm
)

from zkp.groth16.poly_utils import (
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

from zkp.groth16.setup import (
    sigma11,
    sigma12,
    sigma13,
    sigma14,
    sigma15,
    sigma21,
    sigma22
)

from zkp.groth16.proving import (
    proof_a,
    proof_b,
    proof_c
)

from zkp.groth16.verifying import (
    verify,
    lhs,
    rhs,
    verifyResult
)

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

# DB = TinyDB(storage=MemoryStorage) #Memony DB
DB = TinyDB('db.json')               #Storage DB

DATA = Query()

class FR(FQ):
    field_modulus = bn128.curve_order

G1 = bn128.G1
G2 = bn128.G2

app = Flask(__name__)
app.secret_key = "key"

groth_db = DB.table("groth")
plonk_db = DB.table("plonk")
jake_db = DB.table("jake")
Grothes = Query()

def clear_groth_db():
    DB.drop_table('groth')

def clear_groth(type_name):
    groth_db.remove(Grothes.type == type_name)

def update_groth(type_name, value):
    row = groth_db.search(Grothes.type == type_name)
    if row == [] :
        groth_db.insert({"type": type_name, "value":value})
    else:
        clear_groth(type_name)
        groth_db.insert({"type": type_name, "value":value})
        # groth_db.update({"type": type_name, "value":value})

def select_groth(type_name):
    row = groth_db.search(Grothes.type == type_name)
    if row == []: return None
    else:
        return_data = row[0]['value']
        return return_data

## Groth16 Related Functions ##
def make_target_dict(ast_obj):
    if isinstance(ast_obj, ast.Assign):
        assert len(ast_obj.targets) == 1 and isinstance(ast_obj.targets[0], ast.Name)
        target = ast_obj.targets[0].id
        ast_value = ast_obj.value
        return {"targets": target, "value" : make_expr_dict(ast_value)}
    elif isinstance(ast_obj, ast.Return):
        ast_value = ast_obj.value
        return {"value" : make_expr_dict(ast_value)}

def make_expr_dict(ast_value):
    if isinstance(ast_value, ast.Name):
        return ast_value.id
    elif isinstance(ast_value, ast.Constant):
        return ast_value.n
    elif isinstance(ast_value, ast.BinOp):

        left = make_expr_dict(ast_value.left)
        right = make_expr_dict(ast_value.right)

        if isinstance(ast_value.op, ast.Add):
            op = '+'
        elif isinstance(ast_value.op, ast.Mult):
            op = '*'
        elif isinstance(ast_value.op, ast.Sub):
            op = '-'
        elif isinstance(ast_value.op, ast.Div):
            op = '/'
        elif isinstance(ast_value.op, ast.Pow):
            op = '*'

        return {"left" : left, "op": op, "right" : right}

DEFAULT_CODE = """
def qeval(x):
    y = x**3
    return y + x + 5"""

@app.route("/", methods=["POST","GET"])
def main():
    user_code = select_groth("computation.code")
    ast_obj = select_groth("computation.ast_obj")
    flatcode = select_groth("computation.flatcode")
    variables = select_groth("computation.variables")
    abc = select_groth("computation.abc")
    inputs = select_groth("computation.inputs")
    user_inputs = select_groth("computation.user_inputs")
    r_values = select_groth("computation.r_values")
    qap = select_groth("computation.qap")
    qap_lcm = select_groth("computation.qap_lcm")
    qap_fr = select_groth("computation.qap_fr")
    fr_modulus =  select_groth("computation.fr_modulus")

    if user_code == None:
        user_code = DEFAULT_CODE
        update_groth("computation.code", user_code)

    return render_template('groth16/computation.html', \
                           code=user_code, \
                           ast_obj=ast_obj, \
                           flatcode=flatcode, \
                           variables=variables, \
                           abc=abc, \
                           inputs=inputs, \
                           user_inputs=user_inputs, \
                           r_vector=r_values, \
                           qap=qap, \
                           qap_lcm=qap_lcm, \
                           qap_fr=qap_fr, \
                           fr_modulus=fr_modulus \
                           )

@app.route("/code", methods=['POST'])
def save_code():
    if request.method == "POST":
        session.clear() #clear session before save
        user_code = request.form['z-code']
        update_groth("computation.code", user_code)
        session["code"] = user_code
        # return render_template('computation.html', code=session["code"])
        return redirect(url_for('main'))

@app.route("/code/delete", methods=["POST"])
def delete_code():
    if request.method == "POST":
        session.clear()
        groth_db.truncate()
    return redirect(url_for('main'))

@app.route("/code/ast", methods=["POST"])
def ast_table():
    if request.method == "POST":

        user_code = select_groth("computation.code")

        if user_code:
            func_name = parse(user_code)[0].name
            inputs, body = extract_inputs_and_body(parse(user_code))
            final_out = {}
            out = []
            for ast_obj in body:
                obj = make_target_dict(ast_obj)
                out.append(obj)

            final_out["name"] = func_name
            final_out["inputs"] = inputs
            final_out["body"] = out

            #save ast object
            session['ast_obj'] = final_out
            update_groth("computation.ast_obj", final_out)
            # DB.upsert({"type":"groth.computation.ast_obj", "ast_obj":final_out}, DATA.type == "groth.computation.ast_obj")
            # flatcode = flatten_body(body)
            # print(flatcode)

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

def clear_flatcode():
    groth_db.remove(Grothes.type == "computation.flatcode")
    groth_db.remove(Grothes.type == "computation.variables")
    groth_db.remove(Grothes.type == "computation.code")

@app.route("/flatcode/table", methods=["POST"])
def flatcode_table():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            variables = get_var_placement(inputs, flatcode)

            initialize_symbol()

            update_groth("computation.flatcode", flatcode)
            update_groth("computation.variables", variables)

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/r1cs/abc", methods=["POST"])
def abc_matrix():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            initialize_symbol()

            update_groth("computation.abc", {"A": A, "B": B, "C": C})

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

# @app.route("/r1cs/inputs", methods=["POST"])
# def retrieve_values():
#     if request.method == "POST":
#         user_code = session.get("code")
#         if user_code:
#             return redirect(url_for('main'))
#         else:
#             return redirect(url_for('main'))

@app.route("/r1cs/inputs", methods=["POST"])
def retrieve_values():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            update_groth("computation.inputs", inputs)
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/r1cs/inputs/r", methods=["POST"])
def calculate_r():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            form_data = request.form
            user_inputs = []
            for d in form_data:
                user_inputs.append(int(form_data[d]))

            update_groth("computation.user_inputs", form_data)

            # todo : calculate r vector
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            r = assign_variables(inputs, user_inputs, flatcode)

            initialize_symbol()
            # session['r_values'] = r
            update_groth("computation.r_values", r)

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/qap/normal", methods=["POST"])
def create_qap():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)

            Ap, Bp, Cp, Z = r1cs_to_qap(A, B, C)
            initialize_symbol()

            update_groth("computation.qap", {"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z})

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/qap/lcm", methods=["POST"])
def create_qap_lcm():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)

            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            session["qap_lcm"] = {"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z}
            update_groth("computation.qap_lcm", {"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z})

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))


@app.route("/qap/fr", methods=["POST"])
def create_qap_fr():
    if request.method == "POST":
        user_code = select_groth("computation.code")
        r_values = select_groth("computation.r_values")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            initialize_symbol()
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)

            #FR object must be converted to int
            Ax = [ [int(FR(int(n))) for n in vec] for vec in Ap ]
            Bx = [ [int(FR(int(n))) for n in vec] for vec in Bp ]
            Cx = [ [int(FR(int(n))) for n in vec] for vec in Cp ]
            Zx = [ int(FR(int(num))) for num in Z ]
            Rx = [ int(FR(int(num))) for num in r_values ]

            o = {"Ax" : Ax, "Bx": Bx, "Cx": Cx, "Zx": Zx, "Rx": Rx}

            update_groth("computation.qap_fr", o)
            fr_modulus = int(FR.field_modulus)
            update_groth("computation.fr_modulus", fr_modulus)

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))


@app.route("/groth/setup")
def main_setup():

    toxic = select_groth("setup.toxic")
    polys = select_groth("setup.polys")
    polys_x_val = select_groth("setup.polys_x_val")
    numWires = select_groth("setup.numWires")
    numGates = select_groth("setup.numGates")
    g1 = select_groth("setup.g1")
    g2 = select_groth("setup.g2")
    sigmas = select_groth("setup.sigmas")
    gates = select_groth("setup.gates")
    public_gates = select_groth("setup.public_gates")

    return render_template("groth16/setup.html", \
                           toxic = toxic, \
                           polys = polys, \
                           polys_x_val = polys_x_val, \
                           numWires = numWires, \
                           numGates = numGates, \
                           g1 = g1, \
                           g2 = g2, \
                           sigmas = sigmas, \
                           gates = gates, \
                           public_gates = public_gates \
                           )

@app.route("/groth/setup/toxic/save", methods=["POST"])
def setup_save_toxic():
    if request.method == "POST":
        toxic_alpha = request.form['toxic-alpha']
        toxic_beta = request.form['toxic-beta']
        toxic_delta = request.form['toxic-delta']
        toxic_gamma = request.form['toxic-gamma']
        toxic_x_val = request.form['toxic-x-val']

        o = {"alpha":toxic_alpha, "beta" : toxic_beta, "delta" : toxic_delta, "gamma" : toxic_gamma, "x_val": toxic_x_val}

        clear_groth("setup.toxic")
        update_groth("setup.toxic", o)

        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/toxic/clear", methods=["POST"])
def clear_toxic():
    if request.method == "POST":

        clear_groth("setup.toxic")
        clear_groth("setup.polys")
        clear_groth("setup.polys_x_val")
        clear_groth("setup.numWires")
        clear_groth("setup.numGates")
        clear_groth("setup.g1")
        clear_groth("setup.g2")
        clear_groth("setup.gates")
        clear_groth("setup.public_gates")
        clear_groth("setup.sigmas")
        clear_groth("setup.gates")

        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/gates", methods=["POST"])
def load_gates():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            variables = get_var_placement(inputs, flatcode)
            initialize_symbol()

            update_groth("setup.gates", variables)
            DB.upsert({"type":"groth.setup.gates", "gates":variables}, DATA.type == "groth.setup.gates")
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/gates/set", methods=["POST"])
def set_public_gates():
    if request.method == "POST":
        gates = select_groth("setup.gates")
        # print(gates)
        if gates:
            # print("gates in")
            target = [0]
            for i in range(len(gates)-1):
                check = request.form.get("form-check-input-"+str(i+1))
                if check != None:
                    target.append(i+1)
            # session["public_gates"] = target
            update_groth("setup.public_gates", target)

            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/gates/reset", methods=["POST"])
def reset_public_gates():
    if request.method == "POST":

        clear_groth("setup.public_gates")
        clear_groth("setup.sigmas")

        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

# @app.route("/groth/setup/polys", methods=["POST"])
# def clear_toxic():
#     if request.method == "POST":
#         return redirect(url_for('main_setup'))
#     else:
#         return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys", methods=["POST"])
def get_polys():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            Ax = [ [int(FR(int(n))) for n in vec] for vec in Ap ]
            Bx = [ [int(FR(int(n))) for n in vec] for vec in Bp ]
            Cx = [ [int(FR(int(n))) for n in vec] for vec in Cp ]
            Zx = [ int(FR(int(num))) for num in Z ]
            # print(Ax)

            o = {"Ap": Ax, "Bp": Bx, "Cp":Cx, "Zp":Zx}

            print(o)

            update_groth("setup.polys", o)
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys/evaluated", methods=["POST"])
def get_polys_evaluated():
    if request.method == "POST":
        user_code = select_groth("computation.code")
        toxic = select_groth("setup.toxic")

        if user_code:

            x_val = FR(int(toxic["x_val"]))

            # print("x_val?? : {}".format(x_val))

            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            Ax = getFRPoly2D(Ap)
            Bx = getFRPoly2D(Bp)
            Cx = getFRPoly2D(Cp)
            Zx = getFRPoly1D(Z)

            Ax_val = ax_val(Ax, x_val)
            Bx_val = bx_val(Bx, x_val)
            Cx_val = cx_val(Cx, x_val)
            Zx_val = zx_val(Zx, x_val)

            Ax_val_int = [ int(num) for num in Ax_val ]
            Bx_val_int = [ int(num) for num in Bx_val ]
            Cx_val_int = [ int(num) for num in Cx_val ]
            Zx_val_int = int(Zx_val)

            # print("Ax_Val : {}".format(type(Ax_val[0])))

            o = {"Ax_val": Ax_val_int, "Bx_val": Bx_val_int, "Cx_val":Cx_val_int, "Zx_val":Zx_val_int}
            update_groth("setup.polys_x_val", o)
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys/clear", methods=["POST"])
def clear_polys():
    if request.method == "POST":
        clear_groth("setup.polys")
        clear_groth("setup.polys_x_val")

        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/sigma/formula", methods=["POST"])
def sigma_formula():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            numWires = getNumWires(Ap)
            numGates = getNumGates(Ap)
            update_groth("setup.numWires", numWires)
            update_groth("setup.numGates", numGates)

            g1_int = [int(f) for f in G1]
            g2_0 = [int(G2[0].coeffs[0]), int(G2[0].coeffs[1])]
            g2_1 = [int(G2[1].coeffs[0]), int(G2[1].coeffs[0])]
            g2_int = [g2_0, g2_1]

            update_groth("setup.g1", g1_int)
            update_groth("setup.g2", g2_int)
            # print("Wires and Gates : {}, {}".format(numWires, numGates))
            # print("G1 : {}".format(G1))
            # print("type(G1[0]) : {}".format(type(G1[0])))
            # print("G2 : {}".format(G2))

            # print("g1 int {}".format(g1_int))
            # print("g2 int {}".format(g2_int))


            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/sigma/calc", methods=["POST"])
def calculate_sigmas():
    if request.method == "POST":
        user_code = select_groth("computation.code")
        toxic = select_groth("setup.toxic")
        public_gates = select_groth("setup.public_gates")

        if user_code:

            if toxic == None:
                return redirect(url_for('main_setup'))

            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            Ax = getFRPoly2D(Ap)
            Bx = getFRPoly2D(Bp)
            Cx = getFRPoly2D(Cp)
            Zx = getFRPoly1D(Z)
            # print("Ax : {}".format(Ax))
            # print("Bx : {}".format(Bx))
            # print("Cx : {}".format(Cx))
            # print("Zx : {}".format(Zx))

            numGates = getNumGates(Ax)
            numWires = getNumWires(Ax)

            x_val = FR(int(toxic["x_val"]))
            alpha = FR(int(toxic["alpha"]))
            beta = FR(int(toxic["beta"]))
            delta = FR(int(toxic["delta"]))
            gamma = FR(int(toxic["gamma"]))

            # print("x_val : {}".format(x_val))
            # print("alpha : {}".format(alpha))
            # print("beta : {}".format(beta))
            # print("delta : {}".format(delta))
            # print("gamma : {}".format(gamma))

            Ax_val = ax_val(Ax, x_val)
            Bx_val = bx_val(Bx, x_val)
            Cx_val = cx_val(Cx, x_val)
            Zx_val = zx_val(Zx, x_val)

            # print("Ax_val : {}".format(Ax_val))
            # print("Bx_val : {}".format(Bx_val))
            # print("Cx_val : {}".format(Cx_val))
            # print("Zx_val : {}".format(Zx_val))
            # print("Hx_val : {}".format(Hx_val))

            s11 = sigma11(alpha, beta, delta)
            s12 = sigma12(numGates, x_val)
            s13, VAL = sigma13(numWires, alpha, beta, gamma, Ax_val, Bx_val, Cx_val, public_gates)
            s14 = sigma14(numWires, alpha, beta, delta, Ax_val, Bx_val, Cx_val, public_gates)
            s15 = sigma15(numGates, delta, x_val, Zx_val)
            s21 = sigma21(beta, delta, gamma)
            s22 = sigma22(numGates, x_val)

            # print("s11 : {}".format(s11))
            # print("s12 : {}".format(s12))
            # print("s13 : {}".format(s13))
            # print("s14 : {}".format(s14))
            # print("s15 : {}".format(s15))
            # print("s21 : {}".format(s21))
            # print("s22 : {}".format(s22))
            # print("VAL : {}".format(VAL))

            def turn_point_int(li):
                return [int(num) for num in li]

            def turn_g2_int(g2p):
                o = []
                g2p0 = [int(g2p[0].coeffs[0]), int(g2p[0].coeffs[1])]
                g2p1 = [int(g2p[1].coeffs[0]), int(g2p[1].coeffs[1])]
                g2_int = [g2p0, g2p1]
                return g2_int

            s11_int = [turn_point_int(point) for point in s11]
            s12_int = [turn_point_int(point) for point in s12]
            s13_int = [turn_point_int(point) for point in s13]
            s14_int = [turn_point_int(point) for point in s14]
            s15_int = [turn_point_int(point) for point in s15]
            s21_int = [turn_g2_int(point) for point in s21]
            s22_int = [turn_g2_int(point) for point in s22]

            # print("11 : {}".format(s11))
            # print("12 : {}".format(s12))
            # print("13 : {}".format(s13))
            # print("14 : {}".format(s14))
            # print("15 : {}".format(s15))
            # print("21 : {}".format(s21))
            # print("21 : {}".format(s22))

            # print("11_i : {}".format(s11_int))
            # print("12_i : {}".format(s12_int))
            # print("13_i : {}".format(s13_int))
            # print("14_i : {}".format(s14_int))
            # print("15_i : {}".format(s15_int))
            # print("21_i : {}".format(s21_int))
            # print("22_i : {}".format(s22_int))

            o = {"1_1":s11_int, "1_2":s12_int, "1_3":s13_int, "1_4":s14_int, "1_5":s15_int, "2_1":s21_int, "2_2":s22_int}
            update_groth("setup.sigmas", o)

            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/sigma/clear", methods=["POST"])
def clear_sigmas():
    if request.method == "POST":

        clear_groth("setup.numWires")
        clear_groth("setup.numGates")
        clear_groth("setup.g1")
        clear_groth("setup.g2")
        clear_groth("setup.sigmas")

        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

#### PROVING ####

@app.route("/groth/proving")
def main_proving():

    p_random = select_groth("proving.prover_random")
    inputs = select_groth("proving.inputs")
    user_inputs = select_groth("proving.user_inputs")
    r_values = select_groth("proving.r_values")
    public_gates = select_groth("setup.public_gates")
    proofs = select_groth("proving.proofs")
    p_inputs_is_load = select_groth("proving.prover_input_form")


    return render_template("groth16/proving.html", \
                           p_random=p_random, \
                           p_input_is_load=p_inputs_is_load, \
                           inputs=inputs, \
                           user_inputs=user_inputs, \
                           r_values=r_values, \
                           public_gates=public_gates, \
                           proofs=proofs \
                           )

# @app.route("/groth/proving/random/save", methods=["POST"])
# def save_prover_random():
#     if request.method == "POST":
#         user_code = session.get("code")
#         if user_code:
#             return redirect(url_for('main_proving'))
#     else:
#         return redirect(url_for('main_proving'))

@app.route("/groth/proving/random/save", methods=["POST"])
def save_prover_random():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            random_r = request.form['prover-random-r']
            random_s = request.form['prover-random-s']
            o = {"r" : int(random_r), "s" : int(random_s)}

            update_groth("proving.prover_random", o)
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/random/clear", methods=["POST"])
def clear_prover_random():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            clear_groth("proving.prover_random")
            clear_groth("proving.prover_input_form")
            clear_groth("proving.inputs")
            clear_groth("proving.user_inputs")
            clear_groth("proving.r_values")
            clear_groth("proving.proofs")

            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/inputs", methods=["POST"])
def load_prover_input():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            update_groth("proving.prover_input_form", True)
            update_groth("proving.inputs", inputs)
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/witness/calc", methods=["POST"])
def calculate_witness():
    if request.method == "POST":
        user_code = select_groth("computation.code")

        if user_code:
            form_data = request.form
            user_inputs = []
            for d in form_data:
                user_inputs.append(int(form_data[d]))

            # todo : calculate r vector
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            r = assign_variables(inputs, user_inputs, flatcode)
            initialize_symbol()

            update_groth("proving.r_values", r)
            update_groth("proving.user_inputs", form_data)
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/proof/generate", methods=["POST"])
def generate_proof():
    if request.method == "POST":
        #TODO : before call this function, check wehther below data exists

        user_code = select_groth("computation.code")
        user_inputs = select_groth("proving.user_inputs")
        public_gates = select_groth("setup.public_gates")
        prover_random = select_groth("proving.prover_random")
        sigmas = select_groth("setup.sigmas")
        polys = select_groth("setup.polys")

        user_inputs_li = [int(user_inputs[i]) for i in user_inputs]


        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            rx = assign_variables(inputs, user_inputs_li, flatcode)
            initialize_symbol()

            r = prover_random["r"]
            s = prover_random["s"]

            Ax = [ [FR(n) for n in vec] for vec in polys["Ap"] ]
            Bx = [ [FR(n) for n in vec] for vec in polys["Bp"] ]
            Cx = [ [FR(n) for n in vec] for vec in polys["Cp"] ]
            Zx = [ FR(num) for num in polys["Zp"] ]
            Rx = [FR(r) for r in rx]

            Hx, remain = hxr(Ax, Bx, Cx, Zx, Rx)

            # print(Ax)

            def turn_g1_int(g1p):
                return [int(num) for num in g1p]

            def turn_g2_int(g2p):
                o = []
                g2p0 = [int(g2p[0].coeffs[0]), int(g2p[0].coeffs[1])]
                g2p1 = [int(g2p[1].coeffs[0]), int(g2p[1].coeffs[1])]
                g2_int = [g2p0, g2p1]
                return g2_int

            def turn_g2_fq2(g2p_int):
                g2p0 = bn128.FQ2(g2p_int[0])
                g2p1 = bn128.FQ2(g2p_int[1])
                return (g2p0, g2p1)

            def turn_g1_fq(g1_int):
                return (bn128.FQ(g1_int[0]), bn128.FQ(g1_int[1]))

            sigma1_1 = [turn_g1_fq(point) for point in sigmas["1_1"]]
            sigma1_2 = [turn_g1_fq(point) for point in sigmas["1_2"]]
            sigma1_4 = [turn_g1_fq(point) for point in sigmas["1_4"]]
            sigma1_5 = [turn_g1_fq(point) for point in sigmas["1_5"]]
            sigma2_1 = [turn_g2_fq2(point) for point in sigmas["2_1"]]
            sigma2_2 = [turn_g2_fq2(point) for point in sigmas["2_2"]]

            # print("in generate_proof()")
            # print("sigma1_1 : {}".format(sigma1_1))
            # print("type(sigma1_1[0][0]) : {}".format(type(sigma1_1[0][0])))
            # print("sigma1_2 : {}".format(sigma1_2))
            # print("sigma1_3 : {}".format(sigma1_3))
            # print("sigma1_4 : {}".format(sigma1_4))
            # print("sigma1_5 : {}".format(sigma1_5))
            # print("sigma2_1 : {}".format(sigma2_1)) #TODO : not right
            # print("sigma2_2 : {}".format(sigma2_2)) #TODO : not right

            # print(sigma1_1)
            # print(sigma2_2)

            prf_a = proof_a(sigma1_1, sigma1_2, Ax, Rx, r)
            prf_b = proof_b(sigma2_1, sigma2_2, Bx, Rx, s)
            prf_c = proof_c(sigma1_1, sigma1_2, sigma1_4, sigma1_5, Bx, Rx, Hx, s, r, prf_a, public_gates)

            # print(prf_a)
            # print(prf_b)
            # print(prf_c)

            #TODO : should change way of storing data
            value_ = {"proof_a" : turn_g1_int(prf_a), "proof_b" : turn_g2_int(prf_b), "proof_c" : turn_g1_int(prf_c)}
            update_groth("proving.proofs", value_)

            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

#### VERIFYING ####

@app.route("/groth/verifying", methods=['GET', 'POST'])
def main_verifying():
    public_gates_index = select_groth("setup.public_gates")
    r_values = select_groth("proving.r_values")
    proofs = select_groth("proving.proofs")
    public_gates = []
    if public_gates_index != None:
        public_gates = [r_values[i] for i in public_gates_index]

    templateData={
        "proofs":proofs,
        "public_gates":public_gates,
        "lhs":request.args.get("lhs"),
        "rhs":request.args.get("rhs"),
        "verify_result":request.args.get("result")
    }

    return render_template("groth16/verifying.html",**templateData)

@app.route("/groth/verifying/verify", methods=["POST"])
def groth_verify():
    if request.method == "POST":

        user_code = select_groth("computation.code")
        sigmas = select_groth("setup.sigmas")
        public_gates_index = select_groth("setup.public_gates")
        r_values = select_groth("proving.r_values")
        public_gates_with_index = [(i,r_values[i]) for i in public_gates_index]

        if user_code:
            def turn_g2_fq2(g2p_int):
                g2p0 = bn128.FQ2(g2p_int[0])
                g2p1 = bn128.FQ2(g2p_int[1])
                return (g2p0, g2p1)

            def turn_g1_fq(g1_int):
                return (bn128.FQ(g1_int[0]), bn128.FQ(g1_int[1]))

            proofs = select_groth("proving.proofs")

            proof_a_int = [int(st) for st in proofs["proof_a"]]
            proof_b_int = [[int(num) for num in vec] for vec in proofs["proof_b"]]
            proof_c_int = [int(st) for st in proofs["proof_c"]]

            proof_a = turn_g1_fq(proof_a_int)
            proof_b = turn_g2_fq2(proof_b_int)
            proof_c = turn_g1_fq(proof_c_int)

            sigma1_1 = [turn_g1_fq(point) for point in sigmas["1_1"]]
            sigma1_2 = [turn_g1_fq(point) for point in sigmas["1_2"]]
            sigma1_3 = [turn_g1_fq(point) for point in sigmas["1_3"]]
            sigma1_4 = [turn_g1_fq(point) for point in sigmas["1_4"]]
            sigma1_5 = [turn_g1_fq(point) for point in sigmas["1_5"]]
            sigma2_1 = [turn_g2_fq2(point) for point in sigmas["2_1"]]
            sigma2_2 = [turn_g2_fq2(point) for point in sigmas["2_2"]]

            # print("proof_a : {}".format(proof_a))
            # print("proof_b : {}".format(proof_b))
            # print("proof_c : {}".format(proof_c))

            # print("sigma1_1 : {}".format(sigma1_1))
            # print("type(sigma1_1[0][0]) : {}".format(type(sigma1_1[0][0])))
            # print("sigma1_2 : {}".format(sigma1_2))
            # print("sigma1_3 : {}".format(sigma1_3))
            # print("sigma1_4 : {}".format(sigma1_4))
            # print("sigma1_5 : {}".format(sigma1_5))
            # print("sigma2_1 : {}".format(sigma2_1))
            # print("sigma2_2 : {}".format(sigma2_2))

            # print("public_gates : {}".format(public_gates_with_index))

            lh = lhs(proof_a, proof_b)
            # print("lhs : {}".format(lh))
            print("verifyResult..." )
            verify_result = verifyResult(proof_a, proof_b, proof_c, sigma1_1, sigma1_3, sigma2_1,  public_gates_with_index)
            print(verify_result)

            return redirect(url_for('main_verifying', \
                lhs=verify_result["lhs"],\
                rhs=verify_result["rhs"],\
                result=verify_result["result"]), code=307)


    else:
        return redirect(url_for('main_verifying'))

# @app.route("/groth/verifying/proofs", methods=["POST"])
# def get_prover_proofs():
#     if request.method == "POST":

#         if proofs != []: #if proof generated
#             session["groth.verifying.proofs"] = True
#             session["groth.verifying.public_r"] = True
#             return redirect(url_for('main_proving'))
#     else:
#         return redirect(url_for('main_proving'))