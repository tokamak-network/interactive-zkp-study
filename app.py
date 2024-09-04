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

class FR(FQ):
    field_modulus = bn128.curve_order

G1 = bn128.G1
G2 = bn128.G2

app = Flask(__name__)
app.secret_key = "key"


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
    user_code = session.get("code")
    ast_obj = session.get('ast_obj')
    flatcode = session.get('flatcode')
    variables = session.get('variables')
    abc = session.get('abc')
    inputs = session.get('inputs')
    user_inputs = session.get('user_inputs')
    r_vector = session.get('r_values')

    qap = session.get('qap')
    qap_lcm = session.get('qap_lcm')
    
    qap_fr = session.get('qap_fr')
    fr_modulus = session.get('fr_modulus')
    
    if user_code == None:
        user_code = DEFAULT_CODE
    
    return render_template('groth16/computation.html', \
                           code=user_code, \
                           ast_obj=ast_obj, \
                           flatcode=flatcode, \
                           variables=variables, \
                           abc=abc, \
                           inputs=inputs, \
                           user_inputs=user_inputs, \
                           r_vector=r_vector, \
                           qap=qap, \
                           qap_lcm=qap_lcm, \
                           qap_fr=qap_fr, \
                           fr_modulus=fr_modulus \
                           )
    
@app.route("/code", methods=['POST'])
def save_code():
    if request.method == "POST":
        user_code = request.form['z-code']
        session["code"] = user_code
        # return render_template('computation.html', code=session["code"])
        return redirect(url_for('main'))
    
@app.route("/code/delete", methods=["POST"])
def delete_code():
    if request.method == "POST":
        session.clear()
    return redirect(url_for('main'))

@app.route("/code/ast", methods=["POST"])
def ast_table():
    if request.method == "POST":
        user_code = session.get("code")
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
            # flatcode = flatten_body(body)
            # print(flatcode)
            # #[['*', 'sym_1', 'x', 'x'], 
            # # ['*', 'y', 'sym_1', 'x'], 
            # # ['+', 'sym_2', 'y', 'x'], 
            # # ['+', '~out', 'sym_2', 5]]
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

def clear_flatcode():
    session['flatcode'] = None
    session['variables'] = None
        
@app.route("/flatcode/table", methods=["POST"])
def flatcode_table():
    if request.method == "POST":
        user_code = session.get("code")
        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            variables = get_var_placement(inputs, flatcode)
            
            initialize_symbol()
            
            session['flatcode'] = flatcode
            session['variables'] = variables
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))
        
@app.route("/r1cs/abc", methods=["POST"])
def abc_matrix():
    if request.method == "POST":
        user_code = session.get("code")
        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            session["abc"] = {"A": A, "B": B, "C": C}

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
        user_code = session.get("code")
        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            session['inputs'] = inputs
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))
        
@app.route("/r1cs/inputs/r", methods=["POST"])
def calculate_r():
    if request.method == "POST":
        user_code = session.get("code")
        if user_code:
            form_data = request.form
            user_inputs = []
            for d in form_data:
                user_inputs.append(int(form_data[d]))
            print(user_inputs)
            session['user_inputs'] = form_data
            
            # todo : calculate r vector
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            r = assign_variables(inputs, user_inputs, flatcode)
            session['r_values'] = r

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/qap/normal", methods=["POST"])
def create_qap():
    if request.method == "POST":
        user_code = session.get("code")
        if user_code: 
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)

            Ap, Bp, Cp, Z = r1cs_to_qap(A, B, C)

            session["qap"] = {"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z}

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/qap/lcm", methods=["POST"])
def create_qap_lcm():
    if request.method == "POST":
        user_code = session.get("code")
        if user_code: 
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)

            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)

            session["qap_lcm"] = {"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z}

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))
        

@app.route("/qap/fr", methods=["POST"])
def create_qap_fr():
    if request.method == "POST":
        user_code = session.get("code")
        r_values = session.get('r_values')
        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)

            #FR object must be converted to int
            Ax = [ [int(FR(int(n))) for n in vec] for vec in Ap ]
            Bx = [ [int(FR(int(n))) for n in vec] for vec in Bp ]
            Cx = [ [int(FR(int(n))) for n in vec] for vec in Cp ]
            Zx = [ int(FR(int(num))) for num in Z ]
            Rx = [ int(FR(int(num))) for num in r_values ]

            o = {"Ax" : Ax, "Bx": Bx, "Cx": Cx, "Zx": Zx, "Rx": Rx}
            session["qap_fr"] = o
            fr_modulus = int(FR.field_modulus)
            session["fr_modulus"]= fr_modulus

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))
        

@app.route("/groth/setup")
def main_setup():
    toxic = session.get("toxic")
    polys = session.get("polys")
    polys_x_val = session.get("polys_x_val")
    numWires = session.get("numWires")
    numGates = session.get("numGates")
    g1 = session.get("g1")
    g2 = session.get("g2")
    
    return render_template("groth16/setup.html", \
                           toxic = toxic, \
                           polys = polys, \
                           polys_x_val = polys_x_val, \
                           numWires = numWires, \
                           numGates = numGates, \
                           g1 = g1, \
                           g2 = g2,
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
        session["toxic"] = o
        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))
    
@app.route("/groth/setup/toxic/clear", methods=["POST"])
def clear_toxic():
    if request.method == "POST":
        session["toxic"] = None
        session["polys"] = None
        session["polys_x_val"] = None
        session["numWires"] = None
        session["numGates"] = None
        session["g1"] = None
        session["g2"] = None
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
        user_code = session.get("code")
        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)

            Ax = [ [int(FR(int(n))) for n in vec] for vec in Ap ]
            Bx = [ [int(FR(int(n))) for n in vec] for vec in Bp ]
            Cx = [ [int(FR(int(n))) for n in vec] for vec in Cp ]
            Zx = [ int(FR(int(num))) for num in Z ]

            o = {"Ap": Ax, "Bp": Bx, "Cp":Cx, "Zp":Zx}
            session["polys"] = o
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))
    
@app.route("/groth/setup/polys/evaluated", methods=["POST"])
def get_polys_evaluated():
    if request.method == "POST":
        user_code = session.get("code")
        toxic = session.get("toxic")
        if user_code:

            x_val = FR(int(toxic["x_val"]))

            print("x_val?? : {}".format(x_val))

            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)

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

            print("Ax_Val : {}".format(type(Ax_val[0])))

            o = {"Ax_val": Ax_val_int, "Bx_val": Bx_val_int, "Cx_val":Cx_val_int, "Zx_val":Zx_val_int}
            session["polys_x_val"] = o
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys/clear", methods=["POST"])
def clear_polys():
    if request.method == "POST":
        session["polys"] = None
        session["polys_x_val"] = None
        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))
    
@app.route("/groth/setup/sigma/formula", methods=["POST"])
def sigma_formula():
    if request.method == "POST":
        user_code = session.get("code")
        if user_code:
            session["formula"] = True
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)

            numWires = getNumWires(Ap)
            numGates = getNumGates(Ap)
            session["numWires"] = numWires
            session["numGates"] = numGates

            g1_int = [int(f) for f in G1]
            g2_0 = [int(G2[0].coeffs[0]), int(G2[0].coeffs[1])]
            g2_1 = [int(G2[1].coeffs[0]), int(G2[1].coeffs[0])]
            g2_int = [g2_0, g2_1]

            session["g1"] = g1_int
            session["g2"] = g2_int

            # print("Wires and Gates : {}, {}".format(numWires, numGates))
            # print("G1 : {}".format(G1))
            # print("type(G1[0]) : {}".format(type(G1[0])))
            # print("G2 : {}".format(G2))

            # print("g1 int {}".format(g1_int))
            # print("g2 int {}".format(g2_int))


            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/proving")
def main_proving():
    return render_template("groth16/proving.html")

@app.route("/groth/verifying")
def main_verifying():
    return render_template("groth16/verifying.html")