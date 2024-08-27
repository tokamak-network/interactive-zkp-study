from flask import Flask, session, redirect, url_for
from flask import render_template
from flask import request

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
    
    if user_code == None:
        user_code = DEFAULT_CODE
    
    return render_template('computation.html', \
                           code=user_code, \
                           ast_obj=ast_obj, \
                           flatcode=flatcode, \
                           variables=variables, \
                           abc=abc, \
                           inputs=inputs, \
                           user_inputs=user_inputs, \
                           r_vector=r_vector, \
                           qap=qap, \
                           qap_lcm=qap_lcm \
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