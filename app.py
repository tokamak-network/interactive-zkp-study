from flask import Flask, session, redirect, url_for
from flask import render_template
from flask import request

import ast

from zkp.groth16.code_to_r1cs import parse, extract_inputs_and_body, flatten_body, initialize_symbol, get_var_placement

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
    
    if user_code == None:
        user_code = DEFAULT_CODE
    
    return render_template('computation.html', code=user_code, ast_obj=ast_obj, flatcode=flatcode, variables=variables)
    
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
    print("flatcode function in")
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