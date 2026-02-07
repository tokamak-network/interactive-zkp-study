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
    _add_polys,
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
    rhs
)

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

DB = TinyDB('db.json')

DATA = Query()

class FR(FQ):
    field_modulus = bn128.curve_order

G1 = bn128.G1
G2 = bn128.G2

app = Flask(__name__)
app.secret_key = "key"

# PLONK Blueprint 등록
from plonk_routes import plonk_bp, init_plonk_bp
init_plonk_bp(DB)
app.register_blueprint(plonk_bp)


## Groth16 Related Functions ##

def make_target_dict(ast_obj):
    """AST의 Assign 또는 Return 노드를 딕셔너리로 변환한다.

    Args:
        ast_obj: ast.Assign 또는 ast.Return 노드.

    Returns:
        dict: Assign인 경우 {"targets": 변수명, "value": 표현식 딕셔너리},
              Return인 경우 {"value": 표현식 딕셔너리}.
    """
    if isinstance(ast_obj, ast.Assign):
        assert len(ast_obj.targets) == 1 and isinstance(ast_obj.targets[0], ast.Name)
        target = ast_obj.targets[0].id
        ast_value = ast_obj.value
        return {"targets": target, "value" : make_expr_dict(ast_value)}
    elif isinstance(ast_obj, ast.Return):
        ast_value = ast_obj.value
        return {"value" : make_expr_dict(ast_value)}

def make_expr_dict(ast_value):
    """AST 표현식 노드를 재귀적으로 딕셔너리 형태로 변환한다.

    Args:
        ast_value: ast.Name, ast.Constant, 또는 ast.BinOp 노드.

    Returns:
        str | int | dict: Name이면 변수명(str), Constant이면 숫자값(int),
                          BinOp이면 {"left": ..., "op": 연산자, "right": ...} 딕셔너리.
    """
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
    """메인 페이지. DB에서 computation 단계 데이터를 조회하여 렌더링한다.

    Input:
        DB에서 groth.computation.* 타입의 데이터를 조회 (code, ast_obj, flatcode,
        variables, abc, inputs, user_inputs, r_values, qap, qap_lcm, qap_fr, fr_modulus).

    Returns:
        렌더링된 groth16/computation.html 템플릿 (각 데이터를 템플릿 변수로 전달).
    """
    code_search = DB.search(DATA.type == "groth.computation.code")
    if code_search == []: user_code = None
    else: user_code = code_search[0]["code"]

    ast_obj_search = DB.search(DATA.type == "groth.computation.ast_obj")
    if ast_obj_search == []: ast_obj = None
    else: ast_obj = ast_obj_search[0]["ast_obj"]

    flatcode_search = DB.search(DATA.type == "groth.computation.flatcode")
    if flatcode_search == []: flatcode = None
    else: flatcode = flatcode_search[0]["flatcode"]

    variables_search = DB.search(DATA.type == "groth.computation.variables")
    if variables_search == []: variables = None
    else: variables = variables_search[0]["variables"]

    abc_search = DB.search(DATA.type == "groth.computation.abc")
    if abc_search == []: abc = None
    else: abc = abc_search[0]["abc"]

    inputs_search = DB.search(DATA.type == "groth.computation.inputs")
    if inputs_search == []: inputs = None
    else: inputs = inputs_search[0]["inputs"]

    user_inputs_search = DB.search(DATA.type == "groth.computation.user_inputs")
    if user_inputs_search == []: user_inputs = None
    else: user_inputs = user_inputs_search[0]["user_inputs"]

    r_values_search = DB.search(DATA.type == "groth.computation.r_values")
    if r_values_search == []: r_values = None
    else: r_values = r_values_search[0]["r_values"]

    qap_search = DB.search(DATA.type == "groth.computation.qap")
    if qap_search == []: qap = None
    else: qap = qap_search[0]["qap"]

    qap_lcm_search = DB.search(DATA.type == "groth.computation.qap_lcm")
    if qap_lcm_search == []: qap_lcm = None
    else: qap_lcm = qap_lcm_search[0]["qap_lcm"]

    qap_fr_search = DB.search(DATA.type == "groth.computation.qap_fr")
    if qap_fr_search == []: qap_fr = None
    else: qap_fr = qap_fr_search[0]["qap_fr"]

    fr_modulus_search = DB.search(DATA.type == "groth.computation.fr_modulus")
    if fr_modulus_search == []: fr_modulus = None
    else: fr_modulus = fr_modulus_search[0]["fr_modulus"]

    if user_code == None:
        user_code = DEFAULT_CODE

    return render_template('groth16/computation.html',
                           code=user_code,
                           ast_obj=ast_obj,
                           flatcode=flatcode,
                           variables=variables,
                           abc=abc,
                           inputs=inputs,
                           user_inputs=user_inputs,
                           r_vector=r_values,
                           qap=qap,
                           qap_lcm=qap_lcm,
                           qap_fr=qap_fr,
                           fr_modulus=fr_modulus)

@app.route("/code", methods=['POST'])
def save_code():
    """사용자가 입력한 코드를 DB에 저장한다.

    Input:
        request.form['z-code'] (str): 사용자가 입력한 Python 코드 문자열.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.
    """
    if request.method == "POST":
        session.clear()
        user_code = request.form['z-code']
        DB.upsert({"type":"groth.computation.code", "code":user_code}, DATA.type == "groth.computation.code")
        session["code"] = user_code
        return redirect(url_for('main'))

@app.route("/code/delete", methods=["POST"])
def delete_code():
    """저장된 코드와 전체 DB 데이터를 삭제한다.

    Input:
        없음 (POST 요청만 필요).

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        session 초기화, DB 전체 truncate.
    """
    if request.method == "POST":
        session.clear()
        DB.truncate()
    return redirect(url_for('main'))

@app.route("/code/ast", methods=["POST"])
def ast_table():
    """저장된 코드를 AST로 파싱하여 함수명, 입력, 본문 구조를 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.ast_obj 저장 ({"name": 함수명, "inputs": 입력 목록, "body": 본문 딕셔너리 리스트}).
    """
    if request.method == "POST":

        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

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

            session['ast_obj'] = final_out
            DB.upsert({"type":"groth.computation.ast_obj", "ast_obj":final_out}, DATA.type == "groth.computation.ast_obj")
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

def clear_flatcode():
    """DB에서 flatcode와 variables 데이터를 삭제한다.

    Input:
        없음.

    Returns:
        없음.

    Side Effects:
        DB에서 groth.computation.flatcode, groth.computation.variables 제거.
    """
    DB.remove(DATA.type == "groth.computation.flatcode")
    DB.remove(DATA.type == "groth.computation.variables")

@app.route("/flatcode/table", methods=["POST"])
def flatcode_table():
    """코드를 flatten하여 flatcode와 변수 배치를 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.flatcode (list[tuple]), groth.computation.variables (list[str]) 저장.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            variables = get_var_placement(inputs, flatcode)

            initialize_symbol()

            DB.upsert({"type":"groth.computation.flatcode", "flatcode":flatcode}, DATA.type == "groth.computation.flatcode")
            DB.upsert({"type":"groth.computation.variables", "variables":variables}, DATA.type == "groth.computation.variables")
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/r1cs/abc", methods=["POST"])
def abc_matrix():
    """Flatcode로부터 R1CS의 A, B, C 행렬을 생성하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.abc 저장 ({"A": list, "B": list, "C": list}).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            initialize_symbol()
            DB.upsert({"type":"groth.computation.abc", "abc":{"A": A, "B": B, "C": C}}, DATA.type == "groth.computation.abc")

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/r1cs/inputs", methods=["POST"])
def retrieve_values():
    """코드에서 입력 변수 목록을 추출하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.inputs 저장 (list[str]: 입력 변수명 리스트).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            DB.upsert({"type":"groth.computation.inputs", "inputs":inputs}, DATA.type == "groth.computation.inputs")
            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/r1cs/inputs/r", methods=["POST"])
def calculate_r():
    """사용자 입력값으로 witness 벡터 r을 계산하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        request.form (dict): 각 입력 변수에 대한 정수 값 (예: {"x": "3"}).

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.user_inputs (dict), groth.computation.r_values (list[int]: witness 벡터) 저장.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            form_data = request.form
            user_inputs = []
            for d in form_data:
                user_inputs.append(int(form_data[d]))
            DB.upsert({"type":"groth.computation.user_inputs", "user_inputs":form_data}, DATA.type == "groth.computation.user_inputs")

            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            r = assign_variables(inputs, user_inputs, flatcode)

            initialize_symbol()
            DB.upsert({"type":"groth.computation.r_values", "r_values":r}, DATA.type == "groth.computation.r_values")

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/qap/normal", methods=["POST"])
def create_qap():
    """R1CS를 일반 QAP(라그랑주 보간)로 변환하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.qap 저장 ({"Ap": list, "Bp": list, "Cp": list, "Z": list}).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)

            Ap, Bp, Cp, Z = r1cs_to_qap(A, B, C)
            initialize_symbol()

            DB.upsert({"type":"groth.computation.qap", "qap":{"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z}}, DATA.type == "groth.computation.qap")

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))

@app.route("/qap/lcm", methods=["POST"])
def create_qap_lcm():
    """R1CS를 LCM(행렬식) 변형 QAP로 변환하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.qap_lcm 저장 ({"Ap": list, "Bp": list, "Cp": list, "Z": list}).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)

            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            session["qap_lcm"] = {"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z}
            DB.upsert({"type":"groth.computation.qap_lcm", "qap_lcm":{"Ap" : Ap, "Bp":Bp, "Cp": Cp, "Z":Z}}, DATA.type == "groth.computation.qap_lcm")

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))


@app.route("/qap/fr", methods=["POST"])
def create_qap_fr():
    """QAP 다항식을 FR(BN128 curve order) 필드로 변환하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        DB['groth.computation.r_values'] (list[int]): witness 벡터.

    Returns:
        redirect: 메인 페이지('main')로 리다이렉트.

    Side Effects:
        DB에 groth.computation.qap_fr ({"Ax": list, "Bx": list, "Cx": list, "Zx": list, "Rx": list})
        및 groth.computation.fr_modulus (int: BN128 curve order) 저장.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        r_values_search = DB.search(DATA.type == "groth.computation.r_values")
        if r_values_search == []: r_values = None
        else: r_values = r_values_search[0]["r_values"]

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
            DB.upsert({"type":"groth.computation.qap_fr", "qap_fr":o}, DATA.type == "groth.computation.qap_fr")
            fr_modulus = int(FR.field_modulus)
            DB.upsert({"type":"groth.computation.fr_modulus", "fr_modulus":fr_modulus}, DATA.type == "groth.computation.fr_modulus")

            return redirect(url_for('main'))
        else:
            return redirect(url_for('main'))


@app.route("/groth/setup")
def main_setup():
    """Setup 페이지. DB에서 toxic waste, 다항식, sigma 등의 데이터를 조회하여 렌더링한다.

    Input:
        DB에서 groth.setup.* 타입의 데이터를 조회 (toxic, polys, polys_x_val,
        numWires, numGates, g1, g2, sigmas, gates, public_gates).

    Returns:
        렌더링된 groth16/setup.html 템플릿 (각 데이터를 템플릿 변수로 전달).
    """
    toxic_search = DB.search(DATA.type == "groth.setup.toxic")
    if toxic_search == []: toxic = None
    else: toxic = toxic_search[0]["toxic"]

    polys_search = DB.search(DATA.type == "groth.setup.polys")
    if polys_search == []: polys = None
    else: polys = polys_search[0]["polys"]

    polys_x_val_search = DB.search(DATA.type == "groth.setup.polys_x_val")
    if polys_x_val_search == []: polys_x_val = None
    else: polys_x_val = polys_x_val_search[0]["polys_x_val"]

    numWires_search = DB.search(DATA.type == "groth.setup.numWires")
    if numWires_search == []: numWires = None
    else: numWires = numWires_search[0]["numWires"]

    numGates_search = DB.search(DATA.type == "groth.setup.numGates")
    if numGates_search == []: numGates = None
    else: numGates = numGates_search[0]["numGates"]

    g1_search = DB.search(DATA.type == "groth.setup.g1")
    if g1_search == []: g1 = None
    else: g1 = g1_search[0]["g1"]

    g2_search = DB.search(DATA.type == "groth.setup.g2")
    if g2_search == []: g2 = None
    else: g2 = g2_search[0]["g2"]

    sigmas_search = DB.search(DATA.type == "groth.setup.sigmas")
    if sigmas_search == []: sigmas = None
    else: sigmas = sigmas_search[0]["sigmas"]

    gates_search = DB.search(DATA.type == "groth.setup.gates")
    if gates_search == []: gates = None
    else: gates = gates_search[0]["gates"]

    public_gates_search = DB.search(DATA.type == "groth.setup.public_gates")
    if public_gates_search == []: public_gates = None
    else: public_gates = public_gates_search[0]["public_gates"]

    return render_template("groth16/setup.html",
                           toxic=toxic,
                           polys=polys,
                           polys_x_val=polys_x_val,
                           numWires=numWires,
                           numGates=numGates,
                           g1=g1,
                           g2=g2,
                           sigmas=sigmas,
                           gates=gates,
                           public_gates=public_gates)

@app.route("/groth/setup/toxic/save", methods=["POST"])
def setup_save_toxic():
    """Trusted setup의 toxic waste 파라미터(alpha, beta, delta, gamma, x_val)를 DB에 저장한다.

    Input:
        request.form['toxic-alpha'] (str): alpha 값.
        request.form['toxic-beta'] (str): beta 값.
        request.form['toxic-delta'] (str): delta 값.
        request.form['toxic-gamma'] (str): gamma 값.
        request.form['toxic-x-val'] (str): 다항식 평가점 x 값.

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.toxic 저장 ({"alpha", "beta", "delta", "gamma", "x_val"}).
    """
    if request.method == "POST":
        toxic_alpha = request.form['toxic-alpha']
        toxic_beta = request.form['toxic-beta']
        toxic_delta = request.form['toxic-delta']
        toxic_gamma = request.form['toxic-gamma']
        toxic_x_val = request.form['toxic-x-val']

        o = {"alpha":toxic_alpha, "beta" : toxic_beta, "delta" : toxic_delta, "gamma" : toxic_gamma, "x_val": toxic_x_val}
        DB.upsert({"type":"groth.setup.toxic", "toxic":o}, DATA.type == "groth.setup.toxic")
        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/toxic/clear", methods=["POST"])
def clear_toxic():
    """Toxic waste 및 관련 setup 데이터(다항식, sigma, gates 등)를 모두 삭제한다.

    Input:
        없음 (POST 요청만 필요).

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에서 groth.setup.toxic, polys, polys_x_val, numWires, numGates, g1, g2, gates, public_gates, sigmas 제거.
    """
    if request.method == "POST":
        DB.remove(DATA.type == "groth.setup.toxic")
        DB.remove(DATA.type == "groth.setup.polys")
        DB.remove(DATA.type == "groth.setup.polys_x_val")
        DB.remove(DATA.type == "groth.setup.numWires")
        DB.remove(DATA.type == "groth.setup.numGates")
        DB.remove(DATA.type == "groth.setup.g1")
        DB.remove(DATA.type == "groth.setup.g2")
        DB.remove(DATA.type == "groth.setup.gates")
        DB.remove(DATA.type == "groth.setup.public_gates")
        DB.remove(DATA.type == "groth.setup.sigmas")
        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/gates", methods=["POST"])
def load_gates():
    """코드에서 변수 배치(gates)를 추출하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.gates 저장 (list[str]: 변수명 배치 순서, 예: ['~one', 'x', '~out', ...]).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            variables = get_var_placement(inputs, flatcode)
            initialize_symbol()
            DB.upsert({"type":"groth.setup.gates", "gates":variables}, DATA.type == "groth.setup.gates")
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/gates/set", methods=["POST"])
def set_public_gates():
    """사용자가 선택한 공개(public) 게이트 인덱스를 DB에 저장한다.

    Input:
        DB['groth.setup.gates'] (list[str]): 변수 배치 목록.
        request.form['form-check-input-{i}'] (str | None): 체크된 게이트의 폼 값 (0번은 항상 포함).

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.public_gates 저장 (list[int]: 공개 게이트 인덱스, 예: [0, 1]).
    """
    if request.method == "POST":
        gates_search = DB.search(DATA.type == "groth.setup.gates")
        if gates_search == []: gates = None
        else: gates = gates_search[0]["gates"]

        if gates:
            target = [0]
            for i in range(len(gates)-1):
                check = request.form.get("form-check-input-"+str(i+1))
                if check != None:
                    target.append(i+1)
            DB.upsert({"type":"groth.setup.public_gates", "public_gates":target}, DATA.type == "groth.setup.public_gates")
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/gates/reset", methods=["POST"])
def reset_public_gates():
    """공개 게이트 설정과 sigma를 초기화한다.

    Input:
        없음 (POST 요청만 필요).

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에서 groth.setup.public_gates, groth.setup.sigmas 제거.
    """
    if request.method == "POST":
        DB.remove(DATA.type == "groth.setup.public_gates")
        DB.remove(DATA.type == "groth.setup.sigmas")
        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys", methods=["POST"])
def get_polys():
    """R1CS에서 QAP(LCM) 다항식을 생성하고 FR 필드로 변환하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.polys 저장 ({"Ap": list[list[int]], "Bp": list[list[int]],
        "Cp": list[list[int]], "Zp": list[int]} — FR 필드 값).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

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

            o = {"Ap": Ax, "Bp": Bx, "Cp":Cx, "Zp":Zx}
            DB.upsert({"type":"groth.setup.polys", "polys":o}, DATA.type == "groth.setup.polys")
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys/evaluated", methods=["POST"])
def get_polys_evaluated():
    """QAP 다항식을 toxic x_val에서 평가한 값을 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        DB['groth.setup.toxic'] (dict): toxic waste 파라미터 (x_val 사용).

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.polys_x_val 저장 ({"Ax_val": list[int], "Bx_val": list[int],
        "Cx_val": list[int], "Zx_val": int} — x_val에서 평가된 다항식 값).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        toxic_search = DB.search(DATA.type == "groth.setup.toxic")
        if toxic_search == []: toxic = None
        else: toxic = toxic_search[0]["toxic"]

        if user_code:
            x_val = FR(int(toxic["x_val"]))

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

            o = {"Ax_val": Ax_val_int, "Bx_val": Bx_val_int, "Cx_val":Cx_val_int, "Zx_val":Zx_val_int}
            DB.upsert({"type":"groth.setup.polys_x_val", "polys_x_val":o}, DATA.type == "groth.setup.polys_x_val")
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/polys/clear", methods=["POST"])
def clear_polys():
    """다항식 및 평가값 데이터를 DB에서 삭제한다.

    Input:
        없음 (POST 요청만 필요).

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에서 groth.setup.polys, groth.setup.polys_x_val 제거.
    """
    if request.method == "POST":
        DB.remove(DATA.type == "groth.setup.polys")
        DB.remove(DATA.type == "groth.setup.polys_x_val")
        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/sigma/formula", methods=["POST"])
def sigma_formula():
    """와이어/게이트 수와 G1, G2 생성자 정보를 계산하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.numWires (int), groth.setup.numGates (int),
        groth.setup.g1 (list[int]: G1 좌표), groth.setup.g2 (list[list[int]]: G2 좌표) 저장.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)
            A, B, C = flatcode_to_r1cs(inputs, flatcode)
            Ap, Bp, Cp, Z = r1cs_to_qap_times_lcm(A, B, C)
            initialize_symbol()

            numWires = getNumWires(Ap)
            numGates = getNumGates(Ap)
            DB.upsert({"type":"groth.setup.numWires", "numWires":numWires}, DATA.type == "groth.setup.numWires")
            DB.upsert({"type":"groth.setup.numGates", "numGates":numGates}, DATA.type == "groth.setup.numGates")

            g1_int = [int(f) for f in G1]
            g2_0 = [int(G2[0].coeffs[0]), int(G2[0].coeffs[1])]
            g2_1 = [int(G2[1].coeffs[0]), int(G2[1].coeffs[0])]
            g2_int = [g2_0, g2_1]

            DB.upsert({"type":"groth.setup.g1", "g1":g1_int}, DATA.type == "groth.setup.g1")
            DB.upsert({"type":"groth.setup.g2", "g2":g2_int}, DATA.type == "groth.setup.g2")

            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/sigma/calc", methods=["POST"])
def calculate_sigmas():
    """Toxic waste와 다항식 평가값으로 sigma1_1~sigma2_2를 계산하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        DB['groth.setup.toxic'] (dict): toxic waste 파라미터 (alpha, beta, delta, gamma, x_val).
        DB['groth.setup.public_gates'] (list[int]): 공개 게이트 인덱스.

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에 groth.setup.sigmas 저장 ({"1_1"~"1_5": list[list[int]] (G1 포인트),
        "2_1", "2_2": list[list[list[int]]] (G2 포인트)}).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        toxic_search = DB.search(DATA.type == "groth.setup.toxic")
        if toxic_search == []: toxic = None
        else: toxic = toxic_search[0]["toxic"]

        public_gates_search = DB.search(DATA.type == "groth.setup.public_gates")
        if public_gates_search == []: public_gates = None
        else: public_gates = public_gates_search[0]["public_gates"]

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

            numGates = getNumGates(Ax)
            numWires = getNumWires(Ax)

            x_val = FR(int(toxic["x_val"]))
            alpha = FR(int(toxic["alpha"]))
            beta = FR(int(toxic["beta"]))
            delta = FR(int(toxic["delta"]))
            gamma = FR(int(toxic["gamma"]))

            Ax_val = ax_val(Ax, x_val)
            Bx_val = bx_val(Bx, x_val)
            Cx_val = cx_val(Cx, x_val)
            Zx_val = zx_val(Zx, x_val)

            s11 = sigma11(alpha, beta, delta)
            s12 = sigma12(numGates, x_val)
            s13, VAL = sigma13(numWires, alpha, beta, gamma, Ax_val, Bx_val, Cx_val, public_gates)
            s14 = sigma14(numWires, alpha, beta, delta, Ax_val, Bx_val, Cx_val, public_gates)
            s15 = sigma15(numGates, delta, x_val, Zx_val)
            s21 = sigma21(beta, delta, gamma)
            s22 = sigma22(numGates, x_val)

            def turn_point_int(li):
                """EC 포인트의 좌표를 정수 리스트로 변환한다.

                Args:
                    li (tuple[FQ, FQ]): G1 타원곡선 포인트 (FQ 좌표 튜플).

                Returns:
                    list[int]: 정수 좌표 리스트 [x, y].
                """
                return [int(num) for num in li]

            def turn_g2_int(g2p):
                """G2 포인트를 정수 리스트로 변환한다.

                Args:
                    g2p (tuple[FQ2, FQ2]): G2 타원곡선 포인트 (FQ2 좌표 튜플).

                Returns:
                    list[list[int]]: [[x_re, x_im], [y_re, y_im]] 형태의 정수 리스트.
                """
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

            o = {"1_1":s11_int, "1_2":s12_int, "1_3":s13_int, "1_4":s14_int, "1_5":s15_int, "2_1":s21_int, "2_2":s22_int}
            DB.upsert({"type":"groth.setup.sigmas", "sigmas":o}, DATA.type == "groth.setup.sigmas")
            return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

@app.route("/groth/setup/sigma/clear", methods=["POST"])
def clear_sigmas():
    """와이어/게이트 수, G1/G2 생성자, sigma 데이터를 DB에서 삭제한다.

    Input:
        없음 (POST 요청만 필요).

    Returns:
        redirect: setup 페이지('main_setup')로 리다이렉트.

    Side Effects:
        DB에서 groth.setup.numWires, numGates, g1, g2, sigmas 제거.
    """
    if request.method == "POST":
        DB.remove(DATA.type == "groth.setup.numWires")
        DB.remove(DATA.type == "groth.setup.numGates")
        DB.remove(DATA.type == "groth.setup.g1")
        DB.remove(DATA.type == "groth.setup.g2")
        DB.remove(DATA.type == "groth.setup.sigmas")

        return redirect(url_for('main_setup'))
    else:
        return redirect(url_for('main_setup'))

#### PROVING ####

@app.route("/groth/proving")
def main_proving():
    """Proving 페이지. DB에서 prover 관련 데이터를 조회하여 렌더링한다.

    Input:
        DB에서 groth.proving.* 타입의 데이터를 조회 (prover_random, prover_input_form,
        inputs, user_inputs, r_values) 및 groth.setup.public_gates.

    Returns:
        렌더링된 groth16/proving.html 템플릿 (각 데이터를 템플릿 변수로 전달).
    """
    prover_random_search = DB.search(DATA.type == "groth.proving.prover_random")
    if prover_random_search == []: p_random = None
    else: p_random = prover_random_search[0]["prover_random"]

    prover_input_form_search = DB.search(DATA.type == "groth.proving.prover_input_form")
    if prover_input_form_search == []: p_inputs_is_load = None
    else: p_inputs_is_load = prover_input_form_search[0]["prover_input_form"]

    inputs_search = DB.search(DATA.type == "groth.proving.inputs")
    if inputs_search == []: inputs = None
    else: inputs = inputs_search[0]["inputs"]

    user_inputs_search = DB.search(DATA.type == "groth.proving.user_inputs")
    if user_inputs_search == []: user_inputs = None
    else: user_inputs = user_inputs_search[0]["user_inputs"]

    r_values_search = DB.search(DATA.type == "groth.proving.r_values")
    if r_values_search == []: r_values = None
    else: r_values = r_values_search[0]["r_values"]

    public_gates_search = DB.search(DATA.type == "groth.setup.public_gates")
    if public_gates_search == []: public_gates = None
    else: public_gates = public_gates_search[0]["public_gates"]

    proofs = DB.search(DATA.type == "groth.proving.proofs")

    return render_template("groth16/proving.html",
                           p_random=p_random,
                           p_input_is_load=p_inputs_is_load,
                           inputs=inputs,
                           user_inputs=user_inputs,
                           r_values=r_values,
                           public_gates=public_gates,
                           proofs=proofs)

@app.route("/groth/proving/random/save", methods=["POST"])
def save_prover_random():
    """Prover의 랜덤 값(r, s)을 DB에 저장한다.

    Input:
        request.form['prover-random-r'] (str): prover 랜덤 값 r.
        request.form['prover-random-s'] (str): prover 랜덤 값 s.

    Returns:
        redirect: proving 페이지('main_proving')로 리다이렉트.

    Side Effects:
        DB에 groth.proving.prover_random 저장 ({"r": int, "s": int}).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            random_r = request.form['prover-random-r']
            random_s = request.form['prover-random-s']
            o = {"r" : int(random_r), "s" : int(random_s)}
            DB.upsert({"type":"groth.proving.prover_random", "prover_random":o}, DATA.type == "groth.proving.prover_random")
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/random/clear", methods=["POST"])
def clear_prover_random():
    """Prover 관련 데이터(랜덤 값, 입력, witness, proof)를 모두 삭제한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드 (존재 확인용).

    Returns:
        redirect: proving 페이지('main_proving')로 리다이렉트.

    Side Effects:
        DB에서 groth.proving.prover_random, prover_input_form, inputs, user_inputs, r_values, proofs 제거.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            DB.remove(DATA.type == "groth.proving.prover_random")
            DB.remove(DATA.type == "groth.proving.prover_input_form")
            DB.remove(DATA.type == "groth.proving.inputs")
            DB.remove(DATA.type == "groth.proving.user_inputs")
            DB.remove(DATA.type == "groth.proving.r_values")
            DB.remove(DATA.type == "groth.proving.proofs")

            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/inputs", methods=["POST"])
def load_prover_input():
    """코드에서 입력 변수를 추출하여 prover 입력 폼을 활성화한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.

    Returns:
        redirect: proving 페이지('main_proving')로 리다이렉트.

    Side Effects:
        DB에 groth.proving.prover_input_form (bool: True),
        groth.proving.inputs (list[str]: 입력 변수명 리스트) 저장.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            inputs, body = extract_inputs_and_body(parse(user_code))
            DB.upsert({"type":"groth.proving.prover_input_form", "prover_input_form":True}, DATA.type == "groth.proving.prover_input_form")
            DB.upsert({"type":"groth.proving.inputs", "inputs":inputs}, DATA.type == "groth.proving.inputs")
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/witness/calc", methods=["POST"])
def calculate_witness():
    """사용자 입력값으로 witness 벡터 r을 계산하여 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        request.form (dict): 각 입력 변수에 대한 정수 값 (예: {"x": "3"}).

    Returns:
        redirect: proving 페이지('main_proving')로 리다이렉트.

    Side Effects:
        DB에 groth.proving.r_values (list[int]: witness 벡터),
        groth.proving.user_inputs (dict: 사용자 입력 폼 데이터) 저장.
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        if user_code:
            form_data = request.form
            user_inputs = []
            for d in form_data:
                user_inputs.append(int(form_data[d]))

            inputs, body = extract_inputs_and_body(parse(user_code))
            flatcode = flatten_body(body)

            r = assign_variables(inputs, user_inputs, flatcode)
            initialize_symbol()

            DB.upsert({"type":"groth.proving.r_values", "r_values":r}, DATA.type == "groth.proving.r_values")
            DB.upsert({"type":"groth.proving.user_inputs", "user_inputs":form_data}, DATA.type == "groth.proving.user_inputs")
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

@app.route("/groth/proving/proof/generate", methods=["POST"])
def generate_proof():
    """Sigma, witness, 랜덤 값을 사용하여 Groth16 proof (A, B, C)를 생성하고 DB에 저장한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        DB['groth.proving.user_inputs'] (dict): 사용자 입력 값.
        DB['groth.setup.public_gates'] (list[int]): 공개 게이트 인덱스.
        DB['groth.proving.prover_random'] (dict): prover 랜덤 값 {"r": int, "s": int}.
        DB['groth.setup.sigmas'] (dict): sigma1_1~sigma2_2 EC 포인트.
        DB['groth.setup.polys'] (dict): QAP 다항식 (FR 필드 정수 값).

    Returns:
        redirect: proving 페이지('main_proving')로 리다이렉트.

    Side Effects:
        DB에 groth.proving.proofs 저장 ({"proof_a": list[int] (G1), "proof_b": list[list[int]] (G2),
        "proof_c": list[int] (G1)}).
    """
    if request.method == "POST":
        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        user_inputs_search = DB.search(DATA.type == "groth.proving.user_inputs")
        if user_inputs_search == []: user_inputs = None
        else: user_inputs = user_inputs_search[0]["user_inputs"]

        public_gates_search = DB.search(DATA.type == "groth.setup.public_gates")
        if public_gates_search == []: public_gates = None
        else: public_gates = public_gates_search[0]["public_gates"]

        prover_random_search = DB.search(DATA.type == "groth.proving.prover_random")
        if prover_random_search == []: prover_random = None
        else: prover_random = prover_random_search[0]["prover_random"]

        user_inputs_li = [int(user_inputs[i]) for i in user_inputs]

        sigmas_search = DB.search(DATA.type == "groth.setup.sigmas")
        if sigmas_search == []: sigmas = None
        else: sigmas = sigmas_search[0]["sigmas"]

        polys_search = DB.search(DATA.type == "groth.setup.polys")
        if polys_search == []: polys = None
        else: polys = polys_search[0]["polys"]

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

            def turn_g1_int(g1p):
                """G1 포인트를 정수 리스트로 변환한다.

                Args:
                    g1p (tuple[FQ, FQ]): G1 타원곡선 포인트.

                Returns:
                    list[int]: [x, y] 정수 좌표 리스트.
                """
                return [int(num) for num in g1p]

            def turn_g2_int(g2p):
                """G2 포인트를 정수 리스트로 변환한다.

                Args:
                    g2p (tuple[FQ2, FQ2]): G2 타원곡선 포인트.

                Returns:
                    list[list[int]]: [[x_re, x_im], [y_re, y_im]] 정수 리스트.
                """
                g2p0 = [int(g2p[0].coeffs[0]), int(g2p[0].coeffs[1])]
                g2p1 = [int(g2p[1].coeffs[0]), int(g2p[1].coeffs[1])]
                g2_int = [g2p0, g2p1]
                return g2_int

            def turn_g2_fq2(g2p_int):
                """정수 리스트를 FQ2 타입의 G2 포인트로 변환한다.

                Args:
                    g2p_int (list[list[int]]): [[re, im], [re, im]] 형태의 정수 리스트.

                Returns:
                    tuple[FQ2, FQ2]: FQ2 좌표 튜플의 G2 포인트.
                """
                g2p0 = bn128.FQ2(g2p_int[0])
                g2p1 = bn128.FQ2(g2p_int[1])
                return (g2p0, g2p1)

            def turn_g1_fq(g1_int):
                """정수 리스트를 FQ 타입의 G1 포인트로 변환한다.

                Args:
                    g1_int (list[int]): [x, y] 정수 좌표 리스트.

                Returns:
                    tuple[FQ, FQ]: FQ 좌표 튜플의 G1 포인트.
                """
                return (bn128.FQ(g1_int[0]), bn128.FQ(g1_int[1]))

            sigma1_1 = [turn_g1_fq(point) for point in sigmas["1_1"]]
            sigma1_2 = [turn_g1_fq(point) for point in sigmas["1_2"]]
            sigma1_4 = [turn_g1_fq(point) for point in sigmas["1_4"]]
            sigma1_5 = [turn_g1_fq(point) for point in sigmas["1_5"]]
            sigma2_1 = [turn_g2_fq2(point) for point in sigmas["2_1"]]
            sigma2_2 = [turn_g2_fq2(point) for point in sigmas["2_2"]]

            prf_a = proof_a(sigma1_1, sigma1_2, Ax, Rx, r)
            prf_b = proof_b(sigma2_1, sigma2_2, Bx, Rx, s)
            prf_c = proof_c(sigma1_1, sigma1_2, sigma1_4, sigma1_5, Bx, Rx, Hx, s, r, prf_a, public_gates)

            o = {"type": "groth.proving.proofs", "proof_a" : turn_g1_int(prf_a), "proof_b" : turn_g2_int(prf_b), "proof_c" : turn_g1_int(prf_c)}
            DB.upsert(o, DATA.type == "groth.proving.proofs")
            return redirect(url_for('main_proving'))
    else:
        return redirect(url_for('main_proving'))

#### VERIFYING ####

@app.route("/groth/verifying")
def main_verifying():
    """Verifying 페이지. DB에서 proof와 공개 게이트 데이터를 조회하여 렌더링한다.

    Input:
        DB['groth.setup.public_gates'] (list[int]): 공개 게이트 인덱스.
        DB['groth.proving.r_values'] (list[int]): witness 벡터.
        DB['groth.proving.proofs'] (dict): proof_a, proof_b, proof_c.

    Returns:
        렌더링된 groth16/verifying.html 템플릿 (proofs, public_gates를 템플릿 변수로 전달).
    """
    public_gates_search = DB.search(DATA.type == "groth.setup.public_gates")
    if public_gates_search == []: public_gates_index = None
    else: public_gates_index = public_gates_search[0]["public_gates"]

    r_values_search = DB.search(DATA.type == "groth.proving.r_values")
    if r_values_search == []: r_values = None
    else: r_values = r_values_search[0]["r_values"]

    proofs = DB.search(DATA.type == "groth.proving.proofs")
    public_gates = [r_values[i] for i in public_gates_index]

    return render_template("groth16/verifying.html", proofs=proofs, public_gates=public_gates)

@app.route("/groth/verifying/verify", methods=["POST"])
def groth_verify():
    """Proof A, B, C와 sigma를 사용하여 Groth16 페어링 검증을 수행한다.

    Input:
        DB['groth.computation.code'] (str): 저장된 사용자 코드.
        DB['groth.setup.sigmas'] (dict): sigma1_1~sigma2_2 EC 포인트.
        DB['groth.setup.public_gates'] (list[int]): 공개 게이트 인덱스.
        DB['groth.proving.r_values'] (list[int]): witness 벡터.
        DB['groth.proving.proofs'] (dict): proof_a (G1), proof_b (G2), proof_c (G1).

    Returns:
        redirect: verifying 페이지('main_verifying')로 리다이렉트.

    Side Effects:
        verify() 호출로 페어링 검증 수행 (결과: bool).
    """
    if request.method == "POST":

        code_search = DB.search(DATA.type == "groth.computation.code")
        if code_search == []: user_code = None
        else: user_code = code_search[0]["code"]

        sigmas_search = DB.search(DATA.type == "groth.setup.sigmas")
        if sigmas_search == []: sigmas = None
        else: sigmas = sigmas_search[0]["sigmas"]

        public_gates_search = DB.search(DATA.type == "groth.setup.public_gates")
        if public_gates_search == []: public_gates_index = None
        else: public_gates_index = public_gates_search[0]["public_gates"]

        r_values_search = DB.search(DATA.type == "groth.proving.r_values")
        if r_values_search == []: r_values = None
        else: r_values = r_values_search[0]["r_values"]
        public_gates = [(i, r_values[i]) for i in public_gates_index]

        if user_code:
            def turn_g2_fq2(g2p_int):
                """정수 리스트를 FQ2 타입의 G2 포인트로 변환한다.

                Args:
                    g2p_int (list[list[int]]): [[re, im], [re, im]] 형태의 정수 리스트.

                Returns:
                    tuple[FQ2, FQ2]: FQ2 좌표 튜플의 G2 포인트.
                """
                g2p0 = bn128.FQ2(g2p_int[0])
                g2p1 = bn128.FQ2(g2p_int[1])
                return (g2p0, g2p1)

            def turn_g1_fq(g1_int):
                """정수 리스트를 FQ 타입의 G1 포인트로 변환한다.

                Args:
                    g1_int (list[int]): [x, y] 정수 좌표 리스트.

                Returns:
                    tuple[FQ, FQ]: FQ 좌표 튜플의 G1 포인트.
                """
                return (bn128.FQ(g1_int[0]), bn128.FQ(g1_int[1]))

            proofs = DB.search(DATA.type == "groth.proving.proofs")

            proof_a_int = [int(st) for st in proofs[0]["proof_a"]]
            proof_b_int = [[int(num) for num in vec] for vec in proofs[0]["proof_b"]]
            proof_c_int = [int(st) for st in proofs[0]["proof_c"]]

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

            verify_result = verify(proof_a, proof_b, proof_c, sigma1_1, sigma1_3, sigma2_1, public_gates)

            return redirect(url_for('main_verifying'))
    else:
        return redirect(url_for('main_verifying'))


if __name__ == "__main__":
    app.run(port=5001, debug=True)
