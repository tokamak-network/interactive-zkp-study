"""
PLONK Flask Blueprint — 모든 PLONK 엔드포인트
================================================

4개 페이지: Circuit, Setup, Proving, Verifying
총 23개 엔드포인트 (GET 4 + POST 19)
"""

from flask import Blueprint, render_template, redirect, url_for, request
from tinydb import Query

from zkp.plonk.field import FR, CURVE_ORDER
from zkp.plonk.circuit import Circuit, Gate
from zkp.plonk.srs import SRS
from zkp.plonk.preprocessor import preprocess
from zkp.plonk.prover import Proof, ProverState, prove
from zkp.plonk.prover import round1, round2, round3, round4, round5
from zkp.plonk.verifier import verify
from zkp.plonk.transcript import Transcript

from plonk_serializers import (
    serialize_fr, deserialize_fr,
    serialize_g1, deserialize_g1,
    serialize_g2, deserialize_g2,
    serialize_poly, deserialize_poly,
    serialize_fr_list, deserialize_fr_list,
    serialize_transcript, deserialize_transcript,
    serialize_srs, deserialize_srs,
    serialize_preprocessed, deserialize_preprocessed,
    serialize_proof, deserialize_proof,
    g1_short, g2_short, fr_short,
)

plonk_bp = Blueprint('plonk', __name__, url_prefix='/plonk')

DATA = Query()

# DB는 app.py에서 주입
DB = None


def init_plonk_bp(db):
    """app.py에서 DB를 주입받는다."""
    global DB
    DB = db


# ─── DB 헬퍼 ───

def db_get(key):
    """DB에서 키로 데이터를 조회한다."""
    result = DB.search(DATA.type == key)
    if not result:
        return None
    return result[0].get("data")


def db_set(key, data):
    """DB에 키로 데이터를 저장한다."""
    DB.upsert({"type": key, "data": data}, DATA.type == key)


def db_remove(key):
    """DB에서 키를 삭제한다."""
    DB.remove(DATA.type == key)


def db_remove_prefix(prefix):
    """prefix로 시작하는 모든 키를 삭제한다."""
    DB.remove(DATA.type.test(lambda t: t.startswith(prefix)))


# ──────────────────────────────────────────────────────────────
# Circuit 페이지
# ──────────────────────────────────────────────────────────────

@plonk_bp.route("/circuit")
def circuit_page():
    """회로 페이지 렌더."""
    circuit_data = db_get("plonk.circuit.data")
    gates_table = db_get("plonk.circuit.gates_table")
    copy_constraints = db_get("plonk.circuit.copy_constraints")
    witness_table = db_get("plonk.circuit.witness_table")
    gate_check = db_get("plonk.circuit.gate_check")

    return render_template("plonk/circuit.html",
                           circuit_data=circuit_data,
                           gates_table=gates_table,
                           copy_constraints=copy_constraints,
                           witness_table=witness_table,
                           gate_check=gate_check)


@plonk_bp.route("/circuit/load-example", methods=["POST"])
def circuit_load_example():
    """x³+x+5=35 예제 회로를 로드한다."""
    circuit, a_vals, b_vals, c_vals, public_inputs = (
        Circuit.x3_plus_x_plus_5_eq_35()
    )

    # 게이트 테이블 구성
    gates_table = []
    gate_types = ["mul: x·x = x²", "mul: x²·x = x³", "add: x³+x", "add+c: +5=35"]
    for i, gate in enumerate(circuit.gates):
        gates_table.append({
            "index": i,
            "type": gate_types[i] if i < len(gate_types) else "padding",
            "q_L": str(int(gate.q_l)),
            "q_R": str(int(gate.q_r)),
            "q_O": str(int(gate.q_o)),
            "q_M": str(int(gate.q_m)),
            "q_C": str(int(gate.q_c)),
        })

    # Copy constraint 테이블
    wire_names = ["a", "b", "c"]
    copy_constraints = []
    for g1, w1, g2, w2 in circuit.copy_constraints:
        copy_constraints.append({
            "left": f"gate{g1}.{wire_names[w1]}",
            "right": f"gate{g2}.{wire_names[w2]}",
        })

    # 위트니스 테이블
    witness_table = []
    for i in range(circuit.n):
        witness_table.append({
            "index": i,
            "a": str(int(a_vals[i])),
            "b": str(int(b_vals[i])),
            "c": str(int(c_vals[i])),
        })

    # 회로 데이터 (다른 페이지에서 사용)
    circuit_info = {
        "n": circuit.n,
        "num_public_inputs": circuit.num_public_inputs,
        "public_inputs": serialize_fr_list(public_inputs),
        "a_vals": serialize_fr_list(a_vals),
        "b_vals": serialize_fr_list(b_vals),
        "c_vals": serialize_fr_list(c_vals),
        "copy_constraints_raw": circuit.copy_constraints,
    }

    db_set("plonk.circuit.data", circuit_info)
    db_set("plonk.circuit.gates_table", gates_table)
    db_set("plonk.circuit.copy_constraints", copy_constraints)
    db_set("plonk.circuit.witness_table", witness_table)

    return redirect(url_for("plonk.circuit_page"))


@plonk_bp.route("/circuit/check-gates", methods=["POST"])
def circuit_check_gates():
    """게이트 제약을 검증한다."""
    circuit_data = db_get("plonk.circuit.data")
    if not circuit_data:
        return redirect(url_for("plonk.circuit_page"))

    circuit, a_vals, b_vals, c_vals, _ = Circuit.x3_plus_x_plus_5_eq_35()

    results = []
    for i, gate in enumerate(circuit.gates):
        ok = gate.check(a_vals[i], b_vals[i], c_vals[i])
        results.append({"index": i, "ok": ok})

    db_set("plonk.circuit.gate_check", results)
    return redirect(url_for("plonk.circuit_page"))


@plonk_bp.route("/circuit/clear", methods=["POST"])
def circuit_clear():
    """모든 PLONK 데이터를 클리어한다."""
    db_remove_prefix("plonk.")
    return redirect(url_for("plonk.circuit_page"))


# ──────────────────────────────────────────────────────────────
# Setup 페이지
# ──────────────────────────────────────────────────────────────

@plonk_bp.route("/setup")
def setup_page():
    """Setup 페이지 렌더."""
    srs_info = db_get("plonk.srs.info")
    preprocess_info = db_get("plonk.preprocess.info")

    return render_template("plonk/setup.html",
                           srs_info=srs_info,
                           preprocess_info=preprocess_info)


@plonk_bp.route("/setup/srs", methods=["POST"])
def setup_srs():
    """SRS를 생성한다."""
    circuit_data = db_get("plonk.circuit.data")
    if not circuit_data:
        return redirect(url_for("plonk.setup_page"))

    seed_str = request.form.get("srs-seed", "12345")
    seed = int(seed_str) if seed_str else 12345

    n = circuit_data["n"]
    max_degree = 3 * n + 10

    srs = SRS.generate(max_degree=max_degree, seed=seed)

    # SRS 직렬화 저장
    db_set("plonk.srs.raw", serialize_srs(srs))

    # 표시용 정보
    g1_samples = [g1_short(srs.g1_powers[i]) for i in range(min(5, len(srs.g1_powers)))]
    srs_info = {
        "seed": seed,
        "max_degree": max_degree,
        "g1_count": len(srs.g1_powers),
        "g1_samples": g1_samples,
        "g2_0": g2_short(srs.g2_powers[0]) if srs.g2_powers else None,
        "g2_1": g2_short(srs.g2_powers[1]) if len(srs.g2_powers) > 1 else None,
    }
    db_set("plonk.srs.info", srs_info)

    # SRS 변경 시 하위 데이터 클리어
    db_remove_prefix("plonk.preprocess.")
    db_remove_prefix("plonk.prover.")
    db_remove_prefix("plonk.verify.")

    return redirect(url_for("plonk.setup_page"))


@plonk_bp.route("/setup/preprocess", methods=["POST"])
def setup_preprocess():
    """회로 전처리를 실행한다."""
    circuit_data = db_get("plonk.circuit.data")
    srs_raw = db_get("plonk.srs.raw")
    if not circuit_data or not srs_raw:
        return redirect(url_for("plonk.setup_page"))

    srs = deserialize_srs(srs_raw)

    # 회로 재구성
    circuit, a_vals, b_vals, c_vals, public_inputs = Circuit.x3_plus_x_plus_5_eq_35()
    preprocessed = preprocess(circuit, srs)

    # 전처리 결과 저장
    db_set("plonk.preprocess.raw", serialize_preprocessed(preprocessed))

    # 표시용 정보
    preprocess_info = {
        "n": preprocessed.n,
        "omega": fr_short(preprocessed.omega),
        "domain": [fr_short(d) for d in preprocessed.domain],
        # 셀렉터 커밋먼트
        "q_l_comm": g1_short(preprocessed.q_l_comm),
        "q_r_comm": g1_short(preprocessed.q_r_comm),
        "q_o_comm": g1_short(preprocessed.q_o_comm),
        "q_m_comm": g1_short(preprocessed.q_m_comm),
        "q_c_comm": g1_short(preprocessed.q_c_comm),
        # 순열 커밋먼트
        "s_sigma1_comm": g1_short(preprocessed.s_sigma1_comm),
        "s_sigma2_comm": g1_short(preprocessed.s_sigma2_comm),
        "s_sigma3_comm": g1_short(preprocessed.s_sigma3_comm),
        # 순열 배열
        "sigma": preprocessed.sigma,
    }
    db_set("plonk.preprocess.info", preprocess_info)

    # 전처리 변경 시 하위 데이터 클리어
    db_remove_prefix("plonk.prover.")
    db_remove_prefix("plonk.verify.")

    return redirect(url_for("plonk.setup_page"))


@plonk_bp.route("/setup/clear-srs", methods=["POST"])
def setup_clear_srs():
    """SRS + 하위 데이터를 클리어한다."""
    db_remove_prefix("plonk.srs.")
    db_remove_prefix("plonk.preprocess.")
    db_remove_prefix("plonk.prover.")
    db_remove_prefix("plonk.verify.")
    return redirect(url_for("plonk.setup_page"))


@plonk_bp.route("/setup/clear-preprocess", methods=["POST"])
def setup_clear_preprocess():
    """전처리 + 하위 데이터를 클리어한다."""
    db_remove_prefix("plonk.preprocess.")
    db_remove_prefix("plonk.prover.")
    db_remove_prefix("plonk.verify.")
    return redirect(url_for("plonk.setup_page"))


# ──────────────────────────────────────────────────────────────
# Proving 페이지
# ──────────────────────────────────────────────────────────────

def _rebuild_prover_state_up_to(round_num):
    """지정 라운드 직전까지의 ProverState를 DB에서 재구성한다.

    Returns:
        (ProverState, SRS, PreprocessedData) 또는 None (데이터 부족 시)
    """
    circuit_data = db_get("plonk.circuit.data")
    srs_raw = db_get("plonk.srs.raw")
    pp_raw = db_get("plonk.preprocess.raw")
    if not circuit_data or not srs_raw or not pp_raw:
        return None

    srs = deserialize_srs(srs_raw)
    pp = deserialize_preprocessed(pp_raw)

    a_vals = deserialize_fr_list(circuit_data["a_vals"])
    b_vals = deserialize_fr_list(circuit_data["b_vals"])
    c_vals = deserialize_fr_list(circuit_data["c_vals"])
    public_inputs = deserialize_fr_list(circuit_data["public_inputs"])

    state = ProverState(a_vals, b_vals, c_vals, public_inputs, pp, srs)

    # Round 1 복원
    if round_num >= 2:
        r1 = db_get("plonk.prover.round1")
        if not r1:
            return None
        state.a_poly = deserialize_poly(r1["a_poly"])
        state.b_poly = deserialize_poly(r1["b_poly"])
        state.c_poly = deserialize_poly(r1["c_poly"])
        state.pi_poly = deserialize_poly(r1["pi_poly"])
        state.proof.a_comm = deserialize_g1(r1["a_comm"])
        state.proof.b_comm = deserialize_g1(r1["b_comm"])
        state.proof.c_comm = deserialize_g1(r1["c_comm"])
        state.transcript = deserialize_transcript(r1["transcript"])

    # Round 2 복원
    if round_num >= 3:
        r2 = db_get("plonk.prover.round2")
        if not r2:
            return None
        state.beta = deserialize_fr(r2["beta"])
        state.gamma = deserialize_fr(r2["gamma"])
        state.z_poly = deserialize_poly(r2["z_poly"])
        state.proof.z_comm = deserialize_g1(r2["z_comm"])
        state.transcript = deserialize_transcript(r2["transcript"])

    # Round 3 복원
    if round_num >= 4:
        r3 = db_get("plonk.prover.round3")
        if not r3:
            return None
        state.alpha = deserialize_fr(r3["alpha"])
        state.t_lo_poly = deserialize_poly(r3["t_lo_poly"])
        state.t_mid_poly = deserialize_poly(r3["t_mid_poly"])
        state.t_hi_poly = deserialize_poly(r3["t_hi_poly"])
        state.proof.t_lo_comm = deserialize_g1(r3["t_lo_comm"])
        state.proof.t_mid_comm = deserialize_g1(r3["t_mid_comm"])
        state.proof.t_hi_comm = deserialize_g1(r3["t_hi_comm"])
        state.transcript = deserialize_transcript(r3["transcript"])

    # Round 4 복원
    if round_num >= 5:
        r4 = db_get("plonk.prover.round4")
        if not r4:
            return None
        state.zeta = deserialize_fr(r4["zeta"])
        state.proof.a_eval = deserialize_fr(r4["a_eval"])
        state.proof.b_eval = deserialize_fr(r4["b_eval"])
        state.proof.c_eval = deserialize_fr(r4["c_eval"])
        state.proof.s_sigma1_eval = deserialize_fr(r4["s_sigma1_eval"])
        state.proof.s_sigma2_eval = deserialize_fr(r4["s_sigma2_eval"])
        state.proof.z_omega_eval = deserialize_fr(r4["z_omega_eval"])
        state.transcript = deserialize_transcript(r4["transcript"])

    return state, srs, pp


def _clear_rounds_from(start_round):
    """start_round 이후의 라운드 데이터를 클리어한다."""
    for r in range(start_round, 6):
        db_remove(f"plonk.prover.round{r}")
    db_remove("plonk.prover.round_info")
    db_remove_prefix("plonk.verify.")


@plonk_bp.route("/proving")
def proving_page():
    """Proving 페이지 렌더."""
    r1_info = db_get("plonk.prover.round1.info")
    r2_info = db_get("plonk.prover.round2.info")
    r3_info = db_get("plonk.prover.round3.info")
    r4_info = db_get("plonk.prover.round4.info")
    r5_info = db_get("plonk.prover.round5.info")
    proof_summary = db_get("plonk.prover.proof_summary")

    has_prereqs = (db_get("plonk.circuit.data") is not None
                   and db_get("plonk.srs.raw") is not None
                   and db_get("plonk.preprocess.raw") is not None)

    return render_template("plonk/proving.html",
                           r1_info=r1_info,
                           r2_info=r2_info,
                           r3_info=r3_info,
                           r4_info=r4_info,
                           r5_info=r5_info,
                           proof_summary=proof_summary,
                           has_prereqs=has_prereqs)


@plonk_bp.route("/proving/round1", methods=["POST"])
def proving_round1():
    """Round 1: 위트니스 다항식 커밋."""
    _clear_rounds_from(1)
    result = _rebuild_prover_state_up_to(1)
    if result is None:
        return redirect(url_for("plonk.proving_page"))
    state, srs, pp = result

    round1.execute(state)

    # 저장
    r1_data = {
        "a_poly": serialize_poly(state.a_poly),
        "b_poly": serialize_poly(state.b_poly),
        "c_poly": serialize_poly(state.c_poly),
        "pi_poly": serialize_poly(state.pi_poly),
        "a_comm": serialize_g1(state.proof.a_comm),
        "b_comm": serialize_g1(state.proof.b_comm),
        "c_comm": serialize_g1(state.proof.c_comm),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round1", r1_data)

    # 표시용
    r1_info = {
        "a_comm": g1_short(state.proof.a_comm),
        "b_comm": g1_short(state.proof.b_comm),
        "c_comm": g1_short(state.proof.c_comm),
    }
    db_set("plonk.prover.round1.info", r1_info)

    return redirect(url_for("plonk.proving_page"))


@plonk_bp.route("/proving/round2", methods=["POST"])
def proving_round2():
    """Round 2: 순열 누적 z(x)."""
    _clear_rounds_from(2)
    result = _rebuild_prover_state_up_to(2)
    if result is None:
        return redirect(url_for("plonk.proving_page"))
    state, srs, pp = result

    round2.execute(state)

    r2_data = {
        "beta": serialize_fr(state.beta),
        "gamma": serialize_fr(state.gamma),
        "z_poly": serialize_poly(state.z_poly),
        "z_comm": serialize_g1(state.proof.z_comm),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round2", r2_data)

    # z 평가값 (표시용)
    z_evals_display = []
    for i, d in enumerate(state.domain):
        val = state.z_poly.evaluate(d)
        z_evals_display.append(fr_short(val))

    r2_info = {
        "beta": fr_short(state.beta),
        "gamma": fr_short(state.gamma),
        "z_comm": g1_short(state.proof.z_comm),
        "z_evals": z_evals_display,
    }
    db_set("plonk.prover.round2.info", r2_info)

    return redirect(url_for("plonk.proving_page"))


@plonk_bp.route("/proving/round3", methods=["POST"])
def proving_round3():
    """Round 3: 몫 다항식 t(x)."""
    _clear_rounds_from(3)
    result = _rebuild_prover_state_up_to(3)
    if result is None:
        return redirect(url_for("plonk.proving_page"))
    state, srs, pp = result

    round3.execute(state)

    r3_data = {
        "alpha": serialize_fr(state.alpha),
        "t_lo_poly": serialize_poly(state.t_lo_poly),
        "t_mid_poly": serialize_poly(state.t_mid_poly),
        "t_hi_poly": serialize_poly(state.t_hi_poly),
        "t_lo_comm": serialize_g1(state.proof.t_lo_comm),
        "t_mid_comm": serialize_g1(state.proof.t_mid_comm),
        "t_hi_comm": serialize_g1(state.proof.t_hi_comm),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round3", r3_data)

    r3_info = {
        "alpha": fr_short(state.alpha),
        "t_lo_comm": g1_short(state.proof.t_lo_comm),
        "t_mid_comm": g1_short(state.proof.t_mid_comm),
        "t_hi_comm": g1_short(state.proof.t_hi_comm),
    }
    db_set("plonk.prover.round3.info", r3_info)

    return redirect(url_for("plonk.proving_page"))


@plonk_bp.route("/proving/round4", methods=["POST"])
def proving_round4():
    """Round 4: ζ에서 다항식 평가."""
    _clear_rounds_from(4)
    result = _rebuild_prover_state_up_to(4)
    if result is None:
        return redirect(url_for("plonk.proving_page"))
    state, srs, pp = result

    round4.execute(state)

    r4_data = {
        "zeta": serialize_fr(state.zeta),
        "a_eval": serialize_fr(state.proof.a_eval),
        "b_eval": serialize_fr(state.proof.b_eval),
        "c_eval": serialize_fr(state.proof.c_eval),
        "s_sigma1_eval": serialize_fr(state.proof.s_sigma1_eval),
        "s_sigma2_eval": serialize_fr(state.proof.s_sigma2_eval),
        "z_omega_eval": serialize_fr(state.proof.z_omega_eval),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round4", r4_data)

    r4_info = {
        "zeta": fr_short(state.zeta),
        "a_eval": fr_short(state.proof.a_eval),
        "b_eval": fr_short(state.proof.b_eval),
        "c_eval": fr_short(state.proof.c_eval),
        "s_sigma1_eval": fr_short(state.proof.s_sigma1_eval),
        "s_sigma2_eval": fr_short(state.proof.s_sigma2_eval),
        "z_omega_eval": fr_short(state.proof.z_omega_eval),
    }
    db_set("plonk.prover.round4.info", r4_info)

    return redirect(url_for("plonk.proving_page"))


@plonk_bp.route("/proving/round5", methods=["POST"])
def proving_round5():
    """Round 5: 선형화 + 열기 증명."""
    _clear_rounds_from(5)
    result = _rebuild_prover_state_up_to(5)
    if result is None:
        return redirect(url_for("plonk.proving_page"))
    state, srs, pp = result

    round5.execute(state)

    r5_data = {
        "v": serialize_fr(state.v),
        "r_eval": serialize_fr(state.proof.r_eval),
        "W_zeta_comm": serialize_g1(state.proof.W_zeta_comm),
        "W_zeta_omega_comm": serialize_g1(state.proof.W_zeta_omega_comm),
    }
    db_set("plonk.prover.round5", r5_data)

    r5_info = {
        "v": fr_short(state.v),
        "r_eval": fr_short(state.proof.r_eval),
        "W_zeta_comm": g1_short(state.proof.W_zeta_comm),
        "W_zeta_omega_comm": g1_short(state.proof.W_zeta_omega_comm),
    }
    db_set("plonk.prover.round5.info", r5_info)

    # 최종 Proof 요약 저장
    proof = state.build_proof()
    db_set("plonk.prover.proof_raw", serialize_proof(proof))

    proof_summary = {
        "a_comm": g1_short(proof.a_comm),
        "b_comm": g1_short(proof.b_comm),
        "c_comm": g1_short(proof.c_comm),
        "z_comm": g1_short(proof.z_comm),
        "t_lo_comm": g1_short(proof.t_lo_comm),
        "t_mid_comm": g1_short(proof.t_mid_comm),
        "t_hi_comm": g1_short(proof.t_hi_comm),
        "a_eval": fr_short(proof.a_eval),
        "b_eval": fr_short(proof.b_eval),
        "c_eval": fr_short(proof.c_eval),
        "s_sigma1_eval": fr_short(proof.s_sigma1_eval),
        "s_sigma2_eval": fr_short(proof.s_sigma2_eval),
        "z_omega_eval": fr_short(proof.z_omega_eval),
        "r_eval": fr_short(proof.r_eval),
        "W_zeta_comm": g1_short(proof.W_zeta_comm),
        "W_zeta_omega_comm": g1_short(proof.W_zeta_omega_comm),
    }
    db_set("plonk.prover.proof_summary", proof_summary)

    return redirect(url_for("plonk.proving_page"))


@plonk_bp.route("/proving/run-all", methods=["POST"])
def proving_run_all():
    """5라운드 일괄 실행."""
    _clear_rounds_from(1)
    result = _rebuild_prover_state_up_to(1)
    if result is None:
        return redirect(url_for("plonk.proving_page"))
    state, srs, pp = result

    # Round 1
    round1.execute(state)
    r1_data = {
        "a_poly": serialize_poly(state.a_poly),
        "b_poly": serialize_poly(state.b_poly),
        "c_poly": serialize_poly(state.c_poly),
        "pi_poly": serialize_poly(state.pi_poly),
        "a_comm": serialize_g1(state.proof.a_comm),
        "b_comm": serialize_g1(state.proof.b_comm),
        "c_comm": serialize_g1(state.proof.c_comm),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round1", r1_data)
    db_set("plonk.prover.round1.info", {
        "a_comm": g1_short(state.proof.a_comm),
        "b_comm": g1_short(state.proof.b_comm),
        "c_comm": g1_short(state.proof.c_comm),
    })

    # Round 2
    round2.execute(state)
    r2_data = {
        "beta": serialize_fr(state.beta),
        "gamma": serialize_fr(state.gamma),
        "z_poly": serialize_poly(state.z_poly),
        "z_comm": serialize_g1(state.proof.z_comm),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round2", r2_data)
    z_evals_display = []
    for i, d in enumerate(state.domain):
        val = state.z_poly.evaluate(d)
        z_evals_display.append(fr_short(val))
    db_set("plonk.prover.round2.info", {
        "beta": fr_short(state.beta),
        "gamma": fr_short(state.gamma),
        "z_comm": g1_short(state.proof.z_comm),
        "z_evals": z_evals_display,
    })

    # Round 3
    round3.execute(state)
    r3_data = {
        "alpha": serialize_fr(state.alpha),
        "t_lo_poly": serialize_poly(state.t_lo_poly),
        "t_mid_poly": serialize_poly(state.t_mid_poly),
        "t_hi_poly": serialize_poly(state.t_hi_poly),
        "t_lo_comm": serialize_g1(state.proof.t_lo_comm),
        "t_mid_comm": serialize_g1(state.proof.t_mid_comm),
        "t_hi_comm": serialize_g1(state.proof.t_hi_comm),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round3", r3_data)
    db_set("plonk.prover.round3.info", {
        "alpha": fr_short(state.alpha),
        "t_lo_comm": g1_short(state.proof.t_lo_comm),
        "t_mid_comm": g1_short(state.proof.t_mid_comm),
        "t_hi_comm": g1_short(state.proof.t_hi_comm),
    })

    # Round 4
    round4.execute(state)
    r4_data = {
        "zeta": serialize_fr(state.zeta),
        "a_eval": serialize_fr(state.proof.a_eval),
        "b_eval": serialize_fr(state.proof.b_eval),
        "c_eval": serialize_fr(state.proof.c_eval),
        "s_sigma1_eval": serialize_fr(state.proof.s_sigma1_eval),
        "s_sigma2_eval": serialize_fr(state.proof.s_sigma2_eval),
        "z_omega_eval": serialize_fr(state.proof.z_omega_eval),
        "transcript": serialize_transcript(state.transcript),
    }
    db_set("plonk.prover.round4", r4_data)
    db_set("plonk.prover.round4.info", {
        "zeta": fr_short(state.zeta),
        "a_eval": fr_short(state.proof.a_eval),
        "b_eval": fr_short(state.proof.b_eval),
        "c_eval": fr_short(state.proof.c_eval),
        "s_sigma1_eval": fr_short(state.proof.s_sigma1_eval),
        "s_sigma2_eval": fr_short(state.proof.s_sigma2_eval),
        "z_omega_eval": fr_short(state.proof.z_omega_eval),
    })

    # Round 5
    round5.execute(state)
    r5_data = {
        "v": serialize_fr(state.v),
        "r_eval": serialize_fr(state.proof.r_eval),
        "W_zeta_comm": serialize_g1(state.proof.W_zeta_comm),
        "W_zeta_omega_comm": serialize_g1(state.proof.W_zeta_omega_comm),
    }
    db_set("plonk.prover.round5", r5_data)
    db_set("plonk.prover.round5.info", {
        "v": fr_short(state.v),
        "r_eval": fr_short(state.proof.r_eval),
        "W_zeta_comm": g1_short(state.proof.W_zeta_comm),
        "W_zeta_omega_comm": g1_short(state.proof.W_zeta_omega_comm),
    })

    # Proof 요약
    proof = state.build_proof()
    db_set("plonk.prover.proof_raw", serialize_proof(proof))
    proof_summary = {
        "a_comm": g1_short(proof.a_comm),
        "b_comm": g1_short(proof.b_comm),
        "c_comm": g1_short(proof.c_comm),
        "z_comm": g1_short(proof.z_comm),
        "t_lo_comm": g1_short(proof.t_lo_comm),
        "t_mid_comm": g1_short(proof.t_mid_comm),
        "t_hi_comm": g1_short(proof.t_hi_comm),
        "a_eval": fr_short(proof.a_eval),
        "b_eval": fr_short(proof.b_eval),
        "c_eval": fr_short(proof.c_eval),
        "s_sigma1_eval": fr_short(proof.s_sigma1_eval),
        "s_sigma2_eval": fr_short(proof.s_sigma2_eval),
        "z_omega_eval": fr_short(proof.z_omega_eval),
        "r_eval": fr_short(proof.r_eval),
        "W_zeta_comm": g1_short(proof.W_zeta_comm),
        "W_zeta_omega_comm": g1_short(proof.W_zeta_omega_comm),
    }
    db_set("plonk.prover.proof_summary", proof_summary)

    return redirect(url_for("plonk.proving_page"))


@plonk_bp.route("/proving/clear", methods=["POST"])
def proving_clear():
    """프로버 데이터를 클리어한다."""
    db_remove_prefix("plonk.prover.")
    db_remove_prefix("plonk.verify.")
    return redirect(url_for("plonk.proving_page"))


# ──────────────────────────────────────────────────────────────
# Verifying 페이지
# ──────────────────────────────────────────────────────────────

@plonk_bp.route("/verifying")
def verifying_page():
    """검증 페이지 렌더."""
    proof_summary = db_get("plonk.prover.proof_summary")
    verify_result = db_get("plonk.verify.result")

    return render_template("plonk/verifying.html",
                           proof_summary=proof_summary,
                           verify_result=verify_result)


@plonk_bp.route("/verifying/verify", methods=["POST"])
def verifying_verify():
    """검증을 실행한다."""
    proof_raw = db_get("plonk.prover.proof_raw")
    srs_raw = db_get("plonk.srs.raw")
    pp_raw = db_get("plonk.preprocess.raw")
    circuit_data = db_get("plonk.circuit.data")

    if not all([proof_raw, srs_raw, pp_raw, circuit_data]):
        return redirect(url_for("plonk.verifying_page"))

    proof = deserialize_proof(proof_raw)
    srs = deserialize_srs(srs_raw)
    pp = deserialize_preprocessed(pp_raw)
    public_inputs = deserialize_fr_list(circuit_data["public_inputs"])

    # Fiat-Shamir 챌린지 재생 (표시용)
    transcript = Transcript()
    transcript.append_point(b"a_comm", proof.a_comm)
    transcript.append_point(b"b_comm", proof.b_comm)
    transcript.append_point(b"c_comm", proof.c_comm)
    beta = transcript.challenge_scalar(b"beta")
    gamma = transcript.challenge_scalar(b"gamma")
    transcript.append_point(b"z_comm", proof.z_comm)
    alpha = transcript.challenge_scalar(b"alpha")
    transcript.append_point(b"t_lo_comm", proof.t_lo_comm)
    transcript.append_point(b"t_mid_comm", proof.t_mid_comm)
    transcript.append_point(b"t_hi_comm", proof.t_hi_comm)
    zeta = transcript.challenge_scalar(b"zeta")
    transcript.append_scalar(b"a_eval", proof.a_eval)
    transcript.append_scalar(b"b_eval", proof.b_eval)
    transcript.append_scalar(b"c_eval", proof.c_eval)
    transcript.append_scalar(b"s_sigma1_eval", proof.s_sigma1_eval)
    transcript.append_scalar(b"s_sigma2_eval", proof.s_sigma2_eval)
    transcript.append_scalar(b"z_omega_eval", proof.z_omega_eval)
    v = transcript.challenge_scalar(b"v")
    u = transcript.challenge_scalar(b"u")

    # 공개 값 계산
    from zkp.plonk.utils import vanishing_poly_eval, lagrange_basis_eval
    n = pp.n
    omega = pp.omega
    zh_zeta = vanishing_poly_eval(n, zeta)
    l1_zeta = lagrange_basis_eval(0, n, omega, zeta)
    pi_zeta = FR(0)

    # 검증 실행
    result = verify(proof, public_inputs, pp, srs)

    verify_result = {
        "result": result,
        "challenges": {
            "beta": fr_short(beta),
            "gamma": fr_short(gamma),
            "alpha": fr_short(alpha),
            "zeta": fr_short(zeta),
            "v": fr_short(v),
            "u": fr_short(u),
        },
        "public_values": {
            "zh_zeta": fr_short(zh_zeta),
            "l1_zeta": fr_short(l1_zeta),
            "pi_zeta": fr_short(pi_zeta),
        },
    }
    db_set("plonk.verify.result", verify_result)

    return redirect(url_for("plonk.verifying_page"))


@plonk_bp.route("/verifying/clear", methods=["POST"])
def verifying_clear():
    """검증 결과를 클리어한다."""
    db_remove_prefix("plonk.verify.")
    return redirect(url_for("plonk.verifying_page"))
