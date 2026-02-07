"""
PLONK 데이터 직렬화/역직렬화 헬퍼
====================================

TinyDB에 저장 가능한 형태로 PLONK 객체를 변환한다.
FR, G1, G2, Polynomial, Transcript, SRS, PreprocessedData, Proof, ProverState 등.
"""

from py_ecc import bn128
from py_ecc.fields import bn128_FQ as FQ

from zkp.plonk.field import FR, CURVE_ORDER, G1, G2, ec_mul
from zkp.plonk.polynomial import Polynomial
from zkp.plonk.transcript import Transcript
from zkp.plonk.srs import SRS
from zkp.plonk.preprocessor import PreprocessedData, preprocess
from zkp.plonk.prover import Proof, ProverState
from zkp.plonk.field import get_root_of_unity, get_roots_of_unity


# ─── FR ───

def serialize_fr(val):
    """FR → str(int)"""
    return str(int(val))


def deserialize_fr(s):
    """str(int) → FR"""
    return FR(int(s))


# ─── G1 point ───

def serialize_g1(point):
    """G1 point → [str, str] or None"""
    if point is None:
        return None
    return [str(int(point[0])), str(int(point[1]))]


def deserialize_g1(data):
    """[str, str] or None → G1 point"""
    if data is None:
        return None
    return (FQ(int(data[0])), FQ(int(data[1])))


# ─── G2 point ───

def serialize_g2(point):
    """G2 point → [[str,str],[str,str]] or None"""
    if point is None:
        return None
    return [
        [str(int(point[0].coeffs[0])), str(int(point[0].coeffs[1]))],
        [str(int(point[1].coeffs[0])), str(int(point[1].coeffs[1]))]
    ]


def deserialize_g2(data):
    """[[str,str],[str,str]] or None → G2 point"""
    if data is None:
        return None
    return (
        bn128.FQ2([int(data[0][0]), int(data[0][1])]),
        bn128.FQ2([int(data[1][0]), int(data[1][1])])
    )


# ─── Polynomial ───

def serialize_poly(poly):
    """Polynomial → list of str (계수)"""
    if poly is None:
        return None
    return [str(int(c)) for c in poly.coeffs]


def deserialize_poly(data):
    """list of str → Polynomial"""
    if data is None:
        return None
    return Polynomial([FR(int(s)) for s in data])


# ─── FR list ───

def serialize_fr_list(lst):
    """list[FR] → list[str]"""
    return [str(int(v)) for v in lst]


def deserialize_fr_list(data):
    """list[str] → list[FR]"""
    return [FR(int(s)) for s in data]


# ─── Transcript ───

def serialize_transcript(transcript):
    """Transcript → hex string of internal state"""
    return bytes(transcript.state).hex()


def deserialize_transcript(hex_str):
    """hex string → Transcript (state 복원)"""
    t = Transcript.__new__(Transcript)
    t.state = bytearray(bytes.fromhex(hex_str))
    return t


# ─── SRS ───

def serialize_srs(srs):
    """SRS → dict"""
    return {
        "g1_powers": [serialize_g1(p) for p in srs.g1_powers],
        "g2_powers": [serialize_g2(p) for p in srs.g2_powers],
        "max_degree": srs.max_degree
    }


def deserialize_srs(data):
    """dict → SRS"""
    g1_powers = [deserialize_g1(p) for p in data["g1_powers"]]
    g2_powers = [deserialize_g2(p) for p in data["g2_powers"]]
    return SRS(g1_powers, g2_powers, data["max_degree"])


# ─── PreprocessedData ───

def serialize_preprocessed(pp):
    """PreprocessedData → dict"""
    return {
        "n": pp.n,
        "omega": serialize_fr(pp.omega),
        "domain": serialize_fr_list(pp.domain),
        # 셀렉터 다항식
        "q_l_poly": serialize_poly(pp.q_l_poly),
        "q_r_poly": serialize_poly(pp.q_r_poly),
        "q_o_poly": serialize_poly(pp.q_o_poly),
        "q_m_poly": serialize_poly(pp.q_m_poly),
        "q_c_poly": serialize_poly(pp.q_c_poly),
        # 셀렉터 커밋먼트
        "q_l_comm": serialize_g1(pp.q_l_comm),
        "q_r_comm": serialize_g1(pp.q_r_comm),
        "q_o_comm": serialize_g1(pp.q_o_comm),
        "q_m_comm": serialize_g1(pp.q_m_comm),
        "q_c_comm": serialize_g1(pp.q_c_comm),
        # 순열 다항식
        "s_sigma1_poly": serialize_poly(pp.s_sigma1_poly),
        "s_sigma2_poly": serialize_poly(pp.s_sigma2_poly),
        "s_sigma3_poly": serialize_poly(pp.s_sigma3_poly),
        # 순열 커밋먼트
        "s_sigma1_comm": serialize_g1(pp.s_sigma1_comm),
        "s_sigma2_comm": serialize_g1(pp.s_sigma2_comm),
        "s_sigma3_comm": serialize_g1(pp.s_sigma3_comm),
        # 순열 배열
        "sigma": pp.sigma,
        "num_public_inputs": pp.num_public_inputs,
    }


def deserialize_preprocessed(data):
    """dict → PreprocessedData"""
    pp = PreprocessedData()
    pp.n = data["n"]
    pp.omega = deserialize_fr(data["omega"])
    pp.domain = deserialize_fr_list(data["domain"])
    # 셀렉터 다항식
    pp.q_l_poly = deserialize_poly(data["q_l_poly"])
    pp.q_r_poly = deserialize_poly(data["q_r_poly"])
    pp.q_o_poly = deserialize_poly(data["q_o_poly"])
    pp.q_m_poly = deserialize_poly(data["q_m_poly"])
    pp.q_c_poly = deserialize_poly(data["q_c_poly"])
    # 셀렉터 커밋먼트
    pp.q_l_comm = deserialize_g1(data["q_l_comm"])
    pp.q_r_comm = deserialize_g1(data["q_r_comm"])
    pp.q_o_comm = deserialize_g1(data["q_o_comm"])
    pp.q_m_comm = deserialize_g1(data["q_m_comm"])
    pp.q_c_comm = deserialize_g1(data["q_c_comm"])
    # 순열 다항식
    pp.s_sigma1_poly = deserialize_poly(data["s_sigma1_poly"])
    pp.s_sigma2_poly = deserialize_poly(data["s_sigma2_poly"])
    pp.s_sigma3_poly = deserialize_poly(data["s_sigma3_poly"])
    # 순열 커밋먼트
    pp.s_sigma1_comm = deserialize_g1(data["s_sigma1_comm"])
    pp.s_sigma2_comm = deserialize_g1(data["s_sigma2_comm"])
    pp.s_sigma3_comm = deserialize_g1(data["s_sigma3_comm"])
    # 순열 배열
    pp.sigma = data["sigma"]
    pp.num_public_inputs = data["num_public_inputs"]
    return pp


# ─── Proof ───

def serialize_proof(proof):
    """Proof → dict"""
    return {
        # Round 1
        "a_comm": serialize_g1(proof.a_comm),
        "b_comm": serialize_g1(proof.b_comm),
        "c_comm": serialize_g1(proof.c_comm),
        # Round 2
        "z_comm": serialize_g1(proof.z_comm),
        # Round 3
        "t_lo_comm": serialize_g1(proof.t_lo_comm),
        "t_mid_comm": serialize_g1(proof.t_mid_comm),
        "t_hi_comm": serialize_g1(proof.t_hi_comm),
        # Round 4
        "a_eval": serialize_fr(proof.a_eval) if proof.a_eval is not None else None,
        "b_eval": serialize_fr(proof.b_eval) if proof.b_eval is not None else None,
        "c_eval": serialize_fr(proof.c_eval) if proof.c_eval is not None else None,
        "s_sigma1_eval": serialize_fr(proof.s_sigma1_eval) if proof.s_sigma1_eval is not None else None,
        "s_sigma2_eval": serialize_fr(proof.s_sigma2_eval) if proof.s_sigma2_eval is not None else None,
        "z_omega_eval": serialize_fr(proof.z_omega_eval) if proof.z_omega_eval is not None else None,
        # Round 5
        "r_eval": serialize_fr(proof.r_eval) if proof.r_eval is not None else None,
        "W_zeta_comm": serialize_g1(proof.W_zeta_comm),
        "W_zeta_omega_comm": serialize_g1(proof.W_zeta_omega_comm),
    }


def deserialize_proof(data):
    """dict → Proof"""
    proof = Proof()
    # Round 1
    proof.a_comm = deserialize_g1(data.get("a_comm"))
    proof.b_comm = deserialize_g1(data.get("b_comm"))
    proof.c_comm = deserialize_g1(data.get("c_comm"))
    # Round 2
    proof.z_comm = deserialize_g1(data.get("z_comm"))
    # Round 3
    proof.t_lo_comm = deserialize_g1(data.get("t_lo_comm"))
    proof.t_mid_comm = deserialize_g1(data.get("t_mid_comm"))
    proof.t_hi_comm = deserialize_g1(data.get("t_hi_comm"))
    # Round 4
    proof.a_eval = deserialize_fr(data["a_eval"]) if data.get("a_eval") else None
    proof.b_eval = deserialize_fr(data["b_eval"]) if data.get("b_eval") else None
    proof.c_eval = deserialize_fr(data["c_eval"]) if data.get("c_eval") else None
    proof.s_sigma1_eval = deserialize_fr(data["s_sigma1_eval"]) if data.get("s_sigma1_eval") else None
    proof.s_sigma2_eval = deserialize_fr(data["s_sigma2_eval"]) if data.get("s_sigma2_eval") else None
    proof.z_omega_eval = deserialize_fr(data["z_omega_eval"]) if data.get("z_omega_eval") else None
    # Round 5
    proof.r_eval = deserialize_fr(data["r_eval"]) if data.get("r_eval") else None
    proof.W_zeta_comm = deserialize_g1(data.get("W_zeta_comm"))
    proof.W_zeta_omega_comm = deserialize_g1(data.get("W_zeta_omega_comm"))
    return proof


# ─── G1 display helpers ───

def g1_short(point):
    """G1 point → 축약 문자열 (UI 표시용)"""
    if point is None:
        return "∞"
    x_str = str(int(point[0]))
    y_str = str(int(point[1]))
    def shorten(s):
        if len(s) <= 8:
            return s
        return s[:4] + "..." + s[-4:]
    return f"({shorten(x_str)}, {shorten(y_str)})"


def g2_short(point):
    """G2 point → 축약 문자열 (UI 표시용)"""
    if point is None:
        return "∞"
    # G2 point has FQ2 coordinates
    def shorten(s):
        if len(s) <= 8:
            return s
        return s[:4] + "..." + s[-4:]
    x0 = str(int(point[0].coeffs[0]))
    x1 = str(int(point[0].coeffs[1]))
    return f"({shorten(x0)}+{shorten(x1)}i, ...)"


def fr_short(val):
    """FR → 축약 문자열 (UI 표시용)"""
    if val is None:
        return "None"
    s = str(int(val))
    if len(s) <= 10:
        return s
    return s[:4] + "..." + s[-4:]
