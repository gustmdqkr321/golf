import numpy as np
import pandas as pd
import re

# ── 기본 유틸 그대로 사용 ───────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = "".join(filter(str.isalpha, code))
    num     = int("".join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

_CELL_RE = re.compile(r'([A-Za-z]+[0-9]+)')
def eval_expr(arr: np.ndarray, expr: str) -> float:
    def repl(m: re.Match) -> str:
        return str(g(arr, m.group(1)))
    safe = _CELL_RE.sub(repl, expr.replace(" ", ""))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

# ── 보조 수학 유틸 (이 파일 안에 정의) ──────────────────────────────────────
def _angle2d_at(Bx, By, Ax, Ay, Cx, Cy) -> float:
    """2D에서 꼭짓점 B의 ∠ABC (deg, 양수)"""
    v1 = np.array([Ax - Bx, Ay - By], dtype=float)
    v2 = np.array([Cx - Bx, Cy - By], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(abs(np.degrees(np.arccos(cos_t))))

def _dist3(ax, ay, az, bx, by, bz) -> float:
    return float(np.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2))

def _hyp(ab: float, bc: float) -> float:
    # 직각삼각형에서 AC(빗변)
    return float(np.sqrt(max(ab, 0.0)**2 + max(bc, 0.0)**2))

def _leg_from_hyp(ac: float, ab: float) -> float:
    # 직각삼각형에서 BC (AC^2 - AB^2의 양의 제곱근)
    val = ac**2 - ab**2
    return float(np.sqrt(val)) if val > 0 else 0.0

def _angle_aw_cw_at(arr: np.ndarray, n: int) -> float:
    """3D: ∠WÂC, A=AX/AY/AZ, W=AR/AS/AT, C=CN/CO/CP (deg)"""
    AR, AS, AT = g(arr, f"AR{n}"), g(arr, f"AS{n}"), g(arr, f"AT{n}")
    AX, AY, AZ = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")
    CN, CO, CP = g(arr, f"CN{n}"), g(arr, f"CO{n}"), g(arr, f"CP{n}")
    v1 = np.array([AR-AX, AS-AY, AT-AZ], dtype=float)  # A→W
    v2 = np.array([CN-AX, CO-AY, CP-AZ], dtype=float)  # A→C
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0 or np.isnan(n1) or np.isnan(n2):
        return float("nan")
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))

# ─────────────────────────────────────────────────────────────
# 2.1.1.x 요약 표 (그림의 1~12 행만 직접 수식으로 계산)
#   열: [항목, 프로, 일반]
#   - '프로' / '일반' 컬럼명은 웹/엑셀 하이라이트 로직과 매칭됨
# ─────────────────────────────────────────────────────────────
def build_setup_summary_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    # 1) Grip face angle = BN1 - AY1
    def _grip(arr): 
        return eval_expr(arr, "BN1 - AY1")

    # 2) Frontal Bend = atan2(|BB1-L1|, |K1-BA1|) [deg]
    def _frontal_bend(arr):
        ab = abs(eval_expr(arr, "BB1 - L1"))
        ac = abs(eval_expr(arr, "K1 - BA1"))
        return float(np.degrees(np.arctan2(ab, ac)))

    # 3) Body Hinge = ∠ABC with A(BA1,BB1), B(K1,L1), C(CB1,CC1)
    def _body_hinge(arr):
        return _angle2d_at(
            g(arr,"K1"),  g(arr,"L1"),      # B (vertex)
            g(arr,"BA1"), g(arr,"BB1"),     # A
            g(arr,"CB1"), g(arr,"CC1"),     # C (D in 주석)
        )

    # 4) Leg Hinge = ∠ABC with A(K1,L1), B(CB1,CC1), C(CK1,CL1)
    def _leg_hinge(arr):
        return _angle2d_at(
            g(arr,"CB1"), g(arr,"CC1"),     # B (vertex)
            g(arr,"K1"),  g(arr,"L1"),      # A
            g(arr,"CK1"), g(arr,"CL1"),     # C (E in 주석)
        )

    # 5) Arm Extension = Arm Raising 1 = ∠ABC with A(BA1,BB1), B(BM1,BN1), C(K1,L1)
    def _arm_extension(arr):
        return _angle2d_at(
            g(arr,"BM1"), g(arr,"BN1"),     # B (vertex)
            g(arr,"BA1"), g(arr,"BB1"),     # A
            g(arr,"K1"),  g(arr,"L1"),      # C
        )

    # 6) Cocking Angle = ∠WÂC at frame 1 in 3D (see _angle_aw_cw_at)
    def _cocking_angle(arr):
        return _angle_aw_cw_at(arr, 1)

    # 7) Foot Width Z = |CA1 - CM1|
    def _foot_width_z(arr):
        return abs(eval_expr(arr, "CA1 - CM1"))

    # 8) Stance / Ball Position Z = |CA1 - CP1| / |CA1 - CM1|
    def _stance_ball_pos_z(arr):
        width = abs(eval_expr(arr, "CA1 - CM1"))
        num   = abs(eval_expr(arr, "CA1 - CP1"))
        return float(num / width) if width not in (0.0, float("inf"), float("-inf")) else float("nan")

    # 9) Height = LowerBody(leg)+Thigh+UpperBody+Head+|CL1|
    def _height(arr):
        v_leg   = _dist3(g(arr,"CB1"),g(arr,"CC1"),g(arr,"CD1"),
                         g(arr,"CK1"),g(arr,"CL1"),g(arr,"CM1"))
        v_thigh = _dist3(g(arr,"CB1"),g(arr,"CC1"),g(arr,"CD1"),
                         g(arr,"K1"), g(arr,"L1"), g(arr,"M1"))
        v_upper = _hyp(abs(eval_expr(arr,"K1-BA1")),  abs(eval_expr(arr,"BB1-L1")))
        v_head  = _hyp(abs(eval_expr(arr,"BA1-AC1")), abs(eval_expr(arr,"AD1-BB1")))
        return v_leg + v_thigh + v_upper + v_head + abs(g(arr,"CL1"))

    # 10) Setup Height = AD1
    def _setup_height(arr):
        return float(g(arr, "AD1"))

    # 12) Swing Size = Height + UpperArm + Forearm + Club
    def _swing_size(arr, height_val: float):
        upper_arm = _hyp(abs(eval_expr(arr,"BB1-BH1")), abs(eval_expr(arr,"BA1-BG1")))
        forearm   = _hyp(abs(eval_expr(arr,"BH1-BM1")), abs(eval_expr(arr,"BG1-BM1")))
        ac = abs(eval_expr(arr,"BN1"))
        ab = abs(eval_expr(arr,"BM1-CN1"))
        club = _leg_from_hyp(ac, ab)
        return height_val + upper_arm + forearm + club

    # ── 프로/일반 계산 ─────────────────────────────────────────────────────
    # 1
    p_grip = _grip(pro_arr);    a_grip = _grip(ama_arr)
    # 2~6
    p_fb   = _frontal_bend(pro_arr); a_fb = _frontal_bend(ama_arr)
    p_bh   = _body_hinge(pro_arr);   a_bh = _body_hinge(ama_arr)
    p_lh   = _leg_hinge(pro_arr);    a_lh = _leg_hinge(ama_arr)
    p_ax   = _arm_extension(pro_arr);a_ax = _arm_extension(ama_arr)
    p_ck   = _cocking_angle(pro_arr);a_ck = _cocking_angle(ama_arr)
    # 7~8
    p_fw   = _foot_width_z(pro_arr); a_fw = _foot_width_z(ama_arr)
    p_spz  = _stance_ball_pos_z(pro_arr); a_spz = _stance_ball_pos_z(ama_arr)
    # 9~12 (9,10 먼저 구해 11,12에 사용)
    p_h    = _height(pro_arr);  a_h    = _height(ama_arr)
    p_sh   = _setup_height(pro_arr); a_sh = _setup_height(ama_arr)
    p_cr   = p_h - p_sh;        a_cr   = a_h - a_sh
    p_ss   = _swing_size(pro_arr, p_h); a_ss = _swing_size(ama_arr, a_h)

    rows = [
        ["1. Grip",                     p_grip,  a_grip],
        ["2. Frontal Bend",             p_fb,    a_fb],
        ["3. Body Hinge",        p_bh,    a_bh],
        ["4. Leg Hinge",                p_lh,    a_lh],
        ["5. Arm Extension",            p_ax,    a_ax],
        ["6. Cocking Ang",      p_ck,    a_ck],
        ["7. Foot Width Z",      p_fw,    a_fw],
        ["8. Stance/ Ball Position Z",  p_spz,   a_spz],
        ["9. Height",            p_h,     a_h],
        ["10. Setup Height",     p_sh,    a_sh],
        ["11. Crouch",          p_cr,    a_cr],
        ["12. Swing Size",       p_ss,    a_ss],
    ]

    out = pd.DataFrame(rows, columns=["항목", "프로", "일반"])
    for c in ("프로", "일반"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out
