# sections/setup_address/features/_1t.py
from __future__ import annotations
import math
import re
from typing import List
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#                            기본 유틸 (셀 접근/계산)
# ─────────────────────────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, _col_idx(letters)])

_CELL_RE = re.compile(r'([A-Za-z]+[0-9]+)')
def eval_expr(arr: np.ndarray, expr: str) -> float:
    def repl(m: re.Match) -> str:
        return str(g(arr, m.group(1)))
    safe = _CELL_RE.sub(repl, expr.replace(" ", ""))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

def _mid(a: float, b: float) -> float:
    return 0.5 * (a + b)

def _angle2d_at(Bx, By, Ax, Ay, Cx, Cy) -> float:
    """2D에서 꼭짓점 B의 ∠ABC (deg, 양수)"""
    v1 = np.array([Ax - Bx, Ay - By], dtype=float)
    v2 = np.array([Cx - Bx, Cy - By], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return abs(math.degrees(math.acos(cos_t)))


# ─────────────────────────────────────────────────────────────────────────────
#                        10~12 (Arm Raising / Club Lie)
# ─────────────────────────────────────────────────────────────────────────────
def _build_items_10_to_12(arr: np.ndarray, row: int = 1) -> List[List]:
    BA1, BB1 = g(arr, f"BA{row}"), g(arr, f"BB{row}")  # A
    BM1, BN1 = g(arr, f"BM{row}"), g(arr, f"BN{row}")  # F
    K1,  L1  = g(arr, f"K{row}"),  g(arr, f"L{row}")   # B
    CN1, CO1 = g(arr, f"CN{row}"), g(arr, f"CO{row}")  # G

    ang_afb = _angle2d_at(BM1, BN1, BA1, BB1, K1,  L1)       # 10
    ang_afg = _angle2d_at(BM1, BN1, BA1, BB1, CN1, CO1)      # 11
    FH = abs(BN1); HG = abs(BM1 - CN1)                        # 12 (직각)
    ang_fgh = abs(math.degrees(math.atan2(FH, HG)))

    return [
        ["X", "10. Arm Raising 1", ang_afb],
        ["X", "11. Arm Raising 2", ang_afg],
        ["X", "12. Club Lie", ang_fgh],
    ]

# ─────────────────────────────────────────────────────────────────────────────
#                            2.1.1.1 Grip face angle
# ─────────────────────────────────────────────────────────────────────────────
def build_grip_face_angle_table(arr: np.ndarray) -> pd.DataFrame:
    """단일표(L/R = BN1 - AY1)"""
    val = eval_expr(arr, "BN1 - AY1")
    return pd.DataFrame([["L/R", val]],
                        columns=["검사명", "1 formula", "값"])

def build_grip_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """프로/일반 비교표 (식 숨김) — BN1 - AY1"""
    p = eval_expr(pro_arr, "BN1 - AY1")
    a = eval_expr(ama_arr, "BN1 - AY1")
    df = pd.DataFrame(
        [["L/R (BN1−AY1)", p, a, p - a]],
        columns=["항목", "프로", "일반", "차이(프로-일반)"]
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
#                           2.1.1.2 Posture all (1행)
# ─────────────────────────────────────────────────────────────────────────────
def build_posture_all_table(arr: np.ndarray, row: int = 1) -> pd.DataFrame:
    rows: List[List] = [
        ["X", "1. ANK/KNE", "CK1 - CB1", eval_expr(arr, "CK1 - CB1")],
        ["X", "2. ANK/WAI", "CK1 - K1",  eval_expr(arr, "CK1 - K1")],
        ["X", "3. ANK/SHO", "CK1 - BA1", eval_expr(arr, "CK1 - BA1")],
        ["X", "4. WAI/SHO", "BA1 - K1",  eval_expr(arr, "BA1 - K1")],
        ["X", "5. SHO/WRI", "BA1 - BM1", eval_expr(arr, "BA1 - BM1")],
        ["X", "6. WAI/WRI", "BM1 - K1",  eval_expr(arr, "BM1 - K1")],

        # ── [교체된 7~9] ─────────────────────────────────────────────
        # 7) 직각삼각형: AC=K1-BA1, CB=BB1-L1 → ∠ACB (양수)
        ["X", "7. Frontal Bend",
         abs(math.degrees(math.atan2(abs(eval_expr(arr, "BB1 - L1")),
                                     abs(eval_expr(arr, "K1 - BA1")))))],

        # 8) 삼각형: A(BA1,BB1), B(K1,L1), D(CB1,CC1) → ∠ABD (꼭짓점 B)
        ["X", "8. Body Hinege",
         _angle2d_at(
             g(arr, "K1"), g(arr, "L1"),      # B
             g(arr, "BA1"), g(arr, "BB1"),    # A
             g(arr, "CB1"), g(arr, "CC1"),    # D
         )],

        # 9) 삼각형: B(K1,L1), D(CB1,CC1), E(CK1,CL1) → ∠BDE (꼭짓점 D)
        ["X", "9. Leg Hinge",
         _angle2d_at(
             g(arr, "CB1"), g(arr, "CC1"),    # D (vertex)
             g(arr, "K1"),  g(arr, "L1"),     # B
             g(arr, "CK1"), g(arr, "CL1"),    # E
         )],
    ]

    # 이하 기존 10~22, 23(CP1−AZ1) 그대로 유지
    rows.extend(_build_items_10_to_12(arr, row=row))
    rows += [
        ["Y", "13. Head",         eval_expr(arr, "AD1")],
        ["Y", "14. R/L SHO DIF", eval_expr(arr, "AM1 - BB1")],
        ["Y", "15. R/L WAI DIF", eval_expr(arr, "K1 - I1")],
        ["Y", "16. L KNE/L WRI", eval_expr(arr, "AY1 - BQ1")],
        ["Z", "17. FOOT/WAI",
            eval_expr(arr, "(J1+M1)/2 - (CM1+CA1)/2")],
        ["Z", "18. WAI/SHO",
            eval_expr(arr, "(AN1+BC1)/2 - (J1+M1)/2")],
        ["Z", "19. SHO/HED",
            eval_expr(arr, "AE1 - (AN1+BC1)/2")],
        ["Z", "20. SHO/WRI",
            eval_expr(arr, "(AZ1+BO1)/2 - (AN1+BC1)/2")],
        ["Z", "21. WAI/WRI",
            eval_expr(arr, "(AZ1+BO1)/2 - (J1+M1)/2")],
        ["Z", "22. L SHO/L WRI",
            eval_expr(arr, "AZ1 - AN1")],
        ["Z", "23. L WRI/CLU",
            eval_expr(arr, "CP1 - AZ1")],
    ]
    df = pd.DataFrame(rows, columns=["축", "검사명", "1 formula", "값"])
    return df

def build_posture_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    Posture all(1행, 1~23 항목) 기준 프로/일반 비교표.
    열: 축 / 검사명 / 프로 / 일반 / 차이(프로-일반)
    ※ '1 formula'는 숨김.
    """
    # 각 배열로 단일표 생성
    p = build_posture_all_table(pro_arr, row=1)
    a = build_posture_all_table(ama_arr, row=1)

    # 키(축, 검사명, 수식)로 병합 후 컬럼 정리
    m = p.merge(a, on=["축", "검사명", "1 formula"], suffixes=("·프로", "·일반"))
    m = m.rename(columns={"값·프로": "프로", "값·일반": "일반"})
    m["차이(프로-일반)"] = (m["프로"] - m["일반"]).astype(float)

    # 수식 컬럼 숨기기
    m = m.drop(columns=["1 formula"])
    return m

# =========================
# 2.1.1.3 Alignment (L/R)
# =========================
def _alignment_values(arr: np.ndarray) -> dict:
    return {
        "1) Toe"     : eval_expr(arr, "BS1 - CE1"),
        "2) Knee"     : eval_expr(arr, "BP1 - CB1"),
        "3) Waist"       : eval_expr(arr, "H1 - K1"),
        "4) Shoulder" : eval_expr(arr, "AL1 - BA1"),
        "5) Elbow"     : eval_expr(arr, "AR1 - BG1"),
    }

def build_alignment_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = _alignment_values(pro_arr)
    a = _alignment_values(ama_arr)
    rows = []
    for k in p.keys():
        rows.append([k, p[k], a[k], p[k] - a[k]])
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])


# ===============================
# 2.1.1.4 Stance & Ball Position
# ===============================
def _stance_ball_values(arr: np.ndarray) -> dict:
    width = abs(eval_expr(arr, "CA1 - CM1"))
    width_club = abs(eval_expr(arr, "CA1 - CP1")) / (width if width != 0 else float("nan"))
    return {
        "1) Width"               : width,
        "2) Width/Club Pos" : width_club,
        "3) L. Toe / CHD X"     : eval_expr(arr, "CN1 - BS1"),
        "4) L ANK X"                : eval_expr(arr, "CA1"),
    }

def build_stance_ball_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = _stance_ball_values(pro_arr)
    a = _stance_ball_values(ama_arr)
    rows = []
    for k in p.keys():
        rows.append([k, p[k], a[k], p[k] - a[k]])
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])


# ==========================================
# 2.1.1.5 Basic Body Data (Length, cm) — 요약
# ==========================================
def _dist3(ax, ay, az, bx, by, bz) -> float:
    return float(math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2))

def _hyp(ab: float, bc: float) -> float:
    # 직각삼각형에서 AC(빗변)
    return float(math.sqrt(max(ab, 0.0)**2 + max(bc, 0.0)**2))

def _leg_from_hyp(ac: float, ab: float) -> float:
    # 직각삼각형에서 BC (AC^2 - AB^2)
    val = ac**2 - ab**2
    return float(math.sqrt(val)) if val > 0 else 0.0

def _basic_body_values(arr: np.ndarray) -> dict:
    # 1) Lower Body(leg) : knee(cb,cc,cd) ↔ ankle(ck,cl,cm)
    v1 = _dist3(g(arr,"CB1"),g(arr,"CC1"),g(arr,"CD1"),
                g(arr,"CK1"),g(arr,"CL1"),g(arr,"CM1"))
    # 2) Thigh : knee(cb,cc,cd) ↔ waist(k,l,m)
    v2 = _dist3(g(arr,"CB1"),g(arr,"CC1"),g(arr,"CD1"),
                g(arr,"K1"), g(arr,"L1"), g(arr,"M1"))
    # 3) Upper Body : right(AB=K1-BA1, BC=BB1-L1) → AC
    v3 = _hyp(abs(eval_expr(arr,"K1-BA1")), abs(eval_expr(arr,"BB1-L1")))
    # 4) Head : right(AB=BA1-AC1, BC=AD1-BB1) → AC
    v4 = _hyp(abs(eval_expr(arr,"BA1-AC1")), abs(eval_expr(arr,"AD1-BB1")))
    # 5) Height : 1+2+3+4+CL1
    v5 = v1 + v2 + v3 + v4 + abs(g(arr,"CL1"))

    # 6) Upper Arm : right(AB=BB1-BH1, BC=BA1-BG1)
    v6 = _hyp(abs(eval_expr(arr,"BB1-BH1")), abs(eval_expr(arr,"BA1-BG1")))
    # 7) Forearm : right(AB=BH1-BM1, BC=BG1-BM1)
    v7 = _hyp(abs(eval_expr(arr,"BH1-BM1")), abs(eval_expr(arr,"BG1-BM1")))

    # 8) Club : 주어진 AB, AC에서 BC
    ab = abs(eval_expr(arr,"BM1-CN1"))
    ac = abs(g(arr,"BN1"))
    v8 = _leg_from_hyp(ac, ab)

    # 9) Swing Size : 6 + 7 + 8
    v9  = v6 + v7 + v8
    # 10) Sho Width : |AN1| + |BC1|
    v10 = abs(g(arr,"AN1")) + abs(g(arr,"BC1"))
    # 11) Wai Width : |J1| + |M1|
    v11 = abs(g(arr,"J1")) + abs(g(arr,"M1"))

    return {
        "1) Lower Body (leg)"  : v1,
        "2) Body(Thigh)"             : v2,
        "3) Upper Body": v3,
        "4) Head"      : v4,
        "5) Height": v5,
        "6) Upper Arm" : v6,
        "7) Forearm"   : v7,
        "8) Club" : v8,
        "9) Swing Size"  : v9,
        "10) Sho Width": v10,
        "11) Wai Width" : v11,
    }

def build_basic_body_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = _basic_body_values(pro_arr)
    a = _basic_body_values(ama_arr)
    rows = []
    for k in p.keys():
        rows.append([k, p[k], a[k], p[k] - a[k]])
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])
