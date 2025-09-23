# sections/swing_phase/features/_5phase_q6.py
from __future__ import annotations
import re
import math
import numpy as np
import pandas as pd

# ── 기본 유틸 ────────────────────────────────────────────────────────────
_CELL = re.compile(r"[A-Za-z]+[0-9]+")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g(arr: np.ndarray, addr: str) -> float:
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", addr)
    if not m:
        return float("nan")
    c = _col_idx(m.group(1))
    r = int(m.group(2)) - 1
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def _eval_expr(arr: np.ndarray, expr: str) -> float:
    """'CN6-CQ6', 'AX6', 'H6-H4' 같은 식을 안전하게 평가"""
    expr_clean = expr.replace(" ", "")
    def repl(m: re.Match) -> str: return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr_clean)
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        return float("nan")
    try:
        return float(eval(safe, {"__builtins__": None}, {}))
    except Exception:
        return float("nan")

def _ang_90_minus_from_dy_dx(dy: float, dx: float) -> float:
    """
    문서 정의: AB=ΔY, BC=ΔX로 만든 ∠BAC 에 대해 (90° - ∠BAC).
    ΔX, ΔY 절댓값 사용 → 0~90 범위 값.
    """
    try:
        ang = math.degrees(math.atan2(abs(dx), abs(dy)))  # ∠BAC
        return float(90.0 - ang)
    except Exception:
        return float("nan")

# ── 항목 정의 ────────────────────────────────────────────────────────────
# (분류, 검사명, 계산식 or special)
_ITEMS: list[tuple[str, str, object]] = [
    # CHD
    ("CHD", "CHD IN/OUT",              "CN6 - CQ6"),

    # WRI
    ("WRI", "L WRI / R WRI Y",         "AY6 - BN6"),
    ("WRI", "L WRI X",                 "AX6"),
    ("WRI", "5/6 L WRI X",             "AX6 - AX5"),
    ("WRI", "5/6 L WRI ANG Y",         {"type":"ang90", "dy":"AY5-AY6", "dx":"AX5-AX6"}),

    # WRI/CHD
    ("WRI/CHD", "L WRI / CHD X",       "CN6 - AX6"),
    ("WRI/CHD", "L WRI / CHD Y",       "CO6 - AY6"),
    ("WRI/CHD", "L WRI / CHD Z",       "CP6 - AZ6"),

    # WAI/WRI
    ("WAI/WRI", "R WAI / R WRI X",     "BM6 - K6"),

    # SHO
    ("SHO", "5/6 L SHO Y",             "AM5 - AM6"),
    ("SHO", "L/R X",                   "BA6 - AL6"),
    ("SHO", "L/R Y",                   "BB6 - AM6"),
    ("SHO", "R SHO Y 각도",            {"type":"ang90", "dy":"BB5-BB6", "dx":"BA6-BA5"}),

    # SHO/WRI & SHO/ELB
    ("SHO/WRI", "L SHO / L WRI X",     "AX6 - AL6"),
    ("SHO/ELB", "L SHO / L ELB X",     "AR6 - AL6"),

    # WAI
    ("WAI", "L/R WAI X",               "K6 - H6"),
    ("WAI", "L/R WAI Y",               "L6 - I6"),
    ("WAI", "5/6 L WAI Y",             "I6 - I5"),

    # KNE
    ("KNE", "5/6 L KNE Z",             "BR6 - BR5"),
    ("KNE", "5/6 L KNE X",             "BP6 - BP5"),

    # HED
    ("HED", "5/6 HED Y",               "AD6 - AD5"),
    ("HED", "5/6 HED Z",               "AE6 - AE5"),
]

# ── 메인: 표 생성 ─────────────────────────────────────────────────────────
def build_quarter6_phase_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    Frame 6(5/6 구간) 스윙 페이즈 비교표
    반환 컬럼: ['분류','검사명','프로','일반','차이(프로-일반)']
    """
    rows: list[list] = []
    for group, label, spec in _ITEMS:
        if isinstance(spec, str):
            p = _eval_expr(pro_arr, spec)
            a = _eval_expr(ama_arr, spec)
        else:
            dy_expr = spec["dy"]; dx_expr = spec["dx"]
            p = _ang_90_minus_from_dy_dx(_eval_expr(pro_arr, dy_expr),
                                         _eval_expr(pro_arr, dx_expr))
            a = _ang_90_minus_from_dy_dx(_eval_expr(ama_arr, dy_expr),
                                         _eval_expr(ama_arr, dx_expr))
        rows.append([group, label, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류", "검사명", "프로", "일반", "차이(프로-일반)"])
    return df
