# sections/swing_phase/features/_4phase_q5.py
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
    """'CQ5-CN5', 'H5-H4' 같은 식을 안전하게 평가"""
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
    문서 표기: AB=ΔY, BC=ΔX 일 때  `90° - ∠BAC`.
    ΔY, ΔX 의 절댓값을 사용해 0~90 범위 값으로 계산.
    """
    try:
        th = math.degrees(math.atan2(abs(dx), abs(dy)))  # ∠BAC
        return float(90.0 - th)
    except Exception:
        return float("nan")

# ── 항목 정의: (분류, 검사명, 계산식 or special) ───────────────────────────
# special 타입:
#   {"type":"ang90","dy":"AY4-AY5","dx":"AX4-AX5"}  → 90 - angle(ΔY,ΔX)
_ITEMS: list[tuple[str, str, object]] = [
    # CLUB
    ("CLUB", "CHD IN/OUT",     "CQ5 - CN5"),
    ("CLUB", "4/5 CHD X",      "CN5 - CN4"),
    ("CLUB", "4/5 CHD Y",      "CO5 - CO4"),

    # WRI
    ("WRI",  "4/5 L WRI Y",    "AY5 - AY4"),
    ("WRI",  "4/5 L WRI X",    "AX5 - AX4"),
    ("WRI",  "4/5 L WRI Y Ang",
        {"type":"ang90", "dy":"AY4-AY5", "dx":"AX4-AX5"}),

    # ELB
    ("ELB",  "4/5 ELB X",      "BG5 - BG4"),
    ("ELB",  "R ELB / R WRI X","BM5 - BG5"),
    ("ELB",  "4/5 ELB Y",      "BH5 - BH4"),

    # SHO
    ("SHO",  "4/5 R SHO X",    "BA5 - BA4"),
    ("SHO",  "4/5 R SHO Y",    "BB5 - BB4"),
    ("SHO",  "R SHO Y Ang",
        {"type":"ang90", "dy":"BB4-BB5", "dx":"BA4-BA5"}),

    # SHO/WRI
    ("SHO/WRI", "L SHO / L WRI X", "AX6 - AL6"),

    # WAI (좌·우)
    ("WAI",  "4/5 L WAI X",    "H5 - H4"),
    ("WAI",  "4/5 L WAI Z",    "J5 - J4"),
    ("WAI",  "4/5 R WAI X",    "K5 - K4"),
    ("WAI",  "4/5 R WAI Y",    "L5 - L4"),
    ("WAI",  "4/5 R WAI Z",    "M5 - M4"),

    # KNE
    ("KNE",  "4/5 L KNE Z",    "BR5 - BR4"),
    ("KNE",  "4/5 L KNE X",    "BP5 - BP4"),

    # HED
    ("HED",  "4/5 R X",        "CB5 - CB4"),
    ("HED",  "4/5 X",          "AC5 - AC4"),
    ("HED",  "4/5 Y",          "AD5 - AD4"),
    ("HED",  "4/5 Z",          "AE5 - AE4"),
]

# ── 표 생성 함수 ───────────────────────────────────────────────────────────
def build_quarter5_phase_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    Frame 5(4/5 구간) 스윙 페이즈 비교표
    반환 컬럼: ['분류','검사명','프로','일반','차이(프로-일반)']
    """
    rows: list[list] = []

    for group, label, spec in _ITEMS:
        if isinstance(spec, str):
            p = _eval_expr(pro_arr, spec)
            a = _eval_expr(ama_arr, spec)
        else:
            # special: 90 - angle from ΔY, ΔX
            dy_expr = spec["dy"]
            dx_expr = spec["dx"]
            dy_p = _eval_expr(pro_arr, dy_expr)
            dx_p = _eval_expr(pro_arr, dx_expr)
            dy_a = _eval_expr(ama_arr, dy_expr)
            dx_a = _eval_expr(ama_arr, dx_expr)
            p = _ang_90_minus_from_dy_dx(dy_p, dx_p)
            a = _ang_90_minus_from_dy_dx(dy_a, dx_a)

        rows.append([group, label, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류", "검사명", "프로", "일반", "차이(프로-일반)"])
    return df
