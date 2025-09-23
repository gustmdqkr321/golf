# sections/swing_phase/features/_3phase_q4.py
from __future__ import annotations
import re
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
    """'CQ4-CN4', '(J1-M1)-(J4-M4)' 같은 식을 안전하게 평가"""
    expr_clean = expr.replace(" ", "")
    def repl(m: re.Match) -> str: return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr_clean)
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        return float("nan")
    try:
        return float(eval(safe, {"__builtins__": None}, {}))
    except Exception:
        return float("nan")

# ── 항목 정의: (분류, 검사명, 식) ─────────────────────────────────────────
_ITEMS: list[tuple[str, str, str]] = [
    # CHD
    ("CHD",     "CHD IN/OUT",            "CQ4 - CN4"),
    ("CHD",     "L/R WRI X",             "BO4 - AZ4"),

    # WRI
    ("WRI",     "L/R WRI Y",             "BN4 - AY4"),
    ("WRI",     "1/4 R WRI X",           "BM4 - BM1"),
    ("WRI",     "1/4 R WRI Y",           "BN4 - BN1"),

    # WRI/CHD
    ("WRI/CHD", "L WRI/CHD X",           "CN4 - AX4"),
    ("WRI/CHD", "L WRI/CHD Y",           "CO4 - AY4"),
    ("WRI/CHD", "L WRI/CHD Z",           "CP4 - AZ4"),

    # HED/WRI
    ("HED/WRI", "HED / L WRI X",         "AX4 - AC4"),
    ("HED/WRI", "HED / L WRI Y",         "AY4 - AD4"),
    ("HED/WRI", "HED / L WRI Z",         "AZ4 - AE4"),

    # ELB/WRI
    ("ELB/WRI", "R ELB / R WRI X",       "BM4 - BG4"),
    ("ELB/WRI", "R ELB / R WRI Z",       "BO4 - BI4"),

    # SHO
    ("SHO",     "L/R SHO Y DIFF",        "BB4 - AM4"),
    ("SHO",     "R/L SHO ROT Z",         "AN4 - BC4"),
    ("SHO",     "1/4 L SHO Z",           "AN4 - AN1"),
    ("SHO",     "1/4 R SHO Z",           "BC4 - BC1"),

    # WAI
    ("WAI",     "1/4 R WAI X",           "K4 - K1"),
    ("WAI",     "1/4 R WAI Y",           "L4 - L1"),
    ("WAI",     "1/4 R WAI Z",           "M4 - M1"),
    ("WAI",     "1/4 R/L WAI ROT Z",     "(J1 - M1) - (J4 - M4)"),

    # KNE
    ("KNE",     "1/4 R KNE X",           "CB4 - CB1"),

    # HED
    ("HED",     "1/4 HED Y",             "AD4 - AD1"),
]

# ── 표 생성 ────────────────────────────────────────────────────────────────
def build_quarter_phase_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    Frame 4(¼ 구간) 스윙 페이즈 비교표
    반환 컬럼: ['분류','검사명','프로','일반','차이(프로-일반)']
    """
    rows: list[list] = []
    for group, label, expr in _ITEMS:
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append([group, label, p, a, p - a])
    df = pd.DataFrame(rows, columns=["분류", "검사명", "프로", "일반", "차이(프로-일반)"])
    return df
