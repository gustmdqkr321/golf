# sections/swing_phase/features/_2phase.py
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
    """'AX3-AL3', '(BA3-BA2)/(K3-K2)' 같은 식을 안전하게 평가"""
    expr_clean = expr.replace(" ", "")
    def repl(m: re.Match) -> str: return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr_clean)
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        return float("nan")
    try:
        return float(eval(safe, {"__builtins__": None}, {}))
    except Exception:
        return float("nan")

# ── 항목 정의: (분류, 검사명, 계산식) ───────────────────────────────────────
_ITEMS: list[tuple[str, str, str]] = [
    ("CHD",      "CLUB IN/OUT X",             "CQ3 - CN3"),
    ("WRI/CHD",  "L WRI/CHD X",               "CN3 - AX3"),
    ("WRI/CHD",  "L WRI/CHD Y",               "CO3 - AY3"),
    ("WRI/CHD",  "L WRI/CHD Z",               "CP3 - AZ3"),
    ("SHO/WRI",  "BOTH SHO / L WRI X",        "(AX3+BM3)/2 - (AL3+BA3)/2"),
    ("SHO/WRI",  "R SHO / R WRI Z",           "BO3 - BC3"),
    ("SHO",      "2/3 R SHO X",               "BA3 - BA2"),
    ("SHO",      "2/3 R SHO Y",               "BB3 - BB2"),
    ("SHO",      "2/3 R SHO Z",               "BC3 - BC2"),
    ("SHO",      "L SHO / R SHO Z",           "BC3 - AN3"),
    ("SHO",      "L SHO / R SHO Y",           "BB3 - AM3"),
    ("WAI",      "2/3 R WAI X",               "K3 - K2"),
    ("WAI",      "2/3 R WAI Y",               "L3 - L2"),
    ("WAI",      "2/3 R WAI Z",               "M3 - M2"),
    ("WAI/SHO",  "2/3 WAI/SHO X RATIO",       "(BA3 - BA2) / (K3 - K2)"),
]

# ── 표 생성 ────────────────────────────────────────────────────────────────
def build_swing_phase_table_v2(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    rows: list[list] = []
    for group, label, expr in _ITEMS:
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append([group, label, p, a, p - a])
    df = pd.DataFrame(rows, columns=["분류", "검사명", "프로", "일반", "차이(프로-일반)"])
    return df
