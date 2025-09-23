from __future__ import annotations
import re, math
import numpy as np
import pandas as pd

# ── 기본 유틸 ─────────────────────────────────────────────
_CELL = re.compile(r"[A-Za-z]+[0-9]+")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g(arr: np.ndarray, addr: str) -> float:
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", addr)
    if not m: return float("nan")
    c = _col_idx(m.group(1)); r = int(m.group(2)) - 1
    try: return float(arr[r, c])
    except Exception: return float("nan")

def _eval_expr(arr: np.ndarray, expr: str) -> float:
    expr_clean = expr.replace(" ", "")
    def repl(m: re.Match) -> str: return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr_clean)
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        return float("nan")
    try: return float(eval(safe, {"__builtins__": None}, {}))
    except Exception: return float("nan")

def _angle_LSHO(arr: np.ndarray) -> float:
    """특수 케이스: L SHO 각도 = ∠CAB (AB, BC 절대값 기준)"""
    try:
        ab = abs(_eval_expr(arr, "AN8 - BC8"))
        bc = abs(_eval_expr(arr, "AM8 - BB8"))
        ang = math.degrees(math.atan2(bc, ab))
        return ang
    except Exception:
        return float("nan")

# ── 항목 정의 ─────────────────────────────────────────────
_ITEMS: list[tuple[str, str, object]] = [
    ("CHD", "CHD OUT/IN ✔",   "CN8 - CQ8"),
    ("WRI", "R/L WRI Y",      "AY8 - BN8"),
    ("WRI/CHD", "L WRI/CHD X","CN8 - AX8"),
    ("WRI/CHD", "L WRI/CHD Y","CO8 - AY8"),

    ("SHO", "7/8 L SHO X",    "AL8 - AL7"),
    ("SHO", "7/8 L SHO Y",    "AM8 - AM7"),
    ("SHO", "7/8 L SHO Z",    "AN8 - AN7"),
    ("SHO", "R/L SHO Y",      "AM8 - BB8"),

    ("SHO", "L SHO 각도",     {"type":"sho_angle"}),

    ("WAI", "7/8 L WAI X",    "H8 - H7"),
    ("WAI", "7/8 L WAI Y",    "I8 - I7"),
    ("WAI", "7/8 L WAI Z",    "J8 - J7"),

    ("HED", "7/8 HED Y",      "AD8 - AD7"),
    ("HED", "7/8 HED Z",      "AE8 - AE7"),
]

# ── 메인 ─────────────────────────────────────────────
def build_quarter8_phase_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    2.1.10 Quarter 8 (7/8 Phase) 비교표
    반환: ['분류','검사명','프로','일반','차이(프로-일반)']
    """
    rows = []
    for group, label, spec in _ITEMS:
        if isinstance(spec, str):
            p = _eval_expr(pro_arr, spec); a = _eval_expr(ama_arr, spec)
        else:
            if spec.get("type") == "sho_angle":
                p = _angle_LSHO(pro_arr); a = _angle_LSHO(ama_arr)
            else:
                p = a = float("nan")
        rows.append([group, label, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류","검사명","프로","일반","차이(프로-일반)"])
    for c in ["프로","일반","차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
