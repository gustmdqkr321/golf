from __future__ import annotations
import re
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

# ── 항목 정의 ─────────────────────────────────────────────
_ITEMS_Q9: list[tuple[str,str]] = [
    ("L WRI/CHD X",   "CN9 - AX9"),
    ("L WRI/CHD Y",   "CO9 - AY9"),
    ("L SHO/R SHO Y", "BB9 - AM9"),
    ("L SHO/R SHO Z", "BC9 - AN9"),
    ("L WAI/R WAI Z", "M9 - J9"),
    ("8/9 L WAI Z",   "J9 - J8"),
]

_ITEMS_Q10: list[tuple[str,str]] = [
    ("R SHO TURN Z",  "BC10"),
    ("L SHO/R SHO Y", "BB10 - AM10"),
    ("HED/L WRI Z",   "AZ10 - AE10"),
    ("L WRI/CHD Y",   "CO10 - AY10"),
    ("L ANK/R WAI Z", "M10 - CA10"),
    ("R WAI/R SHO Z", "BC10 - M10"),
]

# ── 메인 ─────────────────────────────────────────────
def build_quarter9_10_phase_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    2.1.11 Quarter 9 & 10 (Finish/Follow-through) 비교표
    반환: ['Frame','검사명','프로','일반','차이(프로-일반)']
    """
    rows = []
    for label, expr in _ITEMS_Q9:
        p = _eval_expr(pro_arr, expr); a = _eval_expr(ama_arr, expr)
        rows.append([label, p, a, p - a])

    for label, expr in _ITEMS_Q10:
        p = _eval_expr(pro_arr, expr); a = _eval_expr(ama_arr, expr)
        rows.append([label, p, a, p - a])

    df = pd.DataFrame(rows, columns=["검사명","프로","일반","차이(프로-일반)"])
    for c in ["프로","일반","차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
