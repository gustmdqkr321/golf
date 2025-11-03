# sections/swing_phase/features/_6phase_q7.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── 기본 유틸 (A1 → 값, 간단식 평가) ─────────────────────────────────────
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
    c = _col_idx(m.group(1)); r = int(m.group(2)) - 1
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def _eval_expr(arr: np.ndarray, expr: str) -> float:
    expr_clean = expr.replace(" ", "")
    def repl(m: re.Match) -> str: return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr_clean)
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        return float("nan")
    try:
        return float(eval(safe, {"__builtins__": None}, {}))
    except Exception:
        return float("nan")

# ── 항목 정의 (이미지 표 그대로) ─────────────────────────────────────────
# (분류, 검사명, 식)
_ITEMS: list[tuple[str, str, str]] = [
    # WRI
    ("WRI", "R/L WRI Y",            "AY7 - BN7"),

    # ELB
    ("ELB", "L/R ELB X",            "BG7 - AR7"),

    # SHO
    ("SHO", "L/R SHO X",            "BA7 - AL7"),
    ("SHO", "L/R SHO Y",            "BB7 - AM7"),
    ("SHO", "6/7 L SHO Z",        "AN6 - AN7"),
    ("SHO", "6/7 L SHO X",          "AL7 - AL6"),
    ("SHO", "6/7 L SHO Y",          "AM7 - AM6"),

    # WAI
    ("WAI", "R/L WAI X",            "H7 - K7"),
    ("WAI", "R/L WAI Y",            "I7 - L7"),
    ("WAI", "6/7 L WAI Y",          "I7 - I6"),
    ("WAI", "6/7 L WAI Z",          "J7 - J6"),
    ("WAI/SHO", "WAI/SHO Z",      "(BC7 + AN7)/2 - (M7 + J7)/2"),

    # KNE
    ("KNE", "6/7 L KNE X",          "BP7 - BP6"),

    # HED
    ("HED", "6/7 HED Y",            "AD7 - AD6"),
    ("HED", "6/7 HED Z",            "AE7 - AE6"),
]

# ── 메인: 표 생성 ────────────────────────────────────────────────────────
def build_quarter7_impact_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    2.1.7 Impact (Q7) 비교표
    반환 컬럼: ['분류','검사명','프로','일반','차이(프로-일반)']
    """
    rows: list[list] = []
    for group, label, expr in _ITEMS:
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append([group, label, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류", "검사명", "프로", "일반", "차이(프로-일반)"])
    # 숫자형 보장
    for c in ["프로", "일반", "차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
