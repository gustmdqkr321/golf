# sections/swing_phase/features/_1phase.py
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
    """'AX2-AL2', '(BA2-BA1)/(K2-K1)' 같은 간단식 평가 (안전)"""
    expr_clean = expr.replace(" ", "")
    def repl(m: re.Match) -> str:
        return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr_clean)
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        return float("nan")
    try:
        return float(eval(safe, {"__builtins__": None}, {}))
    except Exception:
        return float("nan")

# ── 항목 정의: (No, 그룹, 검사명, 계산식) ────────────────────────────────────
_ITEMS: list[tuple[str, str, str]] = [
    (  "CHD",      "IN/OUT X",                      "CQ2 - CN2"),
    (  "CHD",      "1/2 Z",                         "CP2 - CP1"),
    (  "WRI",      "1/2 L  Z",                      "AZ2 - AZ1"),
    (  "WRI",      "L/R  Y",                        "BN2 - AY2"),
    (  "WRI/CHD",  "R  X",                          "CN2 - BM2"),
    ( "WRI/CHD",  "R  Y",                          "CO2 - BN2"),
    ( "SHO/WRI",  "L  X",                          "AX2 - AL2"),
    ( "ELB",      "R SHO / R ELB  Z",              "BO2 - BC2"),   # 색깔확인 → 무시
    ( "SHO",      "1/2 R SHO  X",                  "BA2 - BA1"),
    ( "SHO",      "1/2 R SHO  Y",                  "BB2 - BB1"),
    ( "SHO",      "1/2 R SHO  Z",                  "BC2 - BC1"),   # 공식확인 → 무시
    ( "SHO/ELB",  "1/2 R SHO/ELB  SYN X",          "(BA2 - BA1) - (BG2 - BG1)"),
    ( "SHO/ELB",  "1/2 R SHO/ELB  SYN Y",          "(BH2 - BH1) - (BB2 - BB1)"),
    ( "WAI",      "1/2 R WAI  X",                  "K2 - K1"),
    ( "WAI",      "1/2 R WAI  Y",                  "L2 - L1"),
    ( "WAI",      "1/2 R WAI  Z",                  "M2 - M1"),
    ( "WAI/SHO",  "1/2 R WAI/SHO  X",              "(BA2 - BA1) - (K2 - K1)"),
    ( "WAI/SHO",  "1/2 R WAI/SHO  X  RATIO",       "(BA2 - BA1) / (K2 - K1)"),
    ( "WAI/SHO",  "1/2  WAI/SHO  Y",               "(BB2 - BB1) - (K2 - K1)"),
    ( "WAI/SHO",  "1/2  WAI/SHO  Y  RATIO",        "(BB2 - BB1) / (K2 - K1)"),
    ( "KNE",      "1/2 R KNE  X",                  "CB2 - CB1"),
    ( "KNE",      "1/2 R KNE  Z",                  "CD2 - CD1"),
]

# ── 표 생성 ────────────────────────────────────────────────────────────────
def build_swing_phase_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    rows: list[list] = []
    for group, label, expr in _ITEMS:
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append([group, label, p, a, p - a])
    df = pd.DataFrame(rows, columns=["분류", "검사명", "프로", "일반", "차이(프로-일반)"])
    return df
