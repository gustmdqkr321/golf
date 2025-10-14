# sections/club_path/features/_3bc_axcn.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── 기본 유틸: 엑셀 주소(A1) → 값 ──────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

_CELL_RE = re.compile(r"[A-Za-z]+[0-9]+")  # A1, AX6, BO2 …

def g(arr: np.ndarray, code: str) -> float:
    letters = "".join(filter(str.isalpha, code))
    num     = int("".join(filter(str.isdigit, code)))
    return float(arr[num - 1, _col_idx(letters)])

def eval_expr(arr: np.ndarray, expr: str) -> float:
    """'CP6 - CP2' 같은 산술식을 안전하게 평가"""
    def repl(m: re.Match) -> str:
        return str(g(arr, m.group(0)))
    safe = _CELL_RE.sub(repl, expr.replace(" ", ""))
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

# ── 1) 단일 항목: BC4 - BC1 ───────────────────────────────────────────────
def build_bc4_minus_bc1_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    label = "B1/4 R SHO Z"
    expr  = "BC4 - BC1"
    try: p = eval_expr(pro_arr, expr)
    except Exception: p = float("nan")
    try: a = eval_expr(ama_arr, expr)
    except Exception: a = float("nan")
    rows = [[label, expr, p, a, p - a]]
    return pd.DataFrame(rows, columns=["항목", "식", "프로", "일반", "차이(프로-일반)"])

# ── 2) AX/CN/CO/CP (6↔2 프레임) + 조합식 표 ───────────────────────────────
def build_ax_cn_group_6_2_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    스샷에 적힌 8개 항목을 그대로:
      1) AX6-AX2   2) AZ6-AZ2   3) CN6-CN2   4) CO6-CO2   5) CP6-CP2
      6) (CN6-AX6)-(CN2-AX2)
      7) (CO6-AY6)-(CO2-AY2)
      8) (CP6-AZ6)-(CP2-AZ2)
    """
    items = [
        ("2/6 L WRI x",                  "AX6 - AX2"),
        ("2/6 L WRI z",                  "AZ6 - AZ2"),
        ("2/6 CLUB x",                  "CN6 - CN2"),
        ("2/6 CLUB y",                  "CO6 - CO2"),
        ("2/6 CLUB z",                  "CP6 - CP2"),
        ("2/6 : WRI/CHD x",     "(CN6-AX6) - (CN2-AX2)"),
        ("2/6 : WRI/CHD y",     "(CO6-AY6) - (CO2-AY2)"),
        ("2/6 : WRI/CHD z",     "(CP6-AZ6) - (CP2-AZ2)"),
    ]
    rows = []
    for label, expr in items:
        try: p = eval_expr(pro_arr, expr)
        except Exception: p = float("nan")
        try: a = eval_expr(ama_arr, expr)
        except Exception: a = float("nan")
        rows.append([label, expr, p, a, p - a])
    return pd.DataFrame(rows, columns=["항목", "식", "프로", "일반", "차이(프로-일반)"])
