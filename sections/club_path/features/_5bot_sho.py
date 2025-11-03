# sections/club_path/features/_2midpoints.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 엑셀 셀 접근 유틸 ───────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def _g(arr: np.ndarray, code: str) -> float:
    letters = "".join(filter(str.isalpha, code))
    num     = int("".join(filter(str.isdigit, code)))
    return float(arr[num - 1, _col_idx(letters)])

def _mid(a: float, b: float) -> float:
    return 0.5 * (a + b)

def _mid_diff(arr: np.ndarray, l1: str, l2: str, r1: str, r2: str, n: int) -> float:
    """((l1n+l2n)/2) - ((r1n+r2n)/2)"""
    L = _mid(_g(arr, f"{l1}{n}"), _g(arr, f"{l2}{n}"))
    R = _mid(_g(arr, f"{r1}{n}"), _g(arr, f"{r2}{n}"))
    return L - R

def _build_compare(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    mapping: list[tuple[int, str, str, str, str]],
    title: str,
) -> pd.DataFrame:
    rows: list[list] = []
    for n, l1, l2, r1, r2 in mapping:
        p = _mid_diff(pro_arr, l1, l2, r1, r2, n)
        a = _mid_diff(ama_arr, l1, l2, r1, r2, n)
        rows.append([
            f"{n} Frame",
            
            p, a, p - a
        ])
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

# ── 표 #1 ──────────────────────────────────────────────────────────────────
# 스샷 그대로: 1·7프레임만 (AT+BI)/2 − (AN+BC)/2, 나머지는 (AR+BG)/2 − (AL+BA)/2
_MAP_1 = [
    (1, "AT", "BI", "AN", "BC"),
    (2, "AR", "BG", "AL", "BA"),
    (3, "AR", "BG", "AL", "BA"),
    (4, "AR", "BG", "AL", "BA"),
    (5, "AR", "BG", "AL", "BA"),
    (6, "AR", "BG", "AL", "BA"),
    (7, "AT", "BI", "AN", "BC"),
    (8, "AR", "BG", "AL", "BA"),
    (9, "AR", "BG", "AL", "BA"),
]

# ── 표 #2 ──────────────────────────────────────────────────────────────────
# 스샷 그대로: 1·7프레임만 (AZ+BO)/2 − (AN+BC)/2, 나머지는 (AX+BM)/2 − (AL+BA)/2
_MAP_2 = [
    (1, "AZ", "BO", "AN", "BC"),
    (2, "AX", "BM", "AL", "BA"),
    (3, "AX", "BM", "AL", "BA"),
    (4, "AX", "BM", "AL", "BA"),
    (5, "AX", "BM", "AL", "BA"),
    (6, "AX", "BM", "AL", "BA"),
    (7, "AZ", "BO", "AN", "BC"),
    (8, "AX", "BM", "AL", "BA"),
    (9, "AX", "BM", "AL", "BA"),
]

def build_midpoint_tables(pro_arr: np.ndarray, ama_arr: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    반환: (df1, df2)
      - df1: (AR+BG)/2 − (AL+BA)/2 (단, 1·7프레임은 AT/BI, AN/BC)
      - df2: (AX+BM)/2 − (AL+BA)/2 (단, 1·7프레임은 AZ/BO, AN/BC)
    """
    df1 = _build_compare(pro_arr, ama_arr, _MAP_1, title="")
    df2 = _build_compare(pro_arr, ama_arr, _MAP_2, title="")
    return df1, df2
