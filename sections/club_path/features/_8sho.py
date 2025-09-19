# sections/club_path/features/_2shoulder_tables.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, _col_idx(letters)])

def _vals(arr: np.ndarray, L_pair: tuple[str, str], R_pair: tuple[str, str]) -> tuple[list[float], list[float]]:
    """프레임 1~9, L: A−B, R: C−D 값 리스트"""
    L = [g(arr, f"{L_pair[0]}{i}") - g(arr, f"{L_pair[1]}{i}") for i in range(1, 10)]
    R = [g(arr, f"{R_pair[0]}{i}") - g(arr, f"{R_pair[1]}{i}") for i in range(1, 10)]
    return L, R

def _build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray,
                         L_pair: tuple[str, str], R_pair: tuple[str, str]) -> pd.DataFrame:
    cols = ["구분"] + [str(i) for i in range(1, 10)]
    Lp, Rp = _vals(pro_arr, L_pair, R_pair)
    La, Ra = _vals(ama_arr, L_pair, R_pair)
    rows = [
        ["L · 프로", *Lp],
        ["L · 일반", *La],
        ["R · 프로", *Rp],
        ["R · 일반", *Ra],
    ]
    return pd.DataFrame(rows, columns=cols)

# 4.2.4: L=AX−AR, R=BM−BG
def build_cmp_ax_ar__bm_bg(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    return _build_compare_table(pro_arr, ama_arr, ("AX", "AR"), ("BM", "BG"))

# 4.2.5: L=AR−AL, R=BG−BA
def build_cmp_ar_al__bg_ba(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    return _build_compare_table(pro_arr, ama_arr, ("AR", "AL"), ("BG", "BA"))

# 4.2.6: L=AX−AL, R=BM−BA
def build_cmp_ax_al__bm_ba(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    return _build_compare_table(pro_arr, ama_arr, ("AX", "AL"), ("BM", "BA"))
