# sections/swing/features/_19to23.py
from __future__ import annotations
import numpy as np, pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters: idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# 19) 4 R WRI/ELB X : BG4 - BM4
def build_19_r_wri_elb_x(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = g(pro_arr, "BG4") - g(pro_arr, "BM4")
    a = g(ama_arr, "BG4") - g(ama_arr, "BM4")
    df = pd.DataFrame([["4 R WRI/ELB X (BG4-BM4)", p, a, p-a]],
                      columns=["항목","프로","일반","차이(프로-일반)"])
    return df.round(2)

# 20) 1/4 Head Y,Z : AD7-AD1 / AE7-AE1
def build_20_head_quarter(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    pY = g(pro_arr,"AD7")-g(pro_arr,"AD1"); aY = g(ama_arr,"AD7")-g(ama_arr,"AD1")
    pZ = g(pro_arr,"AE7")-g(pro_arr,"AE1"); aZ = g(ama_arr,"AE7")-g(ama_arr,"AE1")
    rows = [
        ["1/4 Head Y (AD7-AD1)", pY, aY, pY-aY],
        ["1/4 Head Z (AE7-AE1)", pZ, aZ, pZ-aZ],
    ]
    return pd.DataFrame(rows, columns=["항목","프로","일반","차이(프로-일반)"]).round(2)

# 21) 8 CHD Y : CO8
def build_21_8_chd_y(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = g(pro_arr,"CO8"); a = g(ama_arr,"CO8")
    return pd.DataFrame([["8 CHD Y (CO8)", p, a, p-a]],
                        columns=["항목","프로","일반","차이(프로-일반)"]).round(2)

# 22) 4/5 CHD SHALLOWING : CN5 - CN4  (대/중/소 분류)
def build_22_chd_shallowing(pro_arr: np.ndarray, ama_arr: np.ndarray,
                            shallow_small: float = 3.0, shallow_large: float = 6.0) -> pd.DataFrame:
    def grade(x: float) -> str:
        ax = abs(x)
        if ax < shallow_small: return "소"
        if ax < shallow_large: return "중"
        return "대"
    p = g(pro_arr,"CN5")-g(pro_arr,"CN4")
    a = g(ama_arr,"CN5")-g(ama_arr,"CN4")
    style = f"P:{grade(p)} / A:{grade(a)}"
    df = pd.DataFrame([["4/5 CHD SHALLOWING (CN5-CN4)", p, a, p-a, style]],
                      columns=["항목","프로","일반","차이(프로-일반)","등급"])
    return df.round(2)

# 23) 4 R KNE X : CB4 - CB1
def build_23_4_r_kne_x(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = g(pro_arr,"CB4")-g(pro_arr,"CB1")
    a = g(ama_arr,"CB4")-g(ama_arr,"CB1")
    return pd.DataFrame([["4 R KNE X (CB4-CB1)", p, a, p-a]],
                        columns=["항목","프로","일반","차이(프로-일반)"]).round(2)
