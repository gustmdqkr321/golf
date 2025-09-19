# sections/swing/features/_31b7b4_norm.py
from __future__ import annotations
import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])


def _delta_b(arr) -> float:
    return g(arr, "B7") - g(arr, "B4")

def _safe_div(v: float, d: float) -> float:
    if d == 0 or not np.isfinite(d):
        return np.nan
    return v / d

def build_dl7_dq7_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """DL7 - DQ7 값만 비교하는 단일 표."""
    p = g(pro_arr, "DL7") - g(pro_arr, "DQ7")
    a = g(ama_arr, "DL7") - g(ama_arr, "DQ7")
    rows = [["DL7 − DQ7", round(p, 2), round(a, 2), round(p - a, 2)]]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

def build_b7b4_normalized_table(pro_arr, ama_arr) -> pd.DataFrame:
    """
    이미지 기준 항목들을 (B7-B4)로 나눈 정규화 표.
    행(왼쪽은 사람이 보기 쉬운 수식 라벨):
      1) [(J7+M7) - (J4+M4)] / (B7-B4)
      2) [(AN7+BC7) - (AN4+BC4)] / (B7-B4)
      3) (AE7 - AE4) / (B7-B4)
      4) (BP7 - BP4) / (B7-B4)
      5) (I7  - I4)  / (B7-B4)
      6) (AM7 - AM4) / (B7-B4)
      7) (AD7 - AD4) / (B7-B4)
    열: ['항목','프로','일반','차이(프로-일반)']
    """
    def compute_rows(arr):
        dB = _delta_b(arr)
        vals = []
        # 1
        v1 = (g(arr,"J7")+g(arr,"M7")) - (g(arr,"J4")+g(arr,"M4"))
        vals.append(_safe_div(v1, dB))
        # 2
        v2 = (g(arr,"AN7")+g(arr,"BC7")) - (g(arr,"AN4")+g(arr,"BC4"))
        vals.append(_safe_div(v2, dB))
        # 3
        v3 = g(arr,"AE7") - g(arr,"AE4")
        vals.append(_safe_div(v3, dB))
        # 4
        v4 = g(arr,"BP7") - g(arr,"BP4")
        vals.append(_safe_div(v4, dB))
        # 5
        v5 = g(arr,"I7") - g(arr,"I4")
        vals.append(_safe_div(v5, dB))
        # 6
        v6 = g(arr,"AM7") - g(arr,"AM4")
        vals.append(_safe_div(v6, dB))
        # 7
        v7 = g(arr,"AD7") - g(arr,"AD4")
        vals.append(_safe_div(v7, dB))
        return vals

    labels = [
        "Z Force WAI",
        "Z Force SHO",
        "Z Force HED",
        "Y Force L KNE X F",
        "Y Force L WAI Y F",
        "Y Force L SHO Y F",
        "Y Force HED Y F",
    ]
    p_vals = compute_rows(pro_arr)
    a_vals = compute_rows(ama_arr)

    rows = []
    for lab, p, a in zip(labels, p_vals, a_vals):
        diff = (p - a) if (np.isfinite(p) and np.isfinite(a)) else np.nan
        rows.append([lab,
                     None if not np.isfinite(p) else round(p, 2),
                     None if not np.isfinite(a) else round(a, 2),
                     None if not np.isfinite(diff) else round(diff, 2)])
    return pd.DataFrame(rows, columns=["항목","프로","일반","차이(프로-일반)"])
