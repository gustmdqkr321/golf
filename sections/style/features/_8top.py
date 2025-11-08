from __future__ import annotations
import numpy as np
import pandas as pd


def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = "".join(filter(str.isalpha, code))
    num     = int("".join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])


def build_frame4_cnax_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """프레임4 비교표: CN4-AX4, CO4-AY4, CP4-AZ4 (프로/일반/차이)"""
    pairs  = [("AX4","AX1"),("AY4","AY1"),("CN4","AX4"), ("CO4","AY4"), ("CP4","AZ4")]
    labels = ["1/4 L WRI X",   "1/4 L WRI Y", "4 L WRI/CHD X",      "4 L WRI/CHD Y",      "4 L WRI/CHD Z"]

    rows = []
    for (c1, c2), lab in zip(pairs, labels):
        p = g(pro_arr, c1) - g(pro_arr, c2)
        a = g(ama_arr, c1) - g(ama_arr, c2)
        rows.append([lab, round(p, 2), round(a, 2)])

    df = pd.DataFrame(rows, columns=["항목", "프로", "일반"])
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                          - pd.to_numeric(df["일반"], errors="coerce")).round(2)
    return df
