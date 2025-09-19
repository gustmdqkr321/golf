
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

def build_frame4_bbam_anbc_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """프레임4: BB4-AM4, AN4-BC4 (프로/일반/차이)"""
    items = [
        ("수평", ("BB4", "AM4")),
        ("수직", ("AN4", "BC4")),
    ]
    rows = []
    for label, (c1, c2) in items:
        p = g(pro_arr, c1) - g(pro_arr, c2)
        a = g(ama_arr, c1) - g(ama_arr, c2)
        rows.append([label, round(p, 2), round(a, 2)])

    df = pd.DataFrame(rows, columns=["항목", "프로", "일반"])
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                          - pd.to_numeric(df["일반"], errors="coerce")).round(2)
    return df
