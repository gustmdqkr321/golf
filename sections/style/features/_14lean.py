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

def build_cp7_minus_az7_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = g(pro_arr, "CP7") - g(pro_arr, "AZ7")
    a = g(ama_arr, "CP7") - g(ama_arr, "AZ7")
    df = pd.DataFrame([["Club Lean imp", round(p, 2), round(a, 2)]],
                      columns=["항목", "프로", "일반"])
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                          - pd.to_numeric(df["일반"], errors="coerce")).round(2)
    return df