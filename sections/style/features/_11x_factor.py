
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


def build_frame4_anbc_minus_jm_delta_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    def metric(arr: np.ndarray) -> float:
        an4 = g(arr, "AN4"); bc4 = g(arr, "BC4")
        j1  = g(arr, "J1");  m1  = g(arr, "M1")
        j4  = g(arr, "J4");  m4  = g(arr, "M4")
        return (an4 - bc4) - ((j1 - m1) - (j4 - m4))

    p = metric(pro_arr)
    a = metric(ama_arr)

    df = pd.DataFrame(
        [["X Factor", round(p, 2), round(a, 2)]],
        columns=["항목", "프로", "일반"]
    )
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                            - pd.to_numeric(df["일반"], errors="coerce")).round(2)
    return df
