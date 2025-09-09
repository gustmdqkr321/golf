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


def build_hk_alba_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    프레임7 기준 3항목 표:
      1) H7 - K7
      2) AL7 - BA7
      3) (AL7 - BA7) - (H7 - K7)
    스타일: 첫번째 값(H7-K7)이 음수면 close, 양수면 open (0이면 neutral)
           → P:프로, A:일반
    """
    def vals(arr: np.ndarray) -> tuple[float,float,float]:
        v1 = g(arr, "H7")  - g(arr, "K7")
        v2 = g(arr, "AL7") - g(arr, "BA7")
        v3 = v2 - v1
        return v1, v2, v3

    p1, p2, p3 = vals(pro_arr)
    a1, a2, a3 = vals(ama_arr)

    def style(x: float) -> str:
        if x > 0:  return "open"
        if x < 0:  return "close"
        return "neutral"

    style_str = f"P:{style(p1)}, A:{style(a1)}"

    rows = [
        ["H7 - K7",                 round(p1, 2), round(a1, 2), round(p1 - a1, 2), style_str],
        ["AL7 - BA7",               round(p2, 2), round(a2, 2), round(p2 - a2, 2), style_str],
        ["(AL7-BA7) - (H7-K7)",     round(p3, 2), round(a3, 2), round(p3 - a3, 2)],
    ]
    df = pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)", "스타일"])
    return df
