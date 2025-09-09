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

def build_cn_cq_style_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    맨 아래에 '스타일' 한 줄을 두고, 프로/일반 숫자 포맷이 깨지지 않도록
    별도 '스타일' 컬럼에 표시합니다. (숫자 컬럼은 NaN)
    """
    def metric(arr: np.ndarray, f: int) -> float:
        return g(arr, f"CN{f}") - g(arr, f"CQ{f}")

    def label(v: float) -> str:
        return "OPN" if v > 0 else ("CLO" if v < 0 else "NEU")

    # 값
    p8 = metric(pro_arr, 8); a8 = metric(ama_arr, 8)
    p6 = metric(pro_arr, 6); a6 = metric(ama_arr, 6)

    # 각 사람별 스타일(두 값 조합)
    p_style = f"{label(p8)}/{label(p6)}"
    a_style = f"{label(a8)}/{label(a6)}"

    rows = [
        ["CN8 - CQ8", p8, a8, p8 - a8, ""],
        ["CN6 - CQ6", p6, a6, p6 - a6, ""],
        ["스타일",     np.nan, np.nan, np.nan, f"P:{p_style} · A:{a_style}"],
    ]
    df = pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)", "스타일"])

    # 숫자 컬럼만 반올림
    for c in ["프로", "일반", "차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    return df
