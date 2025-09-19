# sections/swing/features/_6wri_chd.py
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

def _wri_chd_values(arr: np.ndarray) -> dict:
    """
    1) X = CN2 - BM2  (부호 유지)
    2) Y = CO2 - AX2  (부호 유지)
    3) 세 길이의 합 = ① AC + ② AC + ③ BC  (모두 '길이'이므로 항상 ≥ 0)

       ① AB = BB1 - BH1,  BC = BA1 - BG1           → AC = √(AB² + BC²)
       ② AB = BH1 - BM1,  BC = BG1 - BM1           → AC = √(AB² + BC²)
       ③ AB = BM1 - CN1,  AC = |BN1| (빗변으로 해석) → BC = √(max(AC² - AB², 0))
    """
    # 1, 2: 요청 그대로 부호 있는 차이
    val_x = g(arr, "CN2") - g(arr, "BM2")
    val_y = g(arr, "CO2") - g(arr, "AX2")

    # ① AC (길이 → 항상 양수)
    AB1 = g(arr, "BB1") - g(arr, "BH1")
    BC1 = g(arr, "BA1") - g(arr, "BG1")
    AC1 = float(np.hypot(AB1, BC1))                 # >= 0

    # ② AC
    AB2 = g(arr, "BH1") - g(arr, "BM1")
    BC2 = g(arr, "BG1") - g(arr, "BM1")
    AC2 = float(np.hypot(AB2, BC2))                 # >= 0

    # ③ BC (AC=|BN1|)
    AB3 = g(arr, "BN1")
    AC3 = g(arr, "BM1") - g(arr, "CN1")
    BC3 = float(np.hypot(AB3, AC3))                 # >= 0

    triple_sum = AC1 + AC2 + BC3

    return {"x": val_x, "y": val_y, "sum3": triple_sum}

def build_wri_chd_table_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    요청 형식: 3행만 출력
      1) 2 L WRI/CHD X (= CN2 - BM2)
      2) 2 L WRI/CHD Y (= CO2 - AX2)
      3) 세 길이의 합(①AC+②AC+③BC)  ← '길이'라서 항상 양수
    """
    p = _wri_chd_values(pro_arr)
    a = _wri_chd_values(ama_arr)

    rows = [
        ["2 L WRI/CHD X", round(p["x"], 2),   round(a["x"], 2)],
        ["2 L WRI/CHD Y", round(p["y"], 2),   round(a["y"], 2)],
        ["l WRI/CHD Z", round(p["sum3"], 2), round(a["sum3"], 2)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반"])
