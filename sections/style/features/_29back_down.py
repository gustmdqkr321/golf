# sections/swing/features/_24_waist_yz.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 셀 헬퍼 ─────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ── 공개 API ───────────────────────────────────────────────────────────
def build_waist_yz_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    표 구성(그림 그대로):
      Y:
        1) 1/4 R WAI Y : L4 - L1
        2) 4/7 L WAI Y : I7 - I1
      Z:
        3) 1/4 BOT WAI Z : (J4 + M4) - (J1 - M1)
        4) 4/7 BOT WAI Z : (J7 + M7) - (J4 - M4)
    반환 컬럼: [항목, 프로, 일반, 차이(프로-일반)]
    """
    def metrics(arr: np.ndarray) -> list[float]:
        y_r_14 = g(arr, "L4") - g(arr, "L1")
        y_l_47 = g(arr, "I7") - g(arr, "I1")
        z_bot_14 = (g(arr, "J4") + g(arr, "M4")) - (g(arr, "J1") - g(arr, "M1"))
        z_bot_47 = (g(arr, "J7") + g(arr, "M7")) - (g(arr, "J4") - g(arr, "M4"))
        return [y_r_14, y_l_47, z_bot_14, z_bot_47]

    p = metrics(pro_arr)
    a = metrics(ama_arr)

    rows = [
        ["1/4 R WAI Y",   round(p[0], 2), round(a[0], 2), round(p[0]-a[0], 2)],
        ["4/7 L WAI Y",   round(p[1], 2), round(a[1], 2), round(p[1]-a[1], 2)],
        ["1/4 BOT WAI Z", round(p[2], 2), round(a[2], 2), round(p[2]-a[2], 2)],
        ["4/7 BOT WAI Z", round(p[3], 2), round(a[3], 2), round(p[3]-a[3], 2)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])
