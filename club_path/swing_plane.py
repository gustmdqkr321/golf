# custom_metrics.py

import math
import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_selected_diffs(xlsx_path: Path) -> list[float]:
    """
    다음 8가지 차이를 계산해 리스트로 반환:
      BC4−BC1, BM6−BM2, BO6−BO2,
      CN6−CN2, CO6−CO2, CP6−CP2,
      CN2−BM2, CN6−BM6
    """
    arr = load_sheet(xlsx_path)
    return [
        g(arr, "BC4")  - g(arr, "BC1"),
        g(arr, "BM6")  - g(arr, "BM2"),
        g(arr, "BO6")  - g(arr, "BO2"),
        g(arr, "CN6")  - g(arr, "CN2"),
        g(arr, "CO6")  - g(arr, "CO2"),
        g(arr, "CP6")  - g(arr, "CP2"),
        g(arr, "CN2")  - g(arr, "BM2"),
        g(arr, "CN6")  - g(arr, "BM6"),
    ]

def compute_bac_ax_cn_ay(xlsx_path: Path) -> list[float]:
    """
    Frame1: AB = AX1−CN1, BC = AY1 으로 하는 직각삼각형에서
    각도 ∠BAC = arctan(BC/AB) (°)
    """
    arr = load_sheet(xlsx_path)
    AB = g(arr, "AX1") - g(arr, "CN1")
    BC = g(arr, "AY1")
    angle = math.degrees(math.atan2(BC, AB))
    return [angle]

def compute_bac_with_status(xlsx_path: Path) -> list:
    """
    1) case1~case11의 ∠BAC(°) 계산 (list[0]..list[10])
    2) case6 값이 case2,case3 사이면 'GOOD' 아니면 'BAD' → list[11]
    3) case7 값이 case2,case3 사이면 'GOOD' 아니면 'BAD' → list[12]
    총 13개 요소를 리스트로 반환.
    """
    arr = load_sheet(xlsx_path)
    # define AB,BC for cases 1..11
    defs = {
        1: (("AX1","CN1"),   ("AY1", None)),
        2: (("K1","CN1"),    ("L1", None)),
        3: (("BA1","CN1"),   ("BB1", None)),
        4: (("CN3","BM2"),   ("CO3","BN2")),
        5: (("CN4","BM4"),   ("CO4","BN4")),
        6: (("CN5","BM5"),   ("CO6","BN5")),
        7: (("CN6","BM6"),   ("CO6","BN6")),
        8: (("BG6","BM6"),   ("BH6","BN6")),
        9: (("BM7","CN7"),   ("BN7", None)),
        10:(("CN8","BM8"),   ("CO8","BN8")),
        11:(("CN10","BM10"), ("CO10","BN10")),
    }
    angles = {}
    for i,(A,B) in defs.items():
        # AB
        a0,a1 = A
        AB = abs(g(arr,a0) - g(arr,a1))
        # BC
        b0,b1 = B
        BC = abs(g(arr,b0) - g(arr,b1)) if b1 else abs(g(arr,b0))
        angles[i] = math.degrees(math.atan2(BC, AB))

    # build output
    out: list = [ angles[i] for i in range(1,12) ]  # 1..11
    # statuses
    c2, c3, c6, c7 = angles[2], angles[3], angles[6], angles[7]
    lo, hi = min(c2,c3), max(c2,c3)
    status6 = "GOOD" if lo <= c6 <= hi else "BAD"
    status7 = "GOOD" if lo <= c7 <= hi else "BAD"
    out.append(status6)  # case12
    out.append(status7)  # case13

    return out

if __name__=="__main__":
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    res = compute_bac_with_status(FILE)
    print("case1…case11 angles, case12, case13 statuses:")
    for i,v in enumerate(res, start=1):
        print(f"{i:2d}: {v}")
