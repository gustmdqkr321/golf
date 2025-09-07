# sections/swing/features/calc_first.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 셀 접근 유틸 ──────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def eval_diff(arr: np.ndarray, expr: str) -> float:
    left, right = [s.strip() for s in expr.split("-")]
    return g(arr, left) - g(arr, right)

def angle_acb_from_legs(ab: float, bc: float) -> float:
    # 직각삼각형(∠B=90°)에서 ∠ACB = atan2(|AB|, |BC|)
    return math.degrees(math.atan2(abs(ab), abs(bc)))

def classify(v: float) -> str:
    if v <= 9: return "Flat"
    if v <= 19: return "Standard"
    return "Steep"

# ── 표 계산(단일 엑셀 배열 기준) ───────────────────────────────────────────────
def compute_metrics(arr: np.ndarray, row: int) -> dict:
    AB1 = f"AY{row} - AM{row}";  BC1 = f"AX{row} - AL{row}"  # 1. L Arm Ang
    AB2 = f"BB{row} - AM{row}";  BC2 = f"BA{row} - AL{row}"  # 2. Both Sho Ang
    E4  = f"AY{row} - BB{row}"                                 # 4. R Sho / L Wri Y

    l_arm = angle_acb_from_legs(eval_diff(arr, AB1), eval_diff(arr, BC1))
    both  = angle_acb_from_legs(eval_diff(arr, AB2), eval_diff(arr, BC2))
    diff3 = l_arm - both
    e4    = eval_diff(arr, E4)

    return {"1": l_arm, "2": both, "3": diff3, "4": e4, "class": classify(diff3)}

def build_compare_df(pro: dict, ama: dict) -> pd.DataFrame:
    names = {
        "1": "L Arm Ang",
        "2": "Both Sho Ang",
        "3": "Sho/L Arm Ang (1-2)",
        "4": "R Sho / L Wri Y",
    }
    rows = []
    for no in ["1", "2", "3", "4"]:
        p, a = pro[no], ama[no]
        rows.append([no, names[no], p, a, p - a])
    df = pd.DataFrame(rows, columns=["No", "항목", "프로", "일반", "차이(프로-일반)"])
    df.attrs["pro_class"] = pro["class"]
    df.attrs["ama_class"] = ama["class"]
    return df
