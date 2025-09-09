# sections/swing/features/_24_right_angles.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 공통 셀 헬퍼 ─────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# 직각삼각형(∠B = 90°)에서 ∠BAC = atan2(|BC|, |AB|) [deg]
def _angle_bac(ab: float, bc: float) -> float:
    ab = abs(float(ab))
    bc = abs(float(bc))
    if ab == 0 and bc == 0:
        return np.nan
    return float(np.degrees(np.arctan2(bc, ab)))

def _fmt(x: float) -> float:
    return float(np.round(x, 2))

# ── 공개 API ────────────────────────────────────────────────────────────────
def build_right_angle_bac_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    표 구성(4행):
      1) BAC(직각)  AB=AX1-CN1,        BC=AY1
      2) BAC(직각)  AB=|CN6-AX6|,       BC=CO6-AY6
      3) BAC(직각)  AB=BG6-BM6,         BC=BH6-BN6
      4) (1번 각도) - BAC(직각; AB=AX7-CN7, BC=AY7)
    각 행: [항목, 프로, 일반, 차이(프로-일반)]
    """
    # 각도 계산기
    def angle1(arr):  # AB=AX1-CN1,  BC=AY1
        return _angle_bac(g(arr, "AX1") - g(arr, "CN1"), g(arr, "AY1"))

    def angle2(arr):  # AB=|CN6-AX6|, BC=CO6-AY6
        return _angle_bac(abs(g(arr, "CN6") - g(arr, "AX6")), g(arr, "CO6") - g(arr, "AY6"))

    def angle3(arr):  # AB=BG6-BM6,  BC=BH6-BN6
        return _angle_bac(g(arr, "BG6") - g(arr, "BM6"), g(arr, "BH6") - g(arr, "BN6"))

    def angle4(arr):  # AB=AX7-CN7,  BC=AY7
        return _angle_bac(g(arr, "AX7") - g(arr, "CN7"), g(arr, "AY7"))

    # 프로/일반 값
    p1, a1 = angle1(pro_arr), angle1(ama_arr)
    p2, a2 = angle2(pro_arr), angle2(ama_arr)
    p3, a3 = angle3(pro_arr), angle3(ama_arr)
    p4, a4 = angle4(pro_arr), angle4(ama_arr)

    rows = [
        ["① BAC (AB=AX1−CN1, BC=AY1)",               _fmt(p1), _fmt(a1), _fmt(p1 - a1)],
        ["② BAC (AB=|CN6−AX6|, BC=CO6−AY6)",         _fmt(p2), _fmt(a2), _fmt(p2 - a2)],
        ["③ BAC (AB=BG6−BM6, BC=BH6−BN6)",           _fmt(p3), _fmt(a3), _fmt(p3 - a3)],
        ["④ (①) − BAC(AB=AX7−CN7, BC=AY7)",          _fmt(p1 - p4), _fmt(a1 - a4), _fmt((p1 - p4) - (a1 - a4))],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])
