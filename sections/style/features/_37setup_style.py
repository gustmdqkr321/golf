# sections/swing/features/_32_combo6.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 엑셀 셀 헬퍼 ─────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ── 2D 각도 계산(점 q에서 ∠pqr) ─────────────────────────────────────────────
def _angle_2d(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """
    p, q, r: shape (2,)
    반환: ∠pqr in degrees
    """
    v1 = p - q
    v2 = r - q
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cosv = float(np.clip(v1.dot(v2) / (n1*n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))

# 직각삼각형에서 atan2(|세로|, |가로|) 방식 각도
def _angle_right_from_components(dx: float, dy: float) -> float:
    return float(np.degrees(np.arctan2(abs(dy), abs(dx))))

def _compute_items(arr: np.ndarray) -> list[float]:
    # ① BN1 − AY1
    item1 = g(arr, "BN1") - g(arr, "AY1")

    # ② (직각) ∠ABC | AC=K1−BA1, CB=BB1−L1  → atan2(|CB|, |AC|)
    dx = g(arr, "K1")  - g(arr, "BA1")   # "AC"로 지정된 가로 성분
    dy = g(arr, "BB1") - g(arr, "L1")    # "CB"로 지정된 세로 성분
    item2 = _angle_right_from_components(dx, dy)

    # ③ 2D 삼각형 ∠BDE | B=(K1,L1), D=(CB1,CC1), E=(CK1,CL1)
    B = np.array([g(arr, "K1"),  g(arr, "L1")],  float)
    D = np.array([g(arr, "CB1"), g(arr, "CC1")], float)
    E = np.array([g(arr, "CK1"), g(arr, "CL1")], float)
    item3 = _angle_2d(B, D, E)

    # ④ 2D 삼각형 ∠AFB | A=(BA1,BB1), F=(BM1,BN1), B=(K1,L1)
    A = np.array([g(arr, "BA1"), g(arr, "BB1")], float)
    F = np.array([g(arr, "BM1"), g(arr, "BN1")], float)
    B2= np.array([g(arr, "K1"),  g(arr, "L1")],  float)
    item4 = _angle_2d(A, F, B2)

    # ⑤ (AC1 + AC2 + BC3) / |CA1 − CM1|
    #   AC1 = hypot(BB1−BH1, BA1−BG1)
    AC1 = math.hypot(g(arr,"BB1") - g(arr,"BH1"), g(arr,"BA1") - g(arr,"BG1"))
    #   AC2 = hypot(BH1−BM1, BG1−BM1)
    AC2 = math.hypot(g(arr,"BH1") - g(arr,"BM1"), g(arr,"BG1") - g(arr,"BM1"))
    #   AB3 = BM1−CN1, AC3 = BN1  → BC3 = sqrt(max(0, AC3^2 − AB3^2))
    AB3 = g(arr,"BM1") - g(arr,"CN1")
    AC3 = g(arr,"BN1")
    BC3 = math.sqrt(max(0.0, AC3*AC3 - AB3*AB3))
    S   = AC1 + AC2 + BC3
    denom = abs(g(arr, "CA1") - g(arr, "CM1"))
    item5 = (S / denom) if denom != 0 else float("nan")

    # ⑥ |CA1 − CP1| / |CA1 − CM1|
    num2   = abs(g(arr, "CA1") - g(arr, "CP1"))
    item6  = (num2 / denom) if denom != 0 else float("nan")

    return [item1, item2, item3, item4, item5, item6]

def build_combo6_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    6개 항목 표 생성:
      1) BN1 − AY1
      2) (직각) ∠ABC | AC=K1−BA1, CB=BB1−L1
      3) ∠BDE | B=(K1,L1), D=(CB1,CC1), E=(CK1,CL1)
      4) ∠AFB | A=(BA1,BB1), F=(BM1,BN1), B=(K1,L1)
      5) (AC1 + AC2 + BC3) / |CA1−CM1|
           - AC1=hypot(BB1−BH1, BA1−BG1)
           - AC2=hypot(BH1−BM1, BG1−BM1)
           - BC3: AB=BM1−CN1, AC=BN1 → sqrt(max(0, AC^2−AB^2))
      6) |CA1−CP1| / |CA1−CM1|
    """
    p = _compute_items(pro_arr)
    a = _compute_items(ama_arr)

    labels = [
        "BN1 − AY1",
        "직각 ∠ABC | AC=K1−BA1, CB=BB1−L1",
        "∠BDE | B=(K1,L1), D=(CB1,CC1), E=(CK1,CL1)",
        "∠AFB | A=(BA1,BB1), F=(BM1,BN1), B=(K1,L1)",
        "(AC1 + AC2 + BC3) / |CA1−CM1|",
        "|CA1−CP1| / |CA1−CM1|",
    ]

    rows = []
    for lab, pv, av in zip(labels, p, a):
        rows.append([lab, round(pv, 2), round(av, 2), round(pv - av, 2)])

    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])
