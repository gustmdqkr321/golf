# sections/swing/features/_31_knee_combo.py
from __future__ import annotations
import numpy as np
import pandas as pd
import math

# ── 엑셀 셀 접근 헬퍼 ─────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# 3D 두 점 거리
def _dist3(arr: np.ndarray, p: tuple[str,str,str], q: tuple[str,str,str], n: int = 1) -> float:
    x1, y1, z1 = g(arr, f"{p[0]}{n}"), g(arr, f"{p[1]}{n}"), g(arr, f"{p[2]}{n}")
    x2, y2, z2 = g(arr, f"{q[0]}{n}"), g(arr, f"{q[1]}{n}"), g(arr, f"{q[2]}{n}")
    return float(np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))

# 직각삼각형 AC = hypot(AB, BC)
def _ac_len(ab: float, bc: float) -> float:
    return float(np.hypot(ab, bc))

def _rows_for(arr: np.ndarray) -> list[float]:
    # 1) A(knee: CB1,CC1,CD1), B(ankle: CK1,CL1,CM1) 거리
    d_knee_ankle = _dist3(arr, ("CB","CC","CD"), ("CK","CL","CM"), 1)
    # 2) A(knee), B(waist: K1,L1,M1) 거리
    d_knee_waist = _dist3(arr, ("CB","CC","CD"), ("K","L","M"), 1)
    # 3) AC (AB=K1-BA1, BC=BB1-L1)
    ac1 = _ac_len(g(arr,"K1") - g(arr,"BA1"), g(arr,"BB1") - g(arr,"L1"))
    # 4) AC (AB=BA1-AC1, BC=AD1-BB1)
    ac2 = _ac_len(g(arr,"BA1") - g(arr,"AC1"), g(arr,"AD1") - g(arr,"BB1"))
    # 5) CL1
    cl1 = g(arr, "CL1")
    # 합계
    total = d_knee_ankle + d_knee_waist + ac1 + ac2 + cl1
    return [d_knee_ankle, d_knee_waist, ac1, ac2, cl1, total]

def build_knee_combo_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    행:
      1) dist(A_knee@1, B_ankle@1)
      2) dist(A_knee@1, B_waist@1)
      3) AC | AB=K1-BA1, BC=BB1-L1
      4) AC | AB=BA1-AC1, BC=AD1-BB1
      5) CL1
      6) 합계
      7) 합계/AD1   ← 추가
    열: [항목, 프로, 일반, 차이(프로-일반)]
    """
    p_vals = _rows_for(pro_arr)  # [.., 합계]
    a_vals = _rows_for(ama_arr)

    labels = [
        "키",
    ]

    rows = []
    lab = "키"
    pv = p_vals[-1]
    av = a_vals[-1]
    rows.append([lab, round(pv, 2), round(av, 2), round(pv - av, 2)])

    # ── 추가: 합계/AD1 ─────────────────────────────────────────────
    p_total, a_total = p_vals[-1], a_vals[-1]
    p_ad1 = g(pro_arr, "AD1")
    a_ad1 = g(ama_arr, "AD1")

    def safe_div(x, y):
        return float(x) / float(y) if float(y) != 0 else np.nan

    p_ratio = safe_div(p_total, p_ad1)
    a_ratio = safe_div(a_total, a_ad1)

    rows.append([
        "height/setup",
        round(p_ratio, 3),
        round(a_ratio, 3),
        round((p_ratio - a_ratio) if (np.isfinite(p_ratio) and np.isfinite(a_ratio)) else np.nan, 3),
    ])

    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])



def _knee_combo_total(arr: np.ndarray) -> float:
    # 1) dist(knee, ankle) @1
    A_k = np.array([g(arr,"CB1"), g(arr,"CC1"), g(arr,"CD1")], float)
    B_a = np.array([g(arr,"CK1"), g(arr,"CL1"), g(arr,"CM1")], float)
    d1  = float(np.linalg.norm(A_k - B_a))

    # 2) dist(knee, waist) @1
    B_w = np.array([g(arr,"K1"), g(arr,"L1"), g(arr,"M1")], float)
    d2  = float(np.linalg.norm(A_k - B_w))

    # 3) AC | AB=K1-BA1, BC=BB1-L1
    ac3 = math.hypot(g(arr,"K1") - g(arr,"BA1"), g(arr,"BB1") - g(arr,"L1"))

    # 4) AC | AB=BA1-AC1, BC=AD1-BB1
    ac4 = math.hypot(g(arr,"BA1") - g(arr,"AC1"), g(arr,"AD1") - g(arr,"BB1"))

    # 5) CL1
    cl1 = float(g(arr,"CL1"))

    return d1 + d2 + ac3 + ac4 + cl1

def _two_ac_sum(arr: np.ndarray) -> float:
    # ACa: AB=BB1−BH1, BC=BA1−BG1
    ac_a = math.hypot(g(arr,"BB1") - g(arr,"BH1"), g(arr,"BA1") - g(arr,"BG1"))
    # ACb: AB=BH1−BM1, BC=BG1−BM1
    ac_b = math.hypot(g(arr,"BH1") - g(arr,"BM1"), g(arr,"BG1") - g(arr,"BM1"))
    return ac_a + ac_b

def build_knee_total_over_two_ac_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    행: ['합계 / (ACa+ACb)']
    열: ['항목','프로','일반','차이(프로-일반)']
    - 합계: 무릎 콤보 표(5개 항목)의 총합
    - ACa = hypot(BB1−BH1, BA1−BG1)
      ACb = hypot(BH1−BM1, BG1−BM1)
    """
    p_total = _knee_combo_total(pro_arr)
    a_total = _knee_combo_total(ama_arr)
    p_den   = _two_ac_sum(pro_arr)
    a_den   = _two_ac_sum(ama_arr)

    def safe_div(x, y): return float(x)/float(y) if (y and not np.isnan(y)) else np.nan

    p_ratio = safe_div(p_total, p_den)
    a_ratio = safe_div(a_total, a_den)

    rows = [[
        "상대적 팔 길이 지수",
        round(p_ratio, 3), round(a_ratio, 3), round(p_ratio - a_ratio, 3)
    ]]
    return pd.DataFrame(rows, columns=["항목","프로","일반","차이(프로-일반)"])
