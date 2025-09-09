# sections/swing/features/_25_hinging.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ── 힌징(한 프레임) 계산 ─────────────────────────────────────────────────────
def compute_hinging(AX, AY, AZ, AR, AS, AT, CN, CO, CP) -> float:
    num = (AX - AR) * (CO - AY) - (AY - AS) * (CN - AX)
    len_aw = math.sqrt((AX-AR)**2 + (AY-AS)**2 + (AZ-AT)**2)
    len_wc = math.sqrt((CN-AX)**2 + (CO-AY)**2 + (CP-AZ)**2)
    denom = (len_aw * len_wc) if (len_aw and len_wc) else 0.0
    ratio = (num/denom) if denom != 0.0 else 0.0
    ratio = max(-1.0, min(1.0, ratio))
    return round(-math.degrees(math.asin(ratio)), 2)

def _hinging_series(arr: np.ndarray) -> list[float]:
    out = []
    for i in range(1, 11):
        AR, AS, AT = g(arr, f"AR{i}"), g(arr, f"AS{i}"), g(arr, f"AT{i}")
        AX, AY, AZ = g(arr, f"AX{i}"), g(arr, f"AY{i}"), g(arr, f"AZ{i}")
        CN, CO, CP = g(arr, f"CN{i}"), g(arr, f"CO{i}"), g(arr, f"CP{i}")
        out.append(compute_hinging(AX, AY, AZ, AR, AS, AT, CN, CO, CP))
    return out

def signed_maintenance_score(top: float, dh: float) -> float:
    """TOP→DH 사이 signed 유지지수(%)"""
    delta = dh - top
    drop_ratio = abs(delta) / abs(top) if top != 0 else 0.0
    if np.sign(top) == np.sign(dh) and abs(dh) <= abs(top):
        return round((1 - drop_ratio) * 100, 2)
    else:
        return round(-drop_ratio * 100, 2)

# ── 1) 1–4 구간 한 줄 표 ────────────────────────────────────────────────────
def build_hinging_1_4_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    r = _hinging_series(pro_arr)
    a = _hinging_series(ama_arr)
    r_1_4 = round(r[3] - r[0], 2)   # frame4 - frame1
    a_1_4 = round(a[3] - a[0], 2)
    diff  = round(r_1_4 - a_1_4, 2)
    return pd.DataFrame(
        [["Hinging 1-4", r_1_4, a_1_4, diff]],
        columns=["항목", "프로", "일반", "차이(프로-일반)"]
    )

# ── 2) 전체표(프레임별 + 구간 + 유지지수) ───────────────────────────────────
def build_hinging_full_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    r = _hinging_series(pro_arr)
    a = _hinging_series(ama_arr)

    # 첫 프레임의 델타는 결측으로
    r_delta = [np.nan] + [round(r[i] - r[i-1], 2) for i in range(1, 10)]
    a_delta = [np.nan] + [round(a[i] - a[i-1], 2) for i in range(1, 10)]

    top_idx, dh_idx = 3, 5
    r_1_4 = round(r[top_idx] - r[0], 2)
    r_4_6 = round(r[dh_idx] - r[top_idx], 2)
    a_1_4 = round(a[top_idx] - a[0], 2)
    a_4_6 = round(a[dh_idx] - a[top_idx], 2)

    r_maint = signed_maintenance_score(r[top_idx], r[dh_idx])
    a_maint = signed_maintenance_score(a[top_idx], a[dh_idx])

    index = [str(i) for i in range(1, 11)] + ["1-4", "4-6", "유지지수(%)"]
    data = {
        "프로 힌징(°)": r + [r_1_4, r_4_6, np.nan],
        "Δ프로":        r_delta + [np.nan, np.nan, r_maint],
        "일반 힌징(°)": a + [a_1_4, a_4_6, np.nan],
        "Δ일반":        a_delta + [np.nan, np.nan, a_maint],
    }
    df = pd.DataFrame(data, index=index)
    df.index.name = "Frame"
    return df

