# sections/swing/features/_26_cocking_table.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 엑셀 셀 헬퍼 ──────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ── ∠ABC(AX가 꼭짓점 A, AR/AS/AT=R, CN/CO/CP=C) ─────────────────────────
def _angles_from_arr(arr: np.ndarray) -> list[float]:
    """프레임 1..10의 ∠ABC(°) 리스트 반환"""
    out = []
    for n in range(1, 11):
        AR, AS, AT = g(arr, f"AR{n}"), g(arr, f"AS{n}"), g(arr, f"AT{n}")
        AX, AY, AZ = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")
        CN, CO, CP = g(arr, f"CN{n}"), g(arr, f"CO{n}"), g(arr, f"CP{n}")
        v1 = np.array([AR-AX, AS-AY, AT-AZ], dtype=float)  # AB
        v2 = np.array([CN-AX, CO-AY, CP-AZ], dtype=float)  # AC
        den = np.linalg.norm(v1) * np.linalg.norm(v2)
        if den == 0:
            out.append(np.nan)
            continue
        cos_t = float(np.clip(v1.dot(v2) / den, -1.0, 1.0))
        out.append(math.degrees(math.acos(cos_t)))
    return out

def _maintenance_percent(angle_top: float, angle_dh: float) -> float:
    """
    Cocking_Maintenance (%):
      (1 - |DH - TOP| / |TOP|) * 100
    """
    if angle_top == 0 or np.isnan(angle_top) or np.isnan(angle_dh):
        return np.nan
    return round((1 - abs(angle_dh - angle_top) / abs(angle_top)) * 100, 2)

# ── 1) 요약 표(4, 6, 13번) ───────────────────────────────────────────────
def build_cocking_summary_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    반환(3행):
      4) TOP ∠ABC
      6) DH  ∠ABC
      13) Cocking_Maintenance (%)
    컬럼: [항목, 프로, 일반, 차이(프로-일반)]
    """
    p = _angles_from_arr(pro_arr)
    a = _angles_from_arr(ama_arr)

    top_idx, dh_idx = 3, 5  # 0기준: 4=TOP, 6=DH
    p_top, a_top = p[top_idx], a[top_idx]
    p_dh,  a_dh  = p[dh_idx],  a[dh_idx]
    p_maint = _maintenance_percent(p_top, p_dh)
    a_maint = _maintenance_percent(a_top, a_dh)

    rows = [
        ["4) TOP",  round(p_top, 2) if pd.notna(p_top) else np.nan,
                             round(a_top, 2) if pd.notna(a_top) else np.nan,
                             round((p_top - a_top), 2) if pd.notna(p_top) and pd.notna(a_top) else np.nan],
        ["6) DH",   round(p_dh,  2) if pd.notna(p_dh)  else np.nan,
                             round(a_dh,  2) if pd.notna(a_dh)  else np.nan,
                             round((p_dh - a_dh),  2) if pd.notna(p_dh)  and pd.notna(a_dh)  else np.nan],
        ["Cocking_Maintenance(%)",
                             p_maint, a_maint,
                             round((p_maint - a_maint), 2) if pd.notna(p_maint) and pd.notna(a_maint) else np.nan],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

# ── 2) 전체 표(프레임별 + 구간 + 유지지수) ────────────────────────────────
def build_cocking_full_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    index:
      ['ADD','BH','BH2','TOP','TR','DH','IMP','FH1','FH2','FIN',
       '1-4','4-6','Cocking_Maintenance']
    columns:
      ['프로 ∠ABC(°)','일반 ∠ABC(°)','Δ프로(°)','Δ일반(°)']
    * 1-4, 4-6 행은 각도 컬럼에 구간 Δ값을 기입
    * Cocking_Maintenance 행은 Δ컬럼에 유지지수(%) 기입
    """
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

    p = _angles_from_arr(pro_arr)
    a = _angles_from_arr(ama_arr)

    # Δ (첫 행은 결측)
    p_delta = [np.nan] + [round(p[i] - p[i-1], 2) if pd.notna(p[i]) and pd.notna(p[i-1]) else np.nan
                          for i in range(1, 10)]
    a_delta = [np.nan] + [round(a[i] - a[i-1], 2) if pd.notna(a[i]) and pd.notna(a[i-1]) else np.nan
                          for i in range(1, 10)]

    # 구간 Δ
    top_idx, dh_idx = 3, 5
    p_1_4 = round(p[top_idx] - p[0], 2) if pd.notna(p[top_idx]) and pd.notna(p[0]) else np.nan
    p_4_6 = round(p[dh_idx]  - p[top_idx], 2) if pd.notna(p[dh_idx]) and pd.notna(p[top_idx]) else np.nan
    a_1_4 = round(a[top_idx] - a[0], 2) if pd.notna(a[top_idx]) and pd.notna(a[0]) else np.nan
    a_4_6 = round(a[dh_idx]  - a[top_idx], 2) if pd.notna(a[dh_idx]) and pd.notna(a[top_idx]) else np.nan

    # 유지지수(%)
    p_maint = _maintenance_percent(p[top_idx], p[dh_idx])
    a_maint = _maintenance_percent(a[top_idx], a[dh_idx])

    # 본문(프레임 1~10)
    df = pd.DataFrame({
        "프로 ∠ABC(°)": [round(x, 2) if pd.notna(x) else np.nan for x in p],
        "일반 ∠ABC(°)": [round(x, 2) if pd.notna(x) else np.nan for x in a],
        "Δ프로(°)":      p_delta,
        "Δ일반(°)":      a_delta,
    }, index=labels)

    # 구간/유지 행 추가
    extra_idx = ["1-4", "4-6", "Cocking_Maintenance"]
    extra = pd.DataFrame({
        "프로 ∠ABC(°)": [p_1_4, p_4_6, np.nan],
        "일반 ∠ABC(°)": [a_1_4, a_4_6, np.nan],
        "Δ프로(°)":      [np.nan, np.nan, p_maint],
        "Δ일반(°)":      [np.nan, np.nan, a_maint],
    }, index=extra_idx)

    out = pd.concat([df, extra], axis=0)
    out.index.name = "Frame"
    return out
