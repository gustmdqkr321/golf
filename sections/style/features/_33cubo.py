# sections/swing/features/_27_bowing_table.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 엑셀 셀 헬퍼 ──────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters.upper():
        idx = idx*26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ── 각 프레임 보잉 각도(°) 계산 → 정규화(±90 보정) → 상대각(프레임1 기준) ──
def _compute_bowing_angles_from_arr(arr: np.ndarray) -> list[float]:
    angles = []
    for n in range(1, 11):
        AR,AS,AT = g(arr, f"AR{n}"), g(arr, f"AS{n}"), g(arr, f"AT{n}")  # 팔꿈치
        AX,AY,AZ = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")  # 손목
        AL,AM,AN = g(arr, f"AL{n}"), g(arr, f"AM{n}"), g(arr, f"AN{n}")  # 어깨
        CN,CO,CP = g(arr, f"CN{n}"), g(arr, f"CO{n}"), g(arr, f"CP{n}")  # 클럽헤드

        E = np.array([AR, AS, AT], float)
        W = np.array([AX, AY, AZ], float)
        S = np.array([AL, AM, AN], float)
        C = np.array([CN, CO, CP], float)

        x_axis = W - E
        nx = np.linalg.norm(x_axis)
        if nx != 0: x_axis = x_axis / nx

        y_axis = np.cross(W - S, C - W)
        ny = np.linalg.norm(y_axis)
        if ny != 0: y_axis = y_axis / ny

        z_axis = np.cross(x_axis, y_axis)
        nz = np.linalg.norm(z_axis)
        if nz != 0: z_axis = z_axis / nz

        R = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3
        local = R.T @ (C - W)
        theta = math.degrees(math.atan2(local[2], local[0]))  # X–Z 기울기
        angles.append(theta)

    # ±90° 보정
    normed = []
    for th in angles:
        if th > 90:
            normed.append(180 - th)
        elif th < -90:
            normed.append(-180 - th)
        else:
            normed.append(th)

    # 상대각(프레임1 기준)
    base = normed[0]
    rel = [float(x - base) for x in normed]
    return rel  # 길이 10

def _maintenance(top: float, dh: float) -> float:
    """TOP→DH 유지지수(%) = (1 - |DH-TOP|/|TOP|)*100, 부호/크기 과도변화 시 음수"""
    if top == 0:
        return np.nan
    ratio = abs(dh - top) / abs(top)
    if np.sign(top) == np.sign(dh) and abs(dh) <= abs(top):
        return round((1 - ratio) * 100, 2)
    return round(-ratio * 100, 2)

# ── 1) 요약표(4, 6, 13) ───────────────────────────────────────────────────
def build_bowing_summary_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    반환(3행):
      4) TOP Rel. Bowing(°)
      6) DH  Rel. Bowing(°)
      13) Bowing_Maintenance(%)
    컬럼: [항목, 프로, 일반, 차이(프로-일반)]
    """
    pr = _compute_bowing_angles_from_arr(pro_arr)
    am = _compute_bowing_angles_from_arr(ama_arr)

    top_idx, dh_idx = 3, 5  # 0-based: 4=TOP, 6=DH
    p_top, a_top = pr[top_idx], am[top_idx]
    p_dh,  a_dh  = pr[dh_idx],  am[dh_idx]
    p_m = _maintenance(p_top, p_dh)
    a_m = _maintenance(a_top, a_dh)

    rows = [
        ["4) TOP", round(p_top, 2), round(a_top, 2), round(p_top - a_top, 2)],
        ["6) DH",  round(p_dh,  2), round(a_dh,  2), round(p_dh  - a_dh,  2)],
        ["Bowing_Maintenance(%)",
                                 p_m, a_m,
                                 (round(p_m - a_m, 2) if (pd.notna(p_m) and pd.notna(a_m)) else np.nan)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

# ── 2) 전체표(옵션) ────────────────────────────────────────────────────────
def build_bowing_full_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    index:
      ['ADD','BH','BH2','TOP','TR','DH','IMP','FH1','FH2','FIN',
       '1-4','4-6','Bowing_Maintenance']
    columns:
      ['프로 Rel. Bowing(°)','일반 Rel. Bowing(°)','Δ프로','Δ일반']
    * '1-4','4-6' 행에는 구간 Δ 값 기입
    * 'Bowing_Maintenance' 행에는 Δ컬럼에 유지지수(%) 기입
    """
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

    pr = _compute_bowing_angles_from_arr(pro_arr)
    am = _compute_bowing_angles_from_arr(ama_arr)

    p_delta = [np.nan] + [round(pr[i] - pr[i-1], 2) for i in range(1, 10)]
    a_delta = [np.nan] + [round(am[i] - am[i-1], 2) for i in range(1, 10)]

    top_idx, dh_idx = 3, 5
    p_1_4 = round(pr[top_idx] - pr[0], 2)
    p_4_6 = round(pr[dh_idx]  - pr[top_idx], 2)
    a_1_4 = round(am[top_idx] - am[0], 2)
    a_4_6 = round(am[dh_idx]  - am[top_idx], 2)

    p_m = _maintenance(pr[top_idx], pr[dh_idx])
    a_m = _maintenance(am[top_idx], am[dh_idx])

    df = pd.DataFrame({
        "프로 Rel. Bowing(°)": [round(x, 2) for x in pr],
        "일반 Rel. Bowing(°)": [round(x, 2) for x in am],
        "Δ프로":               p_delta,
        "Δ일반":               a_delta,
    }, index=labels)

    extra = pd.DataFrame({
        "프로 Rel. Bowing(°)": [p_1_4, p_4_6, np.nan],
        "일반 Rel. Bowing(°)": [a_1_4, a_4_6, np.nan],
        "Δ프로":               [np.nan, np.nan, p_m],
        "Δ일반":               [np.nan, np.nan, a_m],
    }, index=["1-4", "4-6", "Bowing_Maintenance"])

    out = pd.concat([df, extra], axis=0)
    out.index.name = "Frame"
    return out
