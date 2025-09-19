# sections/swing/features/_5rasi.py
from __future__ import annotations
import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def _get_xyz_series(arr: np.ndarray, cols3: list[str], max_frame: int = 10) -> np.ndarray:
    """
    프레임 1..max_frame 의 3D 좌표열 반환.
    데이터가 더 적으면 NaN이 포함될 수 있으나, 이후 계산은 안전하게 진행됨.
    """
    out = []
    for t in range(1, max_frame + 1):   # ← 1..10
        out.append([g(arr, f"{cols3[0]}{t}"),
                    g(arr, f"{cols3[1]}{t}"),
                    g(arr, f"{cols3[2]}{t}")])
    return np.asarray(out, float)
CM_TO_M = 0.01
CLUB_COLS = ["CN", "CO", "CP"]  # 클럽헤드 X,Y,Z 컬럼

def _clubhead_xyz_series(arr: np.ndarray, start: int = 1, end: int = 10) -> np.ndarray:
    """프레임 start..end의 클럽헤드 XYZ (m) -> shape: (N,3)"""
    out = []
    for t in range(start, end + 1):
        out.append([g(arr, f"{CLUB_COLS[0]}{t}"),
                    g(arr, f"{CLUB_COLS[1]}{t}"),
                    g(arr, f"{CLUB_COLS[2]}{t}")])
    pts_cm = np.asarray(out, dtype=float)
    return pts_cm * CM_TO_M  # m로 변환

def compute_total_arc_m(arr: np.ndarray, *, start: int = 1, end: int = 10) -> tuple[np.ndarray, float]:
    """
    구간별 아크 길이 Di(1→2 … 9→10) [m], 총합 T_arc [m] 반환
    Di = || P_{i+1} - P_i ||_2
    """
    pts = _clubhead_xyz_series(arr, start=start, end=end)     # (N,3) in meters
    diffs = pts[1:] - pts[:-1]                                # (N-1,3)
    Di = np.linalg.norm(diffs, axis=1)                        # (N-1,) in meters
    return Di, float(np.sum(Di))

def calculate_rasi(total_arc_m: float, arm_length_m: float, club_length_m: float) -> float:
    denom = (arm_length_m or 0.0) + (club_length_m or 0.0)
    return float("nan") if denom == 0 else total_arc_m / denom

def build_rasi_table_from_arrays(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    *,
    arm_len_pro: float,
    club_len_pro: float,
    arm_len_ama: float,
    club_len_ama: float,
) -> pd.DataFrame:
    """항목/프로/일반/차이 표: 총 아크 길이(m), 팔, 클럽, RASI"""
    _, T_pro = compute_total_arc_m(pro_arr)
    _, T_ama = compute_total_arc_m(ama_arr)

    rasi_pro = calculate_rasi(T_pro, arm_len_pro, club_len_pro)
    rasi_ama = calculate_rasi(T_ama, arm_len_ama, club_len_ama)

    ab_pro = compute_takeback_ab_distance_m(pro_arr)
    ab_ama = compute_takeback_ab_distance_m(ama_arr)
    rel_tb_pro = ab_pro / (abs(rasi_pro) if np.isfinite(rasi_pro) and rasi_pro != 0 else np.nan)
    rel_tb_ama = ab_ama / (abs(rasi_ama) if np.isfinite(rasi_ama) and rasi_ama != 0 else np.nan)

    rows = [
        # ["total_arc (m)",   round(T_pro, 3),          round(T_ama, 3)],
        # ["arm_length (m)",  round(float(arm_len_pro), 3), round(float(arm_len_ama), 3)],
        # ["club_length (m)", round(float(club_len_pro), 3), round(float(club_len_ama), 3)],
        ["RASI",            round(rasi_pro, 3),       round(rasi_ama, 3)],
        ["상대적 테이크백 아크 크기 지수", round(rel_tb_pro, 3), round(rel_tb_ama, 3)],
    ]
    df = pd.DataFrame(rows, columns=["항목", "프로", "일반"])
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                          - pd.to_numeric(df["일반"], errors="coerce")).round(3)
    return df

def build_rasi_segments_table(arr: np.ndarray) -> pd.DataFrame:
    """
    (옵션) 구간별 Di를 보여주는 표: 1-2 .. 9-10, total_arc(m)
    """
    Di, T = compute_total_arc_m(arr)
    labels = [f"{i}-{i+1}" for i in range(1, len(Di)+1)]
    df = pd.DataFrame({"구간": labels, "Di (m)": np.round(Di, 3)})
    df.loc[len(df)] = ["total_arc", round(T, 3)]
    return df

def compute_takeback_ab_distance_m(arr: np.ndarray) -> float:
    """
    프레임 1의 A(오른쪽 무릎: CB,CC,CD)와 B(오른쪽 골반: K,L,M) 사이 거리 [m]
    """
    a = np.array([g(arr, "CB1"), g(arr, "CC1"), g(arr, "CD1")], dtype=float)
    b = np.array([g(arr, "K1"),  g(arr, "L1"),  g(arr, "M1")],  dtype=float)
    return float(np.linalg.norm((a - b) * 0.01))