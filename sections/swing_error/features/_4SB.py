# sections/swing_error/features/_4SB.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 엑셀 좌표 유틸 ──────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def col_series(arr: np.ndarray, letters: str, start: int, end: int) -> np.ndarray:
    i = col_letters_to_index(letters)
    s = arr[start - 1 : end, i]
    return pd.to_numeric(pd.Series(s, dtype="object"), errors="coerce").astype(float).to_numpy()

# ── 중심점(좌/우 평균) ───────────────────────────────────────────────────────
def _center_xyz(arr: np.ndarray, start: int, end: int,
                left: tuple[str,str,str], right: tuple[str,str,str]) -> np.ndarray:
    L = np.stack([col_series(arr, left[0], start, end),
                  col_series(arr, left[1], start, end),
                  col_series(arr, left[2], start, end)], axis=1)
    R = np.stack([col_series(arr, right[0], start, end),
                  col_series(arr, right[1], start, end),
                  col_series(arr, right[2], start, end)], axis=1)
    return (L + R) * 0.5  # (N,3)

# 어깨: (AL,AM,AN) & (BA,BB,BC) / 골반: (H,I,J) & (K,L,M)
def _shoulder_center(arr, s, e): return _center_xyz(arr, s, e, ("AL","AM","AN"), ("BA","BB","BC"))
def _pelvis_center(arr, s, e):   return _center_xyz(arr, s, e, ("H","I","J"),   ("K","L","M"))

# ── Side Bend 각도 (YZ-수직에 대한 θ=atan2(ΔY, √(ΔX²+ΔZ²))) ──────────────────
def _side_bend_angles(arr: np.ndarray, start: int = 1, end: int = 10) -> np.ndarray:
    S = _shoulder_center(arr, start, end)  # (N,3)
    P = _pelvis_center(arr, start, end)    # (N,3)
    dY = S[:, 1] - P[:, 1]
    dX = S[:, 0] - P[:, 0]
    dZ = S[:, 2] - P[:, 2]
    horiz = np.sqrt(dX * dX + dZ * dZ)
    angles = np.degrees(np.arctan2(dY, horiz))  # deg
    return angles

def _frame_labels(start: int, end: int, labels: list[str] | None = None) -> list[str]:
    nums = list(range(start, end + 1))
    if labels and len(labels) >= len(nums):
        return [f"{n} ({labels[i]})" for i, n in enumerate(nums)]
    return [str(n) for n in nums]

# ── 단일 리포트 표 ───────────────────────────────────────────────────────────
def build_side_bend_report(arr: np.ndarray, start: int = 1, end: int = 10,
                           labels: list[str] | None = None) -> pd.DataFrame:
    theta = _side_bend_angles(arr, start, end)                # (N,)
    dseg  = np.diff(theta, prepend=theta[0]); dseg[0] = 0.0

    nums   = list(range(start, end + 1))
    frames = _frame_labels(start, end, labels)

    rows = [[nums[i], frames[i], round(theta[i], 2), round(dseg[i], 2)]
            for i in range(len(theta))]

    # 섹션 합계 (1→4, 4→7, 1→7)
    backswing = round(theta[4-1] - theta[1-1], 2)
    downswing = round(theta[7-1] - theta[4-1], 2)
    total_1_7 = round(theta[7-1] - theta[1-1], 2)

    rows += [
        [101, "Backswing (1-4)", np.nan, backswing],
        [102, "Downswing (4-7)", np.nan, downswing],
        [103, "Total (1-7)",     np.nan, total_1_7],
    ]

    df = pd.DataFrame(rows, columns=["_ord","Frame","Side Bend (deg)","Section Change (deg)"])
    df = df.sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)
    return df

# ── 프로/일반 비교표 ─────────────────────────────────────────────────────────
def build_side_bend_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                            start: int = 1, end: int = 10,
                            labels: list[str] | None = None,
                            pro_name: str = "프로", ama_name: str = "일반") -> pd.DataFrame:
    p = build_side_bend_report(pro_arr, start, end, labels)
    a = build_side_bend_report(ama_arr, start, end, labels)

    p.columns = ["Frame", f"{pro_name} Side Bend (deg)", f"{pro_name} Section Change (deg)"]
    a.columns = ["Frame", f"{ama_name} Side Bend (deg)", f"{ama_name} Section Change (deg)"]

    # 정렬키 생성
    def _ord_key(s: str) -> int:
        if s.startswith("Backswing"): return 101
        if s.startswith("Downswing"): return 102
        if s.startswith("Total"):     return 103
        try: return int(str(s).split()[0])
        except: return 999

    p["_ord"] = p["Frame"].map(_ord_key)

    df = p.merge(a, on="Frame", how="outer").sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)

    # 차이(프로-일반)
    def _sub(x, y): return np.nan if (pd.isna(x) or pd.isna(y)) else float(x) - float(y)
    df["Side Bend Δ(프로-일반)"]      = [_sub(x,y) for x,y in zip(df[f"{pro_name} Side Bend (deg)"],     df[f"{ama_name} Side Bend (deg)"])]
    df["Section Change Δ(프로-일반)"] = [_sub(x,y) for x,y in zip(df[f"{pro_name} Section Change (deg)"], df[f"{ama_name} Section Change (deg)"])]
    return df
