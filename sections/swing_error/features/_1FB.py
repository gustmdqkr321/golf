# sections/swing_error/features/frontal_bend.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _mid(a: float, b: float) -> float:
    return 0.5 * (a + b)

def _frame_labels(start: int, end: int, labels: list[str] | None = None) -> list[str]:
    frames = list(range(start, end + 1))
    if labels and len(labels) >= len(frames):
        return [f"{f} ({labels[i]})" for i, f in enumerate(frames)]
    return [str(f) for f in frames]

# ── Step 1. 중심점: 골반 P, 어깨 S (프레임 row) ─────────────────────────────
def center_P(arr: np.ndarray, row: int) -> tuple[float, float, float]:
    # Left Waist:  H,I,J | Right Waist: K,L,M
    Xp = _mid(g(arr, f"H{row}"),  g(arr, f"K{row}"))
    Yp = _mid(g(arr, f"I{row}"),  g(arr, f"L{row}"))
    Zp = _mid(g(arr, f"J{row}"),  g(arr, f"M{row}"))
    return Xp, Yp, Zp

def center_S(arr: np.ndarray, row: int) -> tuple[float, float, float]:
    # Left Shoulder: AL,AM,AN | Right Shoulder: BA,BB,BC
    Xs = _mid(g(arr, f"AL{row}"), g(arr, f"BA{row}"))
    Ys = _mid(g(arr, f"AM{row}"), g(arr, f"BB{row}"))
    Zs = _mid(g(arr, f"AN{row}"), g(arr, f"BC{row}"))
    return Xs, Ys, Zs

# ── Step 2. Frontal Bend(θ) : 수직을 0°로 정의 ──────────────────────────────
def frontal_bend_angle(arr: np.ndarray, row: int) -> tuple[float, float, float]:
    """
    반환: (θ[deg], dX, dY)
      dX = Xs - Xp, dY = Ys - Yp
      θ  = 90° - atan2(dY, dX) * 180/π
         (앞으로 숙여질수록 +, 수직=0°)
    """
    Xp, Yp, _ = center_P(arr, row)
    Xs, Ys, _ = center_S(arr, row)
    dX = Xs - Xp
    dY = Ys - Yp
    theta = 90.0 - math.degrees(math.atan2(dY, dX))
    return float(theta), float(dX), float(dY)

# ── 단일표 (프레임별 + 섹션 요약) ────────────────────────────────────────────
def build_frontal_bend_report(arr: np.ndarray,
                              start: int = 1, end: int = 10,
                              labels: list[str] | None = None) -> pd.DataFrame:
    """
    형식(_2BH.py와 동일 컨벤션):
      Frame | Frontal Bend (deg) | Section Change (deg)
    - 프레임 행: Δseg(θ_i - θ_{i-1}), 첫 프레임은 0.00
    - 요약 행: Backswing(1-4), Downswing(4-7), Total(1-7)
    """
    # θ 시퀀스
    thetas: list[float] = []
    for r in range(start, end + 1):
        th, _, _ = frontal_bend_angle(arr, r)
        thetas.append(float(th))

    # Δseg
    dseg = np.diff(thetas, prepend=thetas[0]).astype(float)
    dseg[0] = 0.0

    fr_col   = _frame_labels(start, end, labels)
    frame_no = list(range(start, end + 1))

    rows: list[list[object]] = [
        [frame_no[i], fr_col[i], round(thetas[i], 2), round(dseg[i], 2)]
        for i in range(len(thetas))
    ]

    # 섹션 합계(Δ는 누적차로 표현)
    backswing = round(thetas[4-1] - thetas[1-1], 2)   # 1→4
    downswing = round(thetas[7-1] - thetas[4-1], 2)   # 4→7
    total_1_7 = round(thetas[7-1] - thetas[1-1], 2)   # 1→7

    rows += [
        [101, "Backswing (1-4)", np.nan, backswing],
        [102, "Downswing (4-7)", np.nan, downswing],
        [103, "Total (1-7)",     np.nan, total_1_7],
    ]

    df = pd.DataFrame(rows, columns=["_ord", "Frame", "Frontal Bend (deg)", "Section Change (deg)"])
    df = df.sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)
    return df

# ── 프로/일반 비교표 ─────────────────────────────────────────────────────────
def build_frontal_bend_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                               start: int = 1, end: int = 10,
                               labels: list[str] | None = None,
                               pro_name: str = "프로", ama_name: str = "일반") -> pd.DataFrame:
    """
    형식:
      Frame | 프로 Frontal Bend (deg) | 프로 Section Change (deg) | 일반 ... | Δ(프로-일반) 2종
    프레임 라벨/요약 행/정렬 규칙은 _2BH.py와 동일
    """
    df_p = build_frontal_bend_report(pro_arr, start, end, labels)
    df_a = build_frontal_bend_report(ama_arr, start, end, labels)

    # 정렬용 키
    def _ord_from_frame(s: str) -> int:
        if s.startswith("Backswing"): return 101
        if s.startswith("Downswing"): return 102
        if s.startswith("Total"):     return 103
        try:
            return int(s.split()[0])
        except Exception:
            return 999

    df_p["_ord"] = df_p["Frame"].map(_ord_from_frame)

    df_p.columns = [
        "Frame",
        f"{pro_name} Frontal Bend (deg)",
        f"{pro_name} Section Change (deg)",
        "_ord",
    ]
    df_a.columns = [
        "Frame",
        f"{ama_name} Frontal Bend (deg)",
        f"{ama_name} Section Change (deg)",
    ]

    df = df_p.merge(df_a, on="Frame", how="outer")
    df = df.sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)

    # Δ(프로-일반)
    def _sub(a, b):
        return np.nan if (pd.isna(a) or pd.isna(b)) else round(float(a) - float(b), 2)

    df["Frontal Bend Δ(프로-일반)"]     = [
        _sub(a, b) for a, b in zip(df[f"{pro_name} Frontal Bend (deg)"], df[f"{ama_name} Frontal Bend (deg)"])
    ]
    df["Section Change Δ(프로-일반)"] = [
        _sub(a, b) for a, b in zip(df[f"{pro_name} Section Change (deg)"], df[f"{ama_name} Section Change (deg)"])
    ]

    # 숫자 컬럼 강제 숫자화(안전)
    num_cols = [
        f"{pro_name} Frontal Bend (deg)",
        f"{pro_name} Section Change (deg)",
        f"{ama_name} Frontal Bend (deg)",
        f"{ama_name} Section Change (deg)",
        "Frontal Bend Δ(프로-일반)",
        "Section Change Δ(프로-일반)",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
