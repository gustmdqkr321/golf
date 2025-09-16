# sections/swing_error/features/_6OTT.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── util ─────────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _center_x(arr: np.ndarray, left_col: str, right_col: str, row: int) -> float:
    return 0.5 * (g(arr, f"{left_col}{row}") + g(arr, f"{right_col}{row}"))

# wrist center X = (AX + BM)/2
def _wrist_x(arr: np.ndarray, row: int) -> float:
    return _center_x(arr, "AX", "BM", row)

# shoulder center X = (AL + BA)/2
def _shoulder_x(arr: np.ndarray, row: int) -> float:
    return _center_x(arr, "AL", "BA", row)

# ── 단일표 (프레임 4,5,6) ────────────────────────────────────────────────────
def build_over_the_top_table(
    arr: np.ndarray,
    frames: tuple[int, int, int] = (4, 5, 6),
    chd_col: str = "CN",            # CHD 열(기본 CN)
    wrist_r_col: str = "BM",        # 오른손목 열(기본 BM)
) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(frames):
        prev = f - 1  # 이전 프레임(4의 기준은 prev=3)

        wx  = _wrist_x(arr, f)
        sx  = _shoulder_x(arr, f)
        wsx = wx - sx

        if i == 0:
            m1 = 0.0
            m2 = 0.0
        else:
            wx_prev  = _wrist_x(arr, prev)
            sx_prev  = _shoulder_x(arr, prev)
            m1 = wx - wx_prev
            m2 = (wx - sx) - (wx_prev - sx_prev)

        chd_vs_wrist = g(arr, f"{chd_col}{f}") - g(arr, f"{wrist_r_col}{f}")

        rows += [
            ["어깨 대비 손목 X(cm)", f, wsx,     "((AX+BM)/2) - ((AL+BA)/2)"],
            ["손목 자체 이동(cm)",   f, m1,      "wristX[f] - wristX[f-1]"],
            ["어깨 대비 손목 이동",  f, m2,      "(wristX-shoulderX)[f] - (…)[f-1]"],
            ["오른 손목 대비 CHD",   f, chd_vs_wrist, f"{chd_col}{f} - {wrist_r_col}{f}"],
        ]

    df = pd.DataFrame(rows, columns=["항목", "Frame", "값", "식"])
    # 보기 좋게 정렬
    order = {"어깨 대비 손목 X(cm)": 1, "손목 자체 이동(cm)": 2, "어깨 대비 손목 이동": 3, "오른 손목 대비 CHD": 4}
    df["_ord"] = df["항목"].map(order)
    df = df.sort_values(["_ord", "Frame"]).drop(columns="_ord").reset_index(drop=True)
    return df

# ── 프로/일반 비교표 ─────────────────────────────────────────────────────────
def build_over_the_top_compare(
    pro_arr: np.ndarray, ama_arr: np.ndarray,
    frames: tuple[int, int, int] = (4, 5, 6),
    chd_col: str = "CN", wrist_r_col: str = "BM"
) -> pd.DataFrame:
    p = build_over_the_top_table(pro_arr, frames=frames, chd_col=chd_col, wrist_r_col=wrist_r_col)
    a = build_over_the_top_table(ama_arr, frames=frames, chd_col=chd_col, wrist_r_col=wrist_r_col)

    df = (
        p.merge(a, on=["항목", "Frame"], suffixes=("·프로", "·일반"))
         .rename(columns={"값·프로": "프로", "값·일반": "일반", "식·프로": "식"})  # 식은 동일하므로 하나만
    )
    df["차이(프로-일반)"] = df["프로"] - df["일반"]
    # 컬럼 정리
    df = df[["항목", "Frame", "식", "프로", "일반", "차이(프로-일반)"]]
    return df
