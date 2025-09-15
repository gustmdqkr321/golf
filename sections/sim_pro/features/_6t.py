# sections/sim_pro/features/_6_ab4_to_mid_avg.py
from __future__ import annotations
import numpy as np
import pandas as pd
import math

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

def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")

def _angle_acb_signed(dy: float, dx: float) -> float:
    """직각삼각형(∠B=90°)에서 ∠ACB. Y<0이면 음수 각도."""
    base = math.degrees(math.atan2(abs(dy), abs(dx)))
    return -base if dy < 0 else base

# ── 단일표: 행 4 고정 ────────────────────────────────────────────────────────
def build_single_table(arr: np.ndarray) -> pd.DataFrame:
    # A(AX4, AY4, AZ4)
    ax, ay, az = g(arr, "AX4"), g(arr, "AY4"), g(arr, "AZ4")

    # B = ((AL4+BA4)/2, (AM4+BB4)/2, (AN4+BC4)/2)
    bx = (g(arr, "AL4") + g(arr, "BA4")) / 2.0
    by = (g(arr, "AM4") + g(arr, "BB4")) / 2.0
    bz = (g(arr, "AN4") + g(arr, "BC4")) / 2.0

    # 축별 차이 (A - B)
    dx = _safe(ax - bx)  # AX4 - ((AL4+BA4)/2)
    dy = _safe(ay - by)  # AY4 - ((AM4+BB4)/2)
    dz = _safe(az - bz)  # AZ4 - ((AN4+BC4)/2)

    # 직각삼각형 각도(∠ACB), Y<0 → 음수
    ang_acb = _safe(_angle_acb_signed(dy, dx))

    # 3차원 거리 |AB|
    dist_ab = _safe(math.sqrt(dx*dx + dy*dy + dz*dz))

    rows = [
        ("AX4 - ((AL4+BA4)/2) (X)", dx),
        ("AY4 - ((AM4+BB4)/2) (Y)", dy),
        ("∠ACB (deg, Y<0→음수)", ang_acb),
        ("AZ4 - ((AN4+BC4)/2) (Z)", dz),
        ("① AB 거리(3D)", dist_ab),
    ]
    return pd.DataFrame(rows, columns=["항목", "값"])

# ── 비교표(프로 vs 일반) ────────────────────────────────────────────────────
def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    df_p = build_single_table(pro_arr).rename(columns={"값": "프로"})
    df_a = build_single_table(ama_arr).rename(columns={"값": "일반"})
    df   = df_p.merge(df_a, on="항목", how="outer")
    for c in ("프로", "일반"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["차이(프로-일반)"] = (df["프로"] - df["일반"]).round(4)
    return df
