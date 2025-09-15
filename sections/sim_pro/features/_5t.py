# sections/sim_pro/features/_5_ab_1_4_distance.py
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
    """'AX4' 같은 코드에서 값을 읽어온다."""
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")

def _angle_acb_signed(dy: float, dx: float) -> float:
    """
    직각삼각형(∠B=90°)에서 ∠ACB.
    - AB = 위의 Y값 = dy
    - BC = 위의 X값 = dx
    - 각도 = atan2(|AB|, |BC|) [deg]
    - 단, AB(=dy)가 음수면 각도를 음수로 표시
    """
    base = math.degrees(math.atan2(abs(dy), abs(dx)))
    return -base if dy < 0 else base

# ── 단일표: A=1행, B=4행 고정 ────────────────────────────────────────────────
def build_single_table(arr: np.ndarray) -> pd.DataFrame:
    # 좌표
    ax1, ay1, az1 = g(arr, "AX1"), g(arr, "AY1"), g(arr, "AZ1")
    ax4, ay4, az4 = g(arr, "AX4"), g(arr, "AY4"), g(arr, "AZ4")

    # 축별 차이 (B - A)
    dx = _safe(ax4 - ax1)   # AX4 - AX1
    dy = _safe(ay4 - ay1)   # AY4 - AY1
    dz = _safe(az4 - az1)   # AZ4 - AZ1

    # 직각삼각형 각도(∠ACB, Y<0 → 음수)
    ang_acb = _safe(_angle_acb_signed(dy, dx))

    # 3차원 거리 |AB|
    dist_ab = _safe(math.sqrt(dx*dx + dy*dy + dz*dz))

    rows = [
        ("AX4 - AX1 (X)", dx),
        ("AY4 - AY1 (Y)", dy),
        ("∠ACB (deg, Y<0→음수)", ang_acb),
        ("AZ4 - AZ1 (Z)", dz),
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
