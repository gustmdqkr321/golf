# sections/sim_pro/features/_7_ar_bg_dist.py
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

def _dist3(arr: np.ndarray, a: tuple[str,str,str], b: tuple[str,str,str]) -> float:
    ax, ay, az = g(arr, a[0]), g(arr, a[1]), g(arr, a[2])
    bx, by, bz = g(arr, b[0]), g(arr, b[1]), g(arr, b[2])
    return _safe(math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2))

# ── 단일표: 3개 행 (1행 거리, 4행 거리, 차이) ─────────────────────────────────
def build_single_table(arr: np.ndarray) -> pd.DataFrame:
    # A1/B1
    d1 = _dist3(arr, ("AR1","AS1","AT1"), ("BG1","BH1","BI1"))
    # A4/B4
    d4 = _dist3(arr, ("AR4","AS4","AT4"), ("BG4","BH4","BI4"))
    # Δ
    delta = _safe(d4 - d1)

    rows = [
        ("AB 거리 (프레임1: A=AR1,AS1,AT1 · B=BG1,BH1,BI1)", d1),
        ("AB 거리 (프레임4: A=AR4,AS4,AT4 · B=BG4,BH4,BI4)", d4),
        ("Δ거리 (4 − 1)", delta),
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
