# sections/sim_pro/features/_9_b_metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

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

def build_single_table(arr: np.ndarray) -> pd.DataFrame:
    b10 = _safe(g(arr, "B10"))
    b4  = _safe(g(arr, "B4"))
    b7  = _safe(g(arr, "B7"))
    denom = (b7 - b4)
    ratio = _safe(b4 / denom) if denom not in (0.0, -0.0) else float("nan")
    rows = [("B10", b10), ("B4/(B7-B4)", ratio)]
    return pd.DataFrame(rows, columns=["항목", "값"])

def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    df_p = build_single_table(pro_arr).rename(columns={"값": "프로"})
    df_a = build_single_table(ama_arr).rename(columns={"값": "일반"})
    df   = df_p.merge(df_a, on="항목", how="outer")
    for c in ("프로", "일반"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["차이(프로-일반)"] = (df["프로"] - df["일반"]).round(4)
    return df
