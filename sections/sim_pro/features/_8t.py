# sections/sim_pro/features/_8_cnco_cp_diffs.py
from __future__ import annotations
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

def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")

# ── 단일표 (행 4 고정) ───────────────────────────────────────────────────────
def build_single_table(arr: np.ndarray) -> pd.DataFrame:
    r = "4"
    v1 = _safe(g(arr, f"CN{r}") - g(arr, f"AY{r}"))  # CN4 - AY4
    v2 = _safe(g(arr, f"CO{r}") - g(arr, f"AY{r}"))  # CO4 - AY4
    v3 = _safe(g(arr, f"CP{r}") - g(arr, f"AZ{r}"))  # CP4 - AZ4
    rows = [("CN4 - AY4", v1), ("CO4 - AY4", v2), ("CP4 - AZ4", v3)]
    return pd.DataFrame(rows, columns=["항목", "값"])

# ── 비교표 (프로 vs 일반) ───────────────────────────────────────────────────
def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    df_p = build_single_table(pro_arr).rename(columns={"값": "프로"})
    df_a = build_single_table(ama_arr).rename(columns={"값": "일반"})
    df   = df_p.merge(df_a, on="항목", how="outer")
    for c in ("프로", "일반"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["차이(프로-일반)"] = (df["프로"] - df["일반"]).round(4)
    return df
