# sections/sim_pro/features/_1body.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 기본 유틸 ────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _safe_float(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")

def diff(arr: np.ndarray, left_code: str, right_code: str) -> float:
    return _safe_float(g(arr, left_code) - g(arr, right_code))

def hypot2(a: float, b: float) -> float:
    return _safe_float(math.hypot(a, b))

def leg_from_hyp(ac: float, ab: float) -> float:
    v = max(0.0, (ac ** 2) - (ab ** 2))
    return _safe_float(math.sqrt(v))

def dist3(arr: np.ndarray, a_xyz: tuple[str,str,str], b_xyz: tuple[str,str,str]) -> float:
    ax, ay, az = (g(arr, a_xyz[0]), g(arr, a_xyz[1]), g(arr, a_xyz[2]))
    bx, by, bz = (g(arr, b_xyz[0]), g(arr, b_xyz[1]), g(arr, b_xyz[2]))
    return _safe_float(math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2))

# ── 단일표: 항상 row=1로 고정 ────────────────────────────────────────────────
def build_single_table(arr: np.ndarray, row: int | None = None) -> pd.DataFrame:
    r = "1"  # ← 여기서 고정

    # 1) knee–ankle 거리: a=(CB,CC,CD) b=(CK,CL,CM)
    v1 = dist3(arr, (f"CB{r}", f"CC{r}", f"CD{r}"), (f"CK{r}", f"CL{r}", f"CM{r}"))

    # 2) knee–waist 거리: a=(CB,CC,CD) b=(K,L,M)
    v2 = dist3(arr, (f"CB{r}", f"CC{r}", f"CD{r}"), (f"K{r}", f"L{r}", f"M{r}"))

    # 3) 직각 AC | AB=K-BA,  BC=BB-L
    ab3 = diff(arr, f"K{r}", f"BA{r}")
    bc3 = diff(arr, f"BB{r}", f"L{r}")
    v3  = hypot2(ab3, bc3)

    # 4) 직각 AC | AB=BA-AC, BC=AD-BB
    ab4 = diff(arr, f"BA{r}", f"AC{r}")
    bc4 = diff(arr, f"AD{r}", f"BB{r}")
    v4  = hypot2(ab4, bc4)

    # 5) 1+2+3+4 + CL
    v5 = _safe_float(v1 + v2 + v3 + v4 + g(arr, f"CL{r}"))

    # 6) 직각 AC | AB=BB-BH, BC=BA-BG
    ab6 = diff(arr, f"BB{r}", f"BH{r}")
    bc6 = diff(arr, f"BA{r}", f"BG{r}")
    v6  = hypot2(ab6, bc6)

    # 7) 직각 AC | AB=BH-BM, BC=BG-BM
    ab7 = diff(arr, f"BH{r}", f"BM{r}")
    bc7 = diff(arr, f"BG{r}", f"BM{r}")
    v7  = hypot2(ab7, bc7)

    # 8) 직각 BC | AB=BM-CN, AC=BN → BC
    ab8 = diff(arr, f"BM{r}", f"CN{r}")
    ac8 = g(arr, f"BN{r}")
    v8  = leg_from_hyp(ac8, ab8)

    # 9) 6+7+8
    v9 = _safe_float(v6 + v7 + v8)

    # 10) |AN| + |BC|
    v10 = _safe_float(abs(g(arr, f"AN{r}")) + abs(g(arr, f"BC{r}")))

    # 11) |J| + |M|
    v11 = _safe_float(abs(g(arr, f"J{r}")) + abs(g(arr, f"M{r}")))

    return pd.DataFrame(
        [
            ("01) knee–ankle 거리  | a=(CB,CC,CD), b=(CK,CL,CM)", v1),
            ("02) knee–waist 거리  | a=(CB,CC,CD), b=(K,L,M)",    v2),
            ("03) 직각 AC | AB=K-BA,  BC=BB-L",                    v3),
            ("04) 직각 AC | AB=BA-AC, BC=AD-BB",                   v4),
            ("05) (1+2+3+4)+CL",                                   v5),
            ("06) 직각 AC | AB=BB-BH, BC=BA-BG",                   v6),
            ("07) 직각 AC | AB=BH-BM, BC=BG-BM",                   v7),
            ("08) 직각 BC | AB=BM-CN, AC=BN  → BC",               v8),
            ("09) (6+7+8)",                                        v9),
            ("10) |AN| + |BC|",                                   v10),
            ("11) |J| + |M|",                                     v11),
        ],
        columns=["항목", "값"]
    )

# ── 비교표(프로 vs 일반) : row 인자는 받아도 무시 ─────────────────────────────
def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    df_p = build_single_table(pro_arr).rename(columns={"값": "프로"})
    df_a = build_single_table(ama_arr).rename(columns={"값": "일반"})
    df   = df_p.merge(df_a, on="항목", how="outer")
    for c in ("프로", "일반"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["차이(프로-일반)"] = (df["프로"] - df["일반"]).round(4)
    return df
