# sections/swing/features/_25_26.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 공통 셀 헬퍼 ─────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def _fmt(x: float) -> float:
    return float(np.round(x, 2))

# ── 25) 6 L WRI/CHD X : CN6 - AX6 ───────────────────────────────────────────
def build_25_wri_chd_x(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    항목: '6 L WRI/CHD X (CN6-AX6)'
    컬럼: 항목 / 프로 / 일반 / 차이(프로-일반)
    """
    def metric(arr: np.ndarray) -> float:
        return g(arr, "CN6") - g(arr, "AX6")

    p = _fmt(metric(pro_arr))
    a = _fmt(metric(ama_arr))
    d = _fmt(p - a)

    return pd.DataFrame(
        [["6 L WRI/CHD X", p, a, d]],
        columns=["항목", "프로", "일반", "차이(프로-일반)"]
    )

# ── 26) 2/6 swing path ──────────────────────────────────────────────────────
def build_26_swing_path(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    2/6 swing path 항목:
      - 2/6 CHD X : CN6 - CN2
      - 2/6 CHD Z : CP6 - CP2
      - 2/6 L WRI X : AX6 - AX2
      - 2/6 L WRI Z : AZ6 - AZ2
    컬럼: 항목 / 프로 / 일반 / 차이(프로-일반) / 프로 방향 / 일반 방향
    """
    def diff(arr: np.ndarray, a: str, b: str) -> float:
        return g(arr, a) - g(arr, b)

    def _dir_x(v: float) -> str:
        if v < 0:
            return "X Bak/Fro"
        if v > 0:
            return "Fro/Bak"
        return ""

    def _dir_z(v: float) -> str:
        if v < 0:
            return "Z OUT"
        if v > 0:
            return "Z IN"
        return ""

    items = [
        ("2/6 CHD X",  ("CN6", "CN2"), _dir_x),  # 1
        ("2/6 CHD Z",  ("CP6", "CP2"), _dir_z),  # 2
        ("2/6 L WRI X",("AX6", "AX2"), _dir_x),  # 3
        ("2/6 L WRI Z",("AZ6", "AZ2"), _dir_z),  # 4
    ]

    rows = []
    for label, (c1, c2), dir_fn in items:
        p_raw = diff(pro_arr, c1, c2)
        a_raw = diff(ama_arr, c1, c2)
        d_raw = p_raw - a_raw

        p = _fmt(p_raw)
        a = _fmt(a_raw)
        d = _fmt(d_raw)

        rows.append([
            label, p, a, d,
            dir_fn(p_raw),  # 프로 방향
            dir_fn(a_raw),  # 일반 방향
        ])

    return pd.DataFrame(
        rows,
        columns=["항목", "프로", "일반", "차이(프로-일반)", "프로 스타일", "일반 스타일"]
    )
