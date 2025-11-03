# sections/<원하는섹션>/features/_ab_midpoints.py
from __future__ import annotations
import re
import math
import numpy as np
import pandas as pd

_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    """엑셀 주소(A1 등) → 배열 값 (float, 실패 시 NaN)"""
    m = _CELL.match(code.strip())
    if not m:
        return float("nan")
    r = int(m.group(2)) - 1
    c = _col_idx(m.group(1))
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def build_ab_midpoints_table(
    arr: np.ndarray, start: int = 1, end: int = 10
) -> pd.DataFrame:
    """
    프레임 start~end:
      A = ((ALn+BAn)/2, (AMn+BBn)/2, (ANn+BCn)/2)
      B = ((AXn+BMn)/2, (AYn+BNn)/2, (AZn+BOn)/2)
      |AB| = sqrt((Bx-Ax)^2 + (By-Ay)^2 + (Bz-Az)^2)
    반환 컬럼: [Frame, Ax, Ay, Az, Bx, By, Bz, |AB|]
    """
    rows: list[list] = []
    for n in range(start, end + 1):
        Ax = (g(arr, f"AL{n}") + g(arr, f"BA{n}"))/2.0
        Ay = (g(arr, f"AM{n}") + g(arr, f"BB{n}"))/2.0
        Az = (g(arr, f"AN{n}") + g(arr, f"BC{n}"))/2.0

        Bx = (g(arr, f"AX{n}") + g(arr, f"BM{n}"))/2.0
        By = (g(arr, f"AY{n}") + g(arr, f"BN{n}"))/2.0
        Bz = (g(arr, f"AZ{n}") + g(arr, f"BO{n}"))/2.0

        dist = math.sqrt((Bx-Ax)**2 + (By-Ay)**2 + (Bz-Az)**2)
        rows.append([n, Ax, Ay, Az, Bx, By, Bz, dist])

    return pd.DataFrame(rows, columns=["Frame", "Ax", "Ay", "Az", "Bx", "By", "Bz", "|AB|"])

def build_ab_distance_compare(
    pro_arr: np.ndarray, ama_arr: np.ndarray, start: int = 1, end: int = 10
) -> pd.DataFrame:
    def _dists(arr):
        out=[]
        for n in range(start, end+1):
            Ax = (g(arr, f"AL{n}") + g(arr, f"BA{n}"))/2.0
            Ay = (g(arr, f"AM{n}") + g(arr, f"BB{n}"))/2.0
            Az = (g(arr, f"AN{n}") + g(arr, f"BC{n}"))/2.0
            Bx = (g(arr, f"AX{n}") + g(arr, f"BM{n}"))/2.0
            By = (g(arr, f"AY{n}") + g(arr, f"BN{n}"))/2.0
            Bz = (g(arr, f"AZ{n}") + g(arr, f"BO{n}"))/2.0
            out.append(math.sqrt((Bx-Ax)**2 + (By-Ay)**2 + (Bz-Az)**2))
        return out

    frames = list(range(start, end+1))
    p = _dists(pro_arr)
    a = _dists(ama_arr)

    df = pd.DataFrame(
        {"Frame": frames, "프로": p, "일반": a}
    )
    # 숫자형 보장 + 반올림(표시는 2자리, dtype은 float 유지)
    df["프로"]  = pd.to_numeric(df["프로"], errors="coerce").round(2)
    df["일반"]  = pd.to_numeric(df["일반"], errors="coerce").round(2)

    # (선택) 차이 컬럼이 필요하면 추가해도 하이라이트에는 영향 없음
    df["차이(프로-일반)"] = (df["프로"] - df["일반"]).round(2)

    return df
