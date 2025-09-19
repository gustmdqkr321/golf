# sections/swing_plane/features/_sp.py
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# 공통 유틸: A1 주소 → 값 / 식 평가
# ─────────────────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

_CELL = re.compile(r"[A-Za-z]+[0-9]+")

def g(arr: np.ndarray, code: str) -> float:
    letters = "".join(filter(str.isalpha, code))
    num     = int("".join(filter(str.isdigit, code)))
    return float(arr[num - 1, _col_idx(letters)])

def eval_expr(arr: np.ndarray, expr: str) -> float:
    def repl(m: re.Match) -> str:
        return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr.replace(" ", ""))
    if not re.fullmatch(r"[-+*/().0-9]+", safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

# ─────────────────────────────────────────────────────────────────────
# 1) 선택 차이 8개 (BC/BM/BO/CN/CO/CP & 보조 2개)
# ─────────────────────────────────────────────────────────────────────
def build_selected_diffs_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    items = [
        ("BC4 − BC1", "BC4 - BC1"),
        ("BM6 − BM2", "BM6 - BM2"),
        ("BO6 − BO2", "BO6 - BO2"),
        ("CN6 − CN2", "CN6 - CN2"),
        ("CO6 − CO2", "CO6 - CO2"),
        ("CP6 − CP2", "CP6 - CP2"),
        ("CN2 − BM2", "CN2 - BM2"),
        ("CN6 − BM6", "CN6 - BM6"),
    ]
    rows = []
    for label, expr in items:
        try: p = eval_expr(pro_arr, expr)
        except Exception: p = float("nan")
        try: a = eval_expr(ama_arr, expr)
        except Exception: a = float("nan")
        rows.append([label, expr, p, a, p - a])
    return pd.DataFrame(rows, columns=["항목", "식", "프로", "일반", "차이(프로-일반)"])

# ─────────────────────────────────────────────────────────────────────
# 2) Frame1 직각 △BAC( AB=AX1−CN1 , BC=AY1 )
# ─────────────────────────────────────────────────────────────────────
def _angle_from_AB_BC(AB: float, BC: float) -> float:
    return math.degrees(math.atan2(abs(BC), abs(AB)))  # deg

def build_bac_ax_cn_ay_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    def calc(arr: np.ndarray) -> float:
        AB = g(arr, "AX1") - g(arr, "CN1")
        BC = g(arr, "AY1")
        return _angle_from_AB_BC(AB, BC)

    try: p = calc(pro_arr)
    except Exception: p = float("nan")
    try: a = calc(ama_arr)
    except Exception: a = float("nan")
    rows = [["∠BAC (AB=AX1−CN1, BC=AY1)", "atan2(|AY1|, |AX1−CN1|)", p, a, p - a]]
    return pd.DataFrame(rows, columns=["항목", "정의/식", "프로(°)", "일반(°)", "차이(프로-일반)"])

# ─────────────────────────────────────────────────────────────────────
# 3) case1…case11 ∠BAC(°) + case6/7 판정(GOOD/BAD)
#    - 각 case의 AB,BC 정의는 질문에서 준 스펙 그대로
# ─────────────────────────────────────────────────────────────────────
_CASE_DEF = {
    1:  (("AX1",  "CN1"),   ("AY1",  None)),
    2:  (("K1",   "CN1"),   ("L1",   None)),
    3:  (("BA1",  "CN1"),   ("BB1",  None)),
    4:  (("CN3",  "BM2"),   ("CO3",  "BN2")),
    5:  (("CN4",  "BM4"),   ("CO4",  "BN4")),
    6:  (("CN5",  "BM5"),   ("CO6",  "BN5")),
    7:  (("CN6",  "BM6"),   ("CO6",  "BN6")),
    8:  (("BG6",  "BM6"),   ("BH6",  "BN6")),
    9:  (("BM7",  "CN7"),   ("BN7",  None)),
    10: (("CN8",  "BM8"),   ("CO8",  "BN8")),
    11: (("CN10", "BM10"),  ("CO10", "BN10")),
}

def _case_angle(arr: np.ndarray, case_idx: int) -> float:
    (a0, a1), (b0, b1) = _CASE_DEF[case_idx]
    AB = abs(g(arr, a0) - g(arr, a1))
    BC = abs(g(arr, b0) - g(arr, b1)) if b1 else abs(g(arr, b0))
    return _angle_from_AB_BC(AB, BC)

def build_bac_cases_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    # 각 케이스 각도
    pro_vals, ama_vals = {}, {}
    for i in range(1, 12):
        try: pro_vals[i] = _case_angle(pro_arr, i)
        except Exception: pro_vals[i] = float("nan")
        try: ama_vals[i] = _case_angle(ama_arr, i)
        except Exception: ama_vals[i] = float("nan")

    # 판정: 각 데이터셋 내에서 case2, case3의 범위 사이면 GOOD
    def judge(vals: dict[int, float], idx: int) -> str:
        c2, c3 = vals.get(2, float("nan")), vals.get(3, float("nan"))
        lo, hi = (min(c2, c3), max(c2, c3))
        v = vals.get(idx, float("nan"))
        return "GOOD" if (not math.isnan(v) and not math.isnan(lo) and lo <= v <= hi) else "BAD"

    jp6, ja6 = judge(pro_vals, 6), judge(ama_vals, 6)
    jp7, ja7 = judge(pro_vals, 7), judge(ama_vals, 7)

    # 테이블 구성
    rows = []
    for i in range(1, 12):
        note = ""
        if i == 6: note = f"P:{jp6} / A:{ja6}"
        if i == 7: note = f"P:{jp7} / A:{ja7}"
        rows.append([f"case{i}", "∠BAC", pro_vals[i], ama_vals[i], pro_vals[i]-ama_vals[i], note])

    return pd.DataFrame(
        rows, columns=["항목", "정의", "프로(°)", "일반(°)", "차이(프로-일반)", "판정(메모)"]
    )
