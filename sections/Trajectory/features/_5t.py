from __future__ import annotations
import re, math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 공통 유틸 (A1 → 값 / 간단 수식 평가)
# ─────────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _addr_to_rc(addr: str) -> tuple[int, int]:
    m = _CELL.match(addr.strip())
    if not m:
        raise ValueError(f"잘못된 셀 주소: {addr}")
    c = _col_idx(m.group(1))
    r = int(m.group(2)) - 1
    return r, c

def g_base(arr: np.ndarray, addr: str) -> float:
    r, c = _addr_to_rc(addr)
    try:
        return float(arr[r, c])
    except Exception:
        return float('nan')

def eval_expr_base(arr: np.ndarray, expr: str) -> float:
    def repl(m): return str(g_base(arr, m.group(0)))
    safe = re.sub(r'[A-Za-z]+\d+', repl, expr.replace(' ', ''))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

# ─────────────────────────────────────────────────────────────
# 직각삼각형 각도 (∠BAC) = atan2(|ΔY|, |ΔX|) [deg]
# ─────────────────────────────────────────────────────────────
def _rt_angle(arr: np.ndarray, dx_expr: str, dy_expr: str) -> float:
    try:
        dx = abs(eval_expr_base(arr, dx_expr))  # |ΔX|
        dy = abs(eval_expr_base(arr, dy_expr))  # |ΔY|
        if (dx == 0 and dy == 0) or np.isnan(dx) or np.isnan(dy):
            return float('nan')
        return float(math.degrees(math.atan2(dy, dx)))
    except Exception:
        return float('nan')

# ── 각도 정의 ───────────────────────────────────────────────
def _l_arm_angle(arr: np.ndarray) -> float:
    return _rt_angle(arr, "AX4-AL4", "AY4-AM4")

def _both_sho_angle(arr: np.ndarray) -> float:
    return _rt_angle(arr, "BA4-AL4", "BB4-AM4")

def _sho_l_arm_angle(arr: np.ndarray) -> float:
    a1 = _l_arm_angle(arr); a2 = _both_sho_angle(arr)
    if np.isnan(a1) or np.isnan(a2):
        return float("nan")
    return a1 - a2

def _rsho_larm_y(arr: np.ndarray) -> float:
    return eval_expr_base(arr, "AY4 - BB4")

# ── 표 생성 ────────────────────────────────────────────────
_ITEMS = [
    ("Angles", "L Arm Ang (∠BAC; AB=AX4-AL4, BC=AY4-AM4)",      "SPECIAL", "LARM"),
    ("Angles", "Both Sho Ang (∠BAC; AB=BA4-AL4, BC=BB4-AM4)",   "SPECIAL", "BOTHSHO"),
    ("Angles", "Sho−L Arm Ang = L Arm − Both Sho",              "SPECIAL", "DIFF"),
    ("Angles", "R Sho / L Arm Y",                               "BASE",    "AY4 - BB4"),
]

def build_arm_shoulder_angle_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """
    Trajectory - Arm/Shoulder Angles (Frame 4)
    """
    p_l = _l_arm_angle(base_pro);  a_l = _l_arm_angle(base_ama)
    p_b = _both_sho_angle(base_pro); a_b = _both_sho_angle(base_ama)
    p_d = _sho_l_arm_angle(base_pro); a_d = _sho_l_arm_angle(base_ama)
    p_y = _rsho_larm_y(base_pro); a_y = _rsho_larm_y(base_ama)

    rows = [
        ("Angles", "L Arm Ang (∠BAC; AB=AX4-AL4, BC=AY4-AM4)",      "atan2(|AY4-AM4|, |AX4-AL4|)", p_l, a_l, p_l-a_l),
        ("Angles", "Both Sho Ang (∠BAC; AB=BA4-AL4, BC=BB4-AM4)",   "atan2(|BB4-AM4|, |BA4-AL4|)", p_b, a_b, p_b-a_b),
        ("Angles", "Sho−L Arm Ang = L Arm − Both Sho",              "diff(two angles)",            p_d, a_d, p_d-a_d),
        ("Angles", "R Sho / L Arm Y",                               "AY4 - BB4",                   p_y, a_y, p_y-a_y),
    ]

    df = pd.DataFrame(rows, columns=["분류","검사명","셀/식","프로","일반","차이(프로-일반)"])
    for c in ["프로","일반","차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
