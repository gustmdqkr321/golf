# sections/club_path/features/_gs_club.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# 고정 바이어스(파일 형식이 항상 일정할 때 여기 숫자만 바꾸면 됨)
# ──────────────────────────────────────────────────────────────────────
GS_ROW_OFFSET = -3
GS_COL_OFFSET = 0

def set_gs_offset(row_offset: int = 0, col_offset: int = 0) -> None:
    """런타임에서 GS 오프셋을 바꾸고 싶을 때 호출"""
    global GS_ROW_OFFSET, GS_COL_OFFSET
    GS_ROW_OFFSET = int(row_offset)
    GS_COL_OFFSET = int(col_offset)

# ──────────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _addr_to_rc(addr: str) -> tuple[int, int]:
    m = _CELL.match(addr.strip())
    if not m:
        raise ValueError(f"잘못된 셀 주소: {addr}")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    return row, col

def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    s = str(x).replace(",", "").replace('"', "").replace("'", "").strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

# ──────────────────────────────────────────────────────────────────────
# GS CSV 셀 읽기(고정 바이어스만 적용)
# ──────────────────────────────────────────────────────────────────────
def g_gs(gs_df: pd.DataFrame, addr: str) -> float:
    r, c = _addr_to_rc(addr)        # A1 → (0,0) 기준 좌표
    rr = max(0, r + GS_ROW_OFFSET)  # 음수 방지
    cc = max(0, c + GS_COL_OFFSET)
    try:
        return _to_float(gs_df.iat[rr, cc])
    except Exception:
        return float("nan")

# ──────────────────────────────────────────────────────────────────────
# 무지개(기존 배열) 식 평가
# ──────────────────────────────────────────────────────────────────────
def g_base(arr: np.ndarray, addr: str) -> float:
    r, c = _addr_to_rc(addr)
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def eval_expr_base(arr: np.ndarray, expr: str) -> float:
    def repl(m):
        return str(g_base(arr, m.group(0)))
    safe = re.sub(r'[A-Za-z]+\d+', repl, expr.replace(" ", ""))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))


# 항목 정의: (분류, 검사명, 소스, 참조/식 또는 특수 사양)
# 소스: "GS" | "BASE" | "SPECIAL"
_ITEMS: list[tuple[str, str, str, object]] = [
    # Loft
    ("Loft", "Add Loft",        "GS",     "B26"),
    ("Loft", "Impact Loft",     "GS",     "B24"),
    ("Loft", "Add-Imp Diff",    "GS",     "B27"),
    ("Loft", "Spin Loft",       "GS",     "B12"),

    # Club Lean (GS)
    ("Club Lean", "Club Lean(Add)",        "GS",     "B60"),
    ("Club Lean", "Club Lean(Imp)",        "GS",     "B61"),
    ("Club Lean", "Club Lean(diff)",       "GS",     "B62"),

    # L WRI/CHD (무지개/베이직)
    ("L WRI/CHD", "L WRI/CHD(add)",        "BASE",   "CP1 - AZ1"),
    ("L WRI/CHD", "L WRI/CHD(imp)",        "BASE",   "CP7 - AZ7"),
    ("L WRI/CHD", "L WRI/CHD(diff)",       "BASE",   "(CP1 - AZ1) - (CP7 - AZ7)"),

    # Lateral Tilt (Impact)
    ("Lateral Tilt(imp)", "Lateral Tilt(imp) Hip",   "GS",  "E61"),
    ("Lateral Tilt(imp)", "Lateral Tilt(imp) Torso", "GS",  "E62"),

    # Low Point / Up-Down Path
    ("Low Point",        "Low Point", "GS", "B14"),
    ("Up / Down Path",   "Up / Down Path", "GS", "B57"),

    # Body Side Bend (Impact)
    ("Body Side Bend (Impact)", "Body side bend Pelvis(deg)",   "GS", "E22"),
    ("Body Side Bend (Impact)", "Body side bend Ribcage(deg)",  "GS", "E23"),

    # L/R SHO DIF (무지개)
    ("L/R SHO DIF", "L/R SHO DIF ADD",      "BASE",   "AM1 - BB1"),
    ("L/R SHO DIF", "L/R SHO DIF IMP",      "BASE",   "AM7 - BB7"),

    # Stance with / Club Position  = |CA1-CP1| / |CA1-CM1|
    ("Stance with/Club Position", "Stance with/Club Position",
                                  "SPECIAL",
                                  {"type": "abs_ratio",
                                   "num": ("CA1", "CP1"),
                                   "den": ("CA1", "CM1")}),

    # Club Speed at Impact
    ("Club", "Club Speed imp",  "BASE",     "DL7"),
]


def _abs_diff(arr: np.ndarray, a1: str, a2: str) -> float:
    """|A1 - A2| (베이직)"""
    return abs(eval_expr_base(arr, f"{a1} - {a2}"))

def _abs_ratio(arr: np.ndarray, num: tuple[str, str], den: tuple[str, str]) -> float:
    """|x1-x2| / |y1-y2|"""
    n = _abs_diff(arr, num[0], num[1])
    d = _abs_diff(arr, den[0], den[1])
    try:
        return float(n / d) if d not in (0.0, -0.0) else float("nan")
    except Exception:
        return float("nan")


def build_trajectory_table(gs_pro: pd.DataFrame, gs_ama: pd.DataFrame,
                           base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """
    Trajectory 섹션 - 1st Feature (Loft / Club Lean / Tilt / Low Point / etc.)
    반환: ['분류','검사명','셀/식','프로','일반','차이(프로-일반)']
    """
    rows: list[list] = []

    for group, label, src, ref in _ITEMS:
        if src == "GS":
            p = g_gs(gs_pro, str(ref)); a = g_gs(gs_ama, str(ref))
        elif src == "BASE":
            p = eval_expr_base(base_pro, str(ref)); a = eval_expr_base(base_ama, str(ref))
        elif src == "SPECIAL":
            if isinstance(ref, dict) and ref.get("type") == "abs_ratio":
                p = _abs_ratio(base_pro, ref["num"], ref["den"])
                a = _abs_ratio(base_ama, ref["num"], ref["den"])
            else:
                p = a = float("nan")
        else:
            p = a = float("nan")

        rows.append([label,
                    
                     p, a, p - a])

    df = pd.DataFrame(rows, columns=["검사명", "프로", "일반", "차이(프로-일반)"])

    # 숫자형 보장
    for c in ["프로", "일반", "차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
