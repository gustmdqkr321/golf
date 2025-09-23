from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── (공용) A1 주소 → 값 ────────────────────────────────────────────────────
_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def _to_float(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return float("nan")

def g_base(arr: np.ndarray, addr: str) -> float:
    m = _CELL.match(addr.strip())
    if not m:
        return float("nan")
    r = int(m.group(2)) - 1
    c = _col_idx(m.group(1))
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def eval_expr_base(arr: np.ndarray, expr: str) -> float:
    """무지개 배열에 대해 'CP7 - CS7' 같은 간단식 평가"""
    def repl(m: re.Match) -> str:
        return str(g_base(arr, m.group(0)))
    safe = re.sub(r'[A-Za-z]+\d+', repl, expr.replace(" ", ""))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        return float("nan")
    try:
        return float(eval(safe, {"__builtins__": None}, {}))
    except Exception:
        return float("nan")

# ── (GS) 주소 → 값  (오프셋은 기존에 쓰던 규칙 유지) ─────────────────────────
# 필요시 프로젝트에서 쓰는 상수와 동일하게 맞춰주세요.
GS_ROW_OFFSET = -3
GS_COL_OFFSET = 0

def g_gs(gs_df: pd.DataFrame, addr: str) -> float:
    m = _CELL.match(addr.strip())
    if not m:
        return float("nan")
    r = max(0, int(m.group(2)) - 1 + GS_ROW_OFFSET)
    c = max(0, _col_idx(m.group(1)) + GS_COL_OFFSET)
    try:
        return _to_float(gs_df.iat[r, c])
    except Exception:
        return float("nan")

# 항목 정의
# (분류, 검사명, 소스, 식/셀)
_ITEMS: list[tuple[str, str, str, str]] = [
    # Club / Ball
    ("Club", "Face Ang",   "BASE", "CP7 - CS7"),
    ("Club", "Club Path",  "GS",   "B9"),
    ("Club", "Face to Path","GS",  "B11"),
    # ("Club", "Ball Flight Shape", "GQ", "skip"),
    ("Club", "Club Speed", "GS",   "B17"),
    ("Club", "Club Loft",  "GS",   "B24"),
    ("Club", "Spin Loft",  "GS",   "B12"),

    # ADD / WRI
    ("ADD/WRI", "L X", "BASE", "AX7 - AX1"),
    ("ADD/WRI", "L Y", "BASE", "AY7 - AY1"),
    ("ADD/WRI", "L Z", "BASE", "AZ7 - AZ1"),
    ("ADD/WRI", "ADD/Grip", "BASE", "(BN7 - AY7) - (BN1 - AY1)"),

    # ADD / ELB
    ("ADD/ELB", "L X", "BASE", "AR7 - AR1"),
    ("ADD/ELB", "L Y", "BASE", "AS7 - AS1"),
    ("ADD/ELB", "L Z", "BASE", "AT7 - AT1"),
    ("ADD/ELB", "L/R X", "BASE", "BG7 - AR7"),

    # ADD / SHO
    ("ADD/SHO", "L/R X ADD", "BASE", "BA1 - AL1"),
    ("ADD/SHO", "L/R X IMP", "BASE", "BA7 - AL7"),
    ("ADD/SHO", "L/R Y ADD", "BASE", "BB1 - AM1"),
    ("ADD/SHO", "L/R Y IMP", "BASE", "BB7 - AM7"),
    ("ADD/SHO", "BOT Z", "BASE", "(AN7+BC7)/2 - (AN1+BC7)/2"),
    ("ADD/SHO", "Turn(deg)", "GS", "E18"),

    # ADD / WAI
    ("ADD/WAI", "L X", "BASE", "H7 - H1"),
    ("ADD/WAI", "L Y", "BASE", "I7 - I1"),
    ("ADD/WAI", "L Z", "BASE", "J7 - J1"),
    ("ADD/WAI", "R X", "BASE", "K7 - K1"),
    ("ADD/WAI", "R Z", "BASE", "M7 - M1"),
    ("ADD/WAI", "BOT Z", "BASE", "(J7+M7)/2 - (J1+M1)/2"),
    ("ADD/WAI", "Turn(deg)", "GS", "E19"),

    # ADD / WAI-SHO
    ("ADD/WAI-SHO", "WAI/SHO Z DIFF", "BASE", "(BB7 - AM7) - (M7 - M1)"),

    # ADD / KNE
    ("ADD/KNE", "L X", "BASE", "BP7 - BP1"),
    ("ADD/KNE", "L Y", "BASE", "BQ7 - BQ1"),
    ("ADD/KNE", "L Z", "BASE", "BR7 - BR1"),

    # HED
    ("HED", "X ✔", "BASE", "AC7 - AC1"),
    ("HED", "Y",   "BASE", "AD7 - AD1"),
    ("HED", "Z",   "BASE", "AE7 - AE1"),
]


def build_summary_phase_table(gs_pro: pd.DataFrame, gs_ama: pd.DataFrame,
                              base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """
    2.1.9 Summary Phase Table (Face Ang ~ HED)
    반환: ['분류','검사명','셀/식','프로','일반','차이(프로-일반)']
    """
    rows = []
    for group, label, src, ref in _ITEMS:
        if src == "GS":
            p = g_gs(gs_pro, ref); a = g_gs(gs_ama, ref)
        elif src == "BASE":
            p = eval_expr_base(base_pro, ref); a = eval_expr_base(base_ama, ref)
        else:  # skip
            p = a = float("nan")
        rows.append([group, label, ref, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류", "검사명", "셀/식", "프로", "일반", "차이(프로-일반)"])
    for c in ["프로", "일반", "차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
