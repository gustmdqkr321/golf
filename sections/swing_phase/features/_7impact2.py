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


_ITEMS: list[tuple[str, str, str]] = [
    ("Turn",      "Pelvis",   "E18"),
    ("Turn",      "Ribcage",  "E19"),
    ("Bend",      "Pelvis",   "E20"),
    ("Bend",      "Ribcage",  "E21"),
    ("Side Bend", "Pelvis",   "E22"),
    ("Side Bend", "Ribcage",  "E23"),
]
def build_turn_bend_table(gs_pro: pd.DataFrame, gs_ama: pd.DataFrame) -> pd.DataFrame:
    """
    2.1.8 Turn/Bend/Side Bend 표 (GS 값 기반)
    반환 컬럼: ['분류','검사명','셀','프로','일반','차이(프로-일반)']
    """
    rows: list[list] = []
    for group, label, addr in _ITEMS:
        p = g_gs(gs_pro, addr)
        a = g_gs(gs_ama, addr)
        rows.append([group, label, addr, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류", "검사명", "셀", "프로", "일반", "차이(프로-일반)"])

    # 숫자형 보장
    for c in ["프로", "일반", "차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
