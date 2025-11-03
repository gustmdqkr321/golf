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

# ── 벡터/각도 유틸 ─────────────────────────────
def _vec(arr: np.ndarray, addrs: tuple[str,str,str]) -> np.ndarray:
    """주어진 (col,row) 주소 3개 → np.array([x,y,z])"""
    return np.array([eval_expr_base(arr, a) for a in addrs], dtype=float)

def _angle_abc(arr: np.ndarray) -> float:
    A = np.array([eval_expr_base(arr, "CN6"),
                  eval_expr_base(arr, "CO6"),
                  eval_expr_base(arr, "CP6")])
    B = np.array([eval_expr_base(arr, "CN7"),
                  eval_expr_base(arr, "CO7"),
                  eval_expr_base(arr, "CP7")])
    C = np.array([eval_expr_base(arr, "CN8"),
                  eval_expr_base(arr, "CO8"),
                  eval_expr_base(arr, "CP8")])
    BA, BC = A - B, C - B
    cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


# ── 항목 정의 ─────────────────────────────
_ITEMS = [
    ("L WRI/CHD", "6 L WRI/CHD Y", "BASE", "CO6 - AY6"),
    ("L WRI/CHD", "8 L WRI/CHD Y", "BASE", "CO8 - AY8"),
    ("L WRI/CHD", "6/7/8 Ang",     "SPECIAL", "angle"),
]


def build_wri_chd_angle_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """
    Trajectory - 3rd Feature: L WRI/CHD Y (6,8) & 6/7/8 Angle
    """
    rows = []
    for group, label, src, ref in _ITEMS:
        if src == "BASE":
            p = eval_expr_base(base_pro, ref); a = eval_expr_base(base_ama, ref)
        elif src == "SPECIAL" and ref == "angle":
            p = _angle_abc(base_pro); a = _angle_abc(base_ama)
        else:
            p = a = float("nan")
        rows.append([group, label, p, a, p - a])

    df = pd.DataFrame(rows, columns=["분류","검사명","프로","일반","차이(프로-일반)"])
    for c in ["프로","일반","차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
