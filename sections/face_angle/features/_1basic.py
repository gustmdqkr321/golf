# sections/face_angle/features/_1basic.py
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

# ── 특수 표기: 'GQ(W3)' → W3의 행번호(3)를 써서 'GQ3'로 해석 ────────────────
_paren = re.compile(r'^([A-Za-z]+)\(([A-Za-z]+)(\d+)\)$')
def _resolve_gq_paren(token: str) -> str | None:
    m = _paren.match(token.strip())
    if not m:
        return None
    col_out = m.group(1)         # 'GQ'
    # 내부 주소의 행번호만 사용 (W3 → 3)
    row_num = m.group(3)
    return f"{col_out}{row_num}"

# ── 표 생성 ────────────────────────────────────────────────────────────────
def build_face_angle_table(
    gs_pro_df: pd.DataFrame,
    gs_ama_df: pd.DataFrame,
    base_pro_arr: np.ndarray,
    base_ama_arr: np.ndarray,
) -> pd.DataFrame:
    """
    이미지에 나온 항목들을 그대로 한 표에 정리.
    columns = ["항목", "셀/식", "프로", "일반", "차이(프로-일반)"]
    """
    items: list[tuple[str, str, str]] = [
        ("B11(GS)",                "B11",                               "GS"),
        ("B38(GS)",                "B38",                               "GS"),
        ("CP1-CS1",                "CP1 - CS1",                         "BASE"),
        ("B36(GS)",                "B36",                               "GS"),
        ("CP7-CS7",                "CP7 - CS7",                         "BASE"),
        ("B39(GS)",                "B39",                               "GS"),
        ("(CP7-CS7)-(CP1-CS1)",    "(CP7 - CS7) - (CP1 - CS1)",         "BASE"),
        ("B32(GS)",                "B32",                               "GS"),
        ("B33(GS)",                "B33",                               "GS"),
        ("B31(GS)",                "B31",                               "GS"),
        ("AY1-BN1",                "AY1 - BN1",                         "BASE"),
        ("AY7-BN7",                "AY7 - BN7",                         "BASE"),
        ("무지개 CP7-AZ7",          "CP7 - AZ7",                         "BASE"),
    ]

    rows: list[list] = []
    for label, token, src in items:
        if src == "GS":
            p = g_gs(gs_pro_df, token)
            a = g_gs(gs_ama_df, token)
        elif src == "BASE":
            p = eval_expr_base(base_pro_arr, token)
            a = eval_expr_base(base_ama_arr, token)
        elif src == "BASE_GQ":
            # 'GQ(W3)' 같은 표기를 'GQ3'로 바꿔 접근
            addr = _resolve_gq_paren(token) or token
            p = g_base(base_pro_arr, addr)
            a = g_base(base_ama_arr, addr)
        else:
            p = a = float("nan")

        rows.append([label, token, p, a, p - a])

    df = pd.DataFrame(rows, columns=["항목", "셀/식", "프로", "일반", "차이(프로-일반)"])
    # 숫자 포맷 안정화
    for c in ("프로", "일반", "차이(프로-일반)"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
