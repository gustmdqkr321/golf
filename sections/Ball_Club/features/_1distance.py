# sections/gs/features/_gs.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# 고정 바이어스(파일 형식이 항상 일정할 때 여기 숫자만 바꾸면 됨)
#   - 현재 증상: 의도한 셀보다 '5행 아래'가 읽힘 → 행 오프셋을 -5로
#   - 열 오프셋이 필요하면 GS_COL_OFFSET 값을 조정
#   - 음수 가능. 인덱스는 0 미만으로 떨어지지 않도록 0으로 클램프됨.
# ──────────────────────────────────────────────────────────────────────
GS_ROW_OFFSET = -3
GS_COL_OFFSET = 0

# (옵션) 런타임에서 오프셋 바꾸고 싶을 때 호출하세요.
def set_gs_offset(row_offset: int = 0, col_offset: int = 0) -> None:
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
    s = str(x)
    s = s.replace(",", "").replace('"', '').replace("'", "").strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

# ──────────────────────────────────────────────────────────────────────
# GS CSV 셀 읽기(고정 바이어스만 적용)
#   - Excel 주소(A1 등)를 DataFrame 인덱스에 매핑할 때
#     (row + GS_ROW_OFFSET, col + GS_COL_OFFSET)로 바로 접근
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

# ──────────────────────────────────────────────────────────────────────
# 혼합 비교표(프로/일반) — 고정 바이어스 기반
#   - 인자 4개: (gs_pro_df, gs_ama_df, base_pro_arr, base_ama_arr)
# ──────────────────────────────────────────────────────────────────────
def build_gs_mixed_compare(
    gs_pro_df: pd.DataFrame,
    gs_ama_df: pd.DataFrame,
    base_pro_arr: np.ndarray,
    base_ama_arr: np.ndarray,
) -> pd.DataFrame:

    # GS(CSV)에서 직접 읽는 항목들
    gs_items: list[tuple[str, str]] = [
        ("H13", "H13(GS)"),
        ("K3",  "K3(GS)"),
        ("B17", "B17(GS)"),
        ("B42", "B42(GS)"),
        ("B19", "B19(GS)"),
        ("B43", "B43(GS)"),
        ("B13", "B13(GS)"),
        ("B44", "B44(GS)"),
        ("B45", "B45(GS)"),
        ("H14", "H14(GS)"),
        ("H18", "H18(GS)"),
        ("B24", "B24(GS)"),
        ("B10", "B10(GS)"),
        ("B12", "B12(GS)"),
        ("B14", "B14(GS)"),
        ("B57", "B57(GS)"),
        ("B60", "B60(GS)"),
        ("B61", "B61(GS)"),
        ("B62", "B62(GS)"),
    ]

    rows: list[list] = []
    for addr, label in gs_items:
        p = g_gs(gs_pro_df, addr)
        a = g_gs(gs_ama_df, addr)
        rows.append([label, addr, p, a, p - a])

    # 무지개(기존)에서 계산하는 항목들
    for label, expr in [
        ("무지개 CP1-AZ1", "CP1 - AZ1"),
        ("무지개 CP7-AZ7", "CP7 - AZ7"),
    ]:
        try: pp = eval_expr_base(base_pro_arr, expr)
        except Exception: pp = float("nan")
        try: aa = eval_expr_base(base_ama_arr, expr)
        except Exception: aa = float("nan")
        rows.append([label, expr, pp, aa, pp - aa])

    return pd.DataFrame(rows, columns=["항목", "셀/식", "프로", "일반", "차이(프로-일반)"])

# ──────────────────────────────────────────────────────────────────────
# 디버그: 현재 바이어스로 몇 개 주소만 빠르게 확인
# ──────────────────────────────────────────────────────────────────────
def debug_probe(gs_df: pd.DataFrame, addrs=("H13","K3","B17")) -> pd.DataFrame:
    rows=[]
    for a in addrs:
        v = g_gs(gs_df, a)
        rows.append([a, (GS_ROW_OFFSET, GS_COL_OFFSET), v])
    return pd.DataFrame(rows, columns=["주소", "GS_OFFSET(row,col)", "값"])
