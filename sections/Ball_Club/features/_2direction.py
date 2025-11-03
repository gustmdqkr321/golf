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
# sections/gs/features/_gs.py  (기존 함수에 추가/수정)
# ... (위쪽 코드 동일) ...

def _classify_ball_flight_shape(cp: float, f2p: float) -> str:
    if cp >= 4:
        if f2p >= 5:  return "Push Slice"
        if f2p <= -5: return "Push Hook"
        if f2p <= cp: return "Push"
    if cp <= -4:
        if f2p >= 5:  return "Pull Slice"
        if f2p <= -5: return "Pull Hook"
        if f2p <= cp: return "Pull"
    if -4 <= cp <= -3 and f2p <= cp: return "Draw"
    if  3 <= cp <=  4 and f2p <= cp: return "Fade"
    if -4 <= cp <=  0 and f2p <= -5: return "Hook"
    if  0 <= cp <=  4 and f2p >=  5: return "Slice"
    if abs(cp) <= 2 and f2p <= cp:   return "Straight"
    return "Straight"

def build_gs_club_table(gs_pro_df, gs_ama_df, base_pro_arr, base_ama_arr, origin_pro=None, origin_ama=None):
    items = [
        ("Face Angle (imp, deg)", "B36", "GS"),
        ("Face Angle (imp, cm)", "CP7 - CS7", "BASE"),
        ("Face Angle (add, deg)", "B38", "GS"),
        ("Face Angle (add, cm)", "CP1 - CS1", "BASE"),
        ("Face Angle (add/imp diff, deg)", "B39", "GS"),
        ("Club Path (deg)", "B9", "GS"),
        ("Face to Path (deg)", "B11", "GS"),
        ("In/Out Path", "B56", "GS"),
        ("Lie (deg)", "B31", "GS"),
    ]

    rows = []
    cp_pro = cp_ama = f2p_pro = f2p_ama = float("nan")

    for label, expr, src in items:
        if src == "GS":
            p = g_gs(gs_pro_df, expr)
            a = g_gs(gs_ama_df, expr)
        else:
            try:    p = eval_expr_base(base_pro_arr, expr)
            except: p = float("nan")
            try:    a = eval_expr_base(base_ama_arr, expr)
            except: a = float("nan")

        rows.append([label, p, a, (p - a), "", ""])  # 분류 칼럼 2개 추가

        if expr == "B9":  cp_pro,  cp_ama  = p, a
        if expr == "B11": f2p_pro, f2p_ama = p, a

    # 5) Ball Flight Shape — 숫자 칼럼은 NaN, 분류 칼럼에 텍스트
    shape_pro = _classify_ball_flight_shape(cp_pro,  f2p_pro)
    shape_ama = _classify_ball_flight_shape(cp_ama,  f2p_ama)
    import numpy as np
    rows.append(["Ball Flight Shape", np.nan, np.nan, np.nan, shape_pro, shape_ama])

    return pd.DataFrame(
        rows,
        columns=["항목","프로","일반","차이(프로-일반)","프로 분류","일반 분류"]
    )
