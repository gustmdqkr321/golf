# sections/swing_error/features/_5PS.py
from __future__ import annotations
import numpy as np
import pandas as pd
import re

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def pelvis_x_center(arr: np.ndarray, row: int) -> float:
    # 골반 좌/우 X 평균: (Hrow + Krow)/2
    return (g(arr, f"H{row}") + g(arr, f"K{row}")) * 0.5
    
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


# ── 단일표(스냅샷) ───────────────────────────────────────────────────────────
def build_pelvis_shift_table(arr: np.ndarray) -> pd.DataFrame:
    v1 = pelvis_x_center(arr, 1)  # ① ADD
    v2 = pelvis_x_center(arr, 4)  # ② TOP
    v3 = pelvis_x_center(arr, 7)  # ③ IMP

    rows = [
        ["ADD", "① (H1 + K1)/2", v1],
        ["TOP", "② (H4 + K4)/2", v2],
        ["IMP", "③ (H7 + K7)/2", v3],
        ["ADD-TOP", "② − ①", v2 - v1],
        ["TOP-IMP", "③ − ②", v3 - v2],
        ["ADD-IMP", "③ − ①", v3 - v1],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

# ── 프로/일반 비교표(골반 X) ────────────────────────────────────────────────
def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    def triplet(arr):
        v1 = pelvis_x_center(arr, 1)
        v2 = pelvis_x_center(arr, 4)
        v3 = pelvis_x_center(arr, 7)
        return v1, v2, v3

    p1, p2, p3 = triplet(pro_arr)
    a1, a2, a3 = triplet(ama_arr)

    rows = [
        ["ADD",  p1, a1, p1 - a1],
        ["TOP",  p2, a2, p2 - a2],
        ["IMP",  p3, a3, p3 - a3],
        ["ADD-TOP",     p2 - p1,  a2 - a1,  (p2 - p1) - (a2 - a1)],
        ["TOP-IMP",     p3 - p2,  a3 - a2,  (p3 - p2) - (a3 - a2)],
        ["ADD-IMP",     p3 - p1,  a3 - a1,  (p3 - p1) - (a3 - a1)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

# (= shoulder X) ----------------------------------------------------------------
def shoulder_x_center(arr: np.ndarray, row: int) -> float:
    # 어깨 좌/우 X 평균: (ALrow + BArow)/2
    return (g(arr, f"AL{row}") + g(arr, f"BA{row}")) * 0.5

def build_shoulder_x_table(arr: np.ndarray,
                           gs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    v1 = shoulder_x_center(arr, 1)
    v2 = shoulder_x_center(arr, 4)
    v3 = shoulder_x_center(arr, 7)
    gs = g_gs(gs_df, "E13") if gs_df is not None else float("nan")  # GS(E13)

    rows = [
        ["ADD",      "① (AL1 + BA1)/2", v1],
        ["TOP",      "② (AL4 + BA4)/2", v2],
        ["IMP",      "③ (AL7 + BA7)/2", v3],
        ["GS",       "E13(GS)",         gs],
        ["ADD-TOP",  "② − ①",           v2 - v1],
        ["TOP-IMP",  "③ − ②",           v3 - v2],
        ["ADD-IMP",  "③ − ①",           v3 - v1],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

def build_shoulder_x_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                             gs_pro_df: pd.DataFrame | None = None,
                             gs_ama_df: pd.DataFrame | None = None) -> pd.DataFrame:
    p = build_shoulder_x_table(pro_arr, gs_pro_df)
    a = build_shoulder_x_table(ama_arr, gs_ama_df)
    df = p[["항목", "값"]].merge(a[["항목", "값"]], on="항목", suffixes=("·프로", "·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로": "프로", "값·일반": "일반"})

# (= head X) --------------------------------------------------------------------
def build_head_x_table(arr: np.ndarray) -> pd.DataFrame:
    v1 = g(arr, "AC1")
    v2 = g(arr, "AC4")
    v3 = g(arr, "AC7")
    rows = [
        ["ADD",     "① AC1", v1],
        ["TOP",     "② AC4", v2],
        ["IMP",     "③ AC7", v3],
        ["ADD-TOP", "② − ①", v2 - v1],
        ["TOP-IMP", "③ − ②", v3 - v2],
        ["ADD-IMP", "③ − ①", v3 - v1],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

def build_head_x_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_head_x_table(pro_arr)
    a = build_head_x_table(ama_arr)
    df = p[["항목", "값"]].merge(a[["항목", "값"]], on="항목", suffixes=("·프로", "·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로": "프로", "값·일반": "일반"})

# (= lift/down Y: waist) --------------------------------------------------------
def pelvis_y_center(arr: np.ndarray, row: int) -> float:
    # 골반 좌/우 Y 평균: (Irow + Lrow)/2
    return (g(arr, f"I{row}") + g(arr, f"L{row}")) * 0.5

def build_waist_lifty_table(arr: np.ndarray,
                            gs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    v_add = pelvis_y_center(arr, 1)
    v_top = pelvis_y_center(arr, 4)
    v_imp = pelvis_y_center(arr, 7)
    v_f1  = pelvis_y_center(arr, 8)
    gs = g_gs(gs_df, "E14") if gs_df is not None else float("nan")  # GS(E14; Pelvis IMP)

    rows = [
        ["ADD - TOP", "(I4 + L4)/2 − (I1 + L1)/2", v_top - v_add],
        ["TOP - IMP", "(I7 + L7)/2 − (I4 + L4)/2", v_imp - v_top],
        ["Pelvis(IMP)", "E14(GS)",                 gs],
        ["IMP - F1",  "(I8 + L8)/2 − (I7 + L7)/2", v_f1 - v_imp],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

def build_waist_lifty_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                              gs_pro_df: pd.DataFrame | None = None,
                              gs_ama_df: pd.DataFrame | None = None) -> pd.DataFrame:
    p = build_waist_lifty_table(pro_arr, gs_pro_df)
    a = build_waist_lifty_table(ama_arr, gs_ama_df)
    df = p[["항목", "값"]].merge(a[["항목", "값"]], on="항목", suffixes=("·프로", "·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로": "프로", "값·일반": "일반"})

# (= lift/down Y: shoulder) -----------------------------------------------------
def shoulder_y_center(arr: np.ndarray, row: int) -> float:
    # 어깨 좌/우 Y 평균: (AMrow + BBrow)/2
    return (g(arr, f"AM{row}") + g(arr, f"BB{row}")) * 0.5

def build_shoulder_lifty_table(arr: np.ndarray,
                               gs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    v_add = shoulder_y_center(arr, 1)
    v_top = shoulder_y_center(arr, 4)
    v_imp = shoulder_y_center(arr, 7)
    v_f1  = shoulder_y_center(arr, 8)
    gs = g_gs(gs_df, "E15") if gs_df is not None else float("nan")  # GS(E15; Ribcage IMP)

    rows = [
        ["ADD - TOP",   "(AM4 + BB4)/2 − (AM1 + BB1)/2", v_top - v_add],
        ["TOP - IMP",   "(AM7 + BB7)/2 − (AM4 + BB4)/2", v_imp - v_top],
        ["Ribcage(IMP)", "E15(GS)",                       gs],
        ["IMP - F1",    "(AM8 + BB8)/2 − (AM7 + BB7)/2", v_f1 - v_imp],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

def build_shoulder_lifty_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                                 gs_pro_df: pd.DataFrame | None = None,
                                 gs_ama_df: pd.DataFrame | None = None) -> pd.DataFrame:
    p = build_shoulder_lifty_table(pro_arr, gs_pro_df)
    a = build_shoulder_lifty_table(ama_arr, gs_ama_df)
    df = p[["항목", "값"]].merge(a[["항목", "값"]], on="항목", suffixes=("·프로", "·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로": "프로", "값·일반": "일반"})

# (= head Y) --------------------------------------------------------------------
def build_head_y_table(arr: np.ndarray) -> pd.DataFrame:
    d_add_top = g(arr, "AD4") - g(arr, "AD1")
    d_top_imp = g(arr, "AD7") - g(arr, "AD4")
    d_imp_f1  = g(arr, "AD8") - g(arr, "AD7")
    rows = [
        ["ADD - TOP", "AD4 − AD1", d_add_top],
        ["TOP - IMP", "AD7 − AD4", d_top_imp],
        ["IMP - F1",  "AD8 − AD7", d_imp_f1],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

def build_head_y_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_head_y_table(pro_arr)
    a = build_head_y_table(ama_arr)
    df = p[["항목", "값"]].merge(a[["항목", "값"]], on="항목", suffixes=("·프로", "·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로": "프로", "값·일반": "일반"})