# sections/swing_error/features/_7Trust2.py
from __future__ import annotations
import numpy as np
import pandas as pd
import re

# ── 공통 유틸 ────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _fmt(rows, baseline=None, yellow=None, cols=("검사명","현재결과","식")) -> pd.DataFrame:
    baseline = baseline or {}
    yellow   = yellow   or {}

    df_in = pd.DataFrame(rows, columns=list(cols))
    out = pd.DataFrame({
        "검사명": df_in["검사명"],
        "현재결과": df_in["현재결과"],
    })
    out.insert(0, "seg", out["검사명"])       # ← seg 보장
    out.set_index("검사명", drop=False, inplace=True)  # 인덱스
    return out



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

# ─────────────────────────────────────────────────────────────────────────────
# 3.3 Early Extension (Waist Thrust X)
def build_33_early_extension(arr: np.ndarray,
                             gs_df: pd.DataFrame | None = None,
                             baseline=None, yellow=None) -> pd.DataFrame:
    # 1/7 = (H1+K1)/2 − (H7+K7)/2  ← BASE(무지개)
    val_17 = 0.5*(g(arr,"H1")+g(arr,"K1")) - 0.5*(g(arr,"H7")+g(arr,"K7"))

    gs = g_gs(gs_df, "E12") if gs_df is not None else np.nan
    rows = [
        ("1/7",            val_17, "(H1+K1)/2 − (H7+K7)/2"),
        ("Pelvis Thrust",  gs,     "E12(GS)"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.4 Flat Sho Plane  (어깨 L/R Y, 등)
def build_34_flat_sho_plane(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("1 L/R Y",   g(arr,"BB1")-g(arr,"AM1"), "BB1 − AM1"),
        ("2. 4 L/R Y",g(arr,"BB4")-g(arr,"AM4"), "BB4 − AM4"),
        ("3. 1/4 R SHO", g(arr,"BB4")-g(arr,"BB1"), "BB4 − BB1"),
        ("4. 1/7 R SHO", g(arr,"BB7")-g(arr,"BB1"), "BB7 − BB1"),
        ("5. 10 L/R SHO Y", g(arr,"BB10")-g(arr,"AM10"), "BB10 − AM10"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.5 Flying Elbow
def build_35_flying_elbow(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("1. BOTH ELB 1 Z/4 X DIFF", (g(arr,"BG4")-g(arr,"AR4")) - (g(arr,"AT1")-g(arr,"BI1")),
         "(BG4−AR4) − (AT1−BI1)"),
        ("2. 4 R WRI/ ELB X", g(arr,"BG4")-g(arr,"BM4"), "BG4 − BM4"),
        # ("3. 1/4 DIST DIFF", np.nan, "추가 규칙 필요(보류)")
    ]
    return _fmt(rows, baseline, yellow)

# 3.6 Sway
def build_36_sway(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("1. 1/4 R KNE", g(arr,"CD4")-g(arr,"CD1"), "CD4 − CD1"),
        ("2. 1/4 R WAI", g(arr,"M4") - g(arr,"M1"), "M4 − M1"),
        ("3. 1/2 R SHO", g(arr,"BC2") - g(arr,"BC1"), "BC2 − BC1"),
        ("4. Head 1/4",  g(arr,"AE4") - g(arr,"AE1"), "AE4 − AE1"),
        ("4. Head 4/7",  g(arr,"AE7") - g(arr,"AE4"), "AE7 − AE4"),
        ("4. Head 7/8",  g(arr,"AE8") - g(arr,"AE7"), "AE8 − AE7"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.7 Casting (간단식 우선)
def build_37_casting(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("1. 5 R WRI/CHD Z", g(arr,"CP5")-g(arr,"BO5"), "CP5 − BO5"),
        ("2. 6 R WRI/CHD Y", g(arr,"CO6")-g(arr,"BN6"), "CO6 − BN6"),
        ("3. 6 R WRI/CHD Z", g(arr,"CP6")-g(arr,"BO6"), "CP6 − BO6"),
        # 복합 조건(5–7 최소값/0 조건)은 후속 규칙 필요 → 보류
    ]
    return _fmt(rows, baseline, yellow)

# 3.8 Hanging back (Z, “−” Greater)
def build_38_hanging_back(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("1/7 R WAI Z", g(arr,"M7")-g(arr,"M1"), "M7 − M1"),
        ("1/7 L WAI Z", g(arr,"J7")-g(arr,"J1"), "J7 − J1"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.9 Slide (Z, “+” Greater)
def build_39_slide(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("7 L ANK/L KNE Z", g(arr,"BR7")-g(arr,"CA7"), "BR7 − CA7"),
        ("1/7 L WAI Z",     g(arr,"J7") - g(arr,"J1"), "J7 − J1"),
        ("1/7 L SHO Z",     g(arr,"AN7")- g(arr,"AN1"),"AN7 − AN1"),
        ("1/7 HED Z",       g(arr,"AE7")- g(arr,"AE1"),"AE7 − AE1"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.10 Overswing (Y, “−” Greater)
def build_310_overswing_y(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [("4 R WRI/CHD Y", g(arr,"CO4")-g(arr,"BN4"), "CO4 − BN4")]
    return _fmt(rows, baseline, yellow)

# 3.11 Cross over (X, “−” Greater)
def build_311_cross_over_x(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [("4 R WRI/CHD X", g(arr,"CN4")-g(arr,"BM4"), "CN4 − BM4")]
    return _fmt(rows, baseline, yellow)

# 3.12 Reverse spine (Z, “+” Greater)
def build_312_reverse_spine(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    sho = 0.5*(g(arr,"AN4")+g(arr,"BC4"))
    wai = 0.5*(g(arr,"J4") +g(arr,"M4"))
    rows = [("4 WAI/SHO Z", sho - wai, "(AN4+BC4)/2 − (J4+M4)/2")]
    return _fmt(rows, baseline, yellow)

# 3.13 Chicken wing
def build_313_chicken_wing(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("8 L ELB/L WRI Z", g(arr,"AZ8")-g(arr,"AT8"), "AZ8 − AT8"),
        ("BOTH ELB 1 Z/8 X DIFF", (g(arr,"AR8")-g(arr,"BG8"))-(g(arr,"AT1")-g(arr,"BI1")),
         "(AR8−BG8) − (AT1−BI1)"),
        # ("1/8 DIST DIFF(추가)", np.nan, "추가 규칙 필요(보류)"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.14 Scooping
def build_314_scooping(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [
        ("1/7 L WRI Z", g(arr,"AZ7")-g(arr,"AZ1"), "AZ7 − AZ1"),
        ("8 L WRI/CHD Y", g(arr,"CO7")-g(arr,"AY7"), "CO7 − AY7"),
    ]
    return _fmt(rows, baseline, yellow)

# 3.15 Reverse C Finish
def build_315_reverse_c_finish(arr: np.ndarray, baseline=None, yellow=None) -> pd.DataFrame:
    rows = [("1/10 R SHO Z", g(arr,"BC10")-g(arr,"BC1"), "BC10 − BC1")]
    return _fmt(rows, baseline, yellow)
