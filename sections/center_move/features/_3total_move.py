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
    
# 프레임 라벨(고정: 10프레임)
_FRAMES_LABELS = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","Finish"]
_FR_IDX = list(range(1, 11))  # 1..10

# 축별 마커 맵 (좌/우 없으면 None → 단일 포인트)
_MARK_Z = {
    "Ankle":    ("CA","CM"),
    "Knee":     ("BR","CD"),
    "Waist":    ("J","M"),
    "Shoulder": ("AN","BC"),
    "Head":     ("AE", None),
}
_MARK_X = {
    "Knee":     ("BP","CB"),
    "Waist":    ("H","K"),
    "Shoulder": ("AL","BA"),
    "Head":     ("AC", None),
}
_MARK_Y = {
    "Knee":     ("BQ","CC"),
    "Waist":    ("I","L"),
    "Shoulder": ("AM","BB"),
    "Head":     ("AD", None),
}

def _val(arr: np.ndarray, addr: str) -> float:
    return float(g_base(arr, addr))

def _axis_value(arr: np.ndarray, L: str, R: str | None, n: int) -> float:
    if R is None:
        return _val(arr, f"{L}{n}")
    return ( _val(arr, f"{L}{n}") + _val(arr, f"{R}{n}") ) / 2.0

def _build_axis_table(base_pro: np.ndarray, base_ama: np.ndarray,
                      mark_map: dict[str, tuple[str, str | None]],
                      axis_letter: str,
                      pro_label: str="Pro", ama_label: str="Ama") -> pd.DataFrame:
    """
    프레임별 값 + (1-4, 4-7, 7-10, Total) 4줄 추가.
    1-4/4-7/7-10 구간은 Pro와 Ama 부호 불일치 시 Ama 쪽에 '!' 표시.
    Total 은 |d1|+|d2|+|d3|.
    """
    # 1) 프레임별 값 테이블
    data = {"Frame": _FRAMES_LABELS}
    for part, (L, R) in mark_map.items():
        p_vals, a_vals = [], []
        for n in _FR_IDX:
            p = _axis_value(base_pro, L, R, n)
            a = _axis_value(base_ama, L, R, n)
            p_vals.append(round(p, 2)); a_vals.append(round(a, 2))
        data[f"{pro_label} {part} {axis_letter}"] = p_vals
        data[f"{ama_label} {part} {axis_letter}"] = a_vals
    df = pd.DataFrame(data)

    # 2) 구간 변화량 계산
    # 인덱스: 0=ADD, 3=TOP, 6=IMP, 9=Finish → (1-4), (4-7), (7-10)
    numeric_cols = [c for c in df.columns if c != "Frame"]
    d1 = (df.loc[3, numeric_cols] - df.loc[0, numeric_cols]).round(2)  # 1→4
    d2 = (df.loc[6, numeric_cols] - df.loc[3, numeric_cols]).round(2)  # 4→7
    d3 = (df.loc[9, numeric_cols] - df.loc[6, numeric_cols]).round(2)  # 7→10
    dT = (abs(d1) + abs(d2) + abs(d3)).round(2)                       # Total = |d1|+|d2|+|d3|

    r1 = pd.concat([pd.Series({"Frame": "1-4"}),  d1])
    r2 = pd.concat([pd.Series({"Frame": "4-7"}),  d2])
    r3 = pd.concat([pd.Series({"Frame": "7-10"}), d3])
    rT = pd.concat([pd.Series({"Frame": "Total"}), dT])
    df = pd.concat([df, pd.DataFrame([r1, r2, r3, rT])], ignore_index=True)

    # 3) 부호 불일치 '!' 표시 (1-4, 4-7, 7-10만)
    seg_rows = [len(df)-4, len(df)-3, len(df)-2]  # 1-4,4-7,7-10
    for col in df.columns:
        if col == "Frame": 
            continue
        # 이 컬럼이 Ama 컬럼이면 Pro 대응 컬럼 찾기
        if col.startswith(f"{ama_label} "):
            pro_col = col.replace(f"{ama_label} ", f"{pro_label} ")
            for r in seg_rows:
                p = float(df.at[r, pro_col])
                a = float(df.at[r, col])
                # 보기 좋게 ±기호와 함께, 부호 다르면 Ama에 '!' 추가
                df.at[r, pro_col] = f"{p:+.2f}"
                df.at[r, col]     = f"{a:+.2f}!" if p * a < 0 else f"{a:+.2f}"
    total_row = len(df) - 1  # 'Total'
    for col in df.columns:
        if col == "Frame":
            continue
        try:
            df.at[total_row, col] = f"{float(df.at[total_row, col]):.2f}"
        except Exception:
            pass

    return df

# 공개 API ─────────────────────────────────────────
def build_z_report_table(base_pro: np.ndarray, base_ama: np.ndarray,
                         pro_label: str="Pro", ama_label: str="Ama") -> pd.DataFrame:
    return _build_axis_table(base_pro, base_ama, _MARK_Z, "Z", pro_label, ama_label)

def build_x_report_table(base_pro: np.ndarray, base_ama: np.ndarray,
                         pro_label: str="Pro", ama_label: str="Ama") -> pd.DataFrame:
    return _build_axis_table(base_pro, base_ama, _MARK_X, "X", pro_label, ama_label)

def build_y_report_table(base_pro: np.ndarray, base_ama: np.ndarray,
                         pro_label: str="Pro", ama_label: str="Ama") -> pd.DataFrame:
    return _build_axis_table(base_pro, base_ama, _MARK_Y, "Y", pro_label, ama_label)
