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
    
_FRAMES = list(range(1, 11))
_SEGS   = [("1-4", 0, 3), ("4-7", 3, 6), ("7-10", 6, 9), ("1-7", 0, 6)]

def _com_x(arr, n: int) -> float:
    # 0.08*AC + 0.35*(양 어깨X)/2 + 0.30*(양 힙X)/2 + 0.15*(양 무릎X)/2 + 0.12*(양 발뒤꿈치X)/2
    return (
        0.08 * g_base(arr, f"AC{n}") +
        0.35 * (g_base(arr, f"AL{n}") + g_base(arr, f"BA{n}"))/2 +
        0.30 * (g_base(arr, f"H{n}")  + g_base(arr, f"K{n}"))/2 +
        0.15 * (g_base(arr, f"BP{n}") + g_base(arr, f"CB{n}"))/2 +
        0.12 * (g_base(arr, f"BY{n}") + g_base(arr, f"CK{n}"))/2
    )

def _base_x(arr, n: int) -> float:
    # 발 중심 60% 지점(간단화): 뒤꿈치 40% + 발끝 60%
    return (
        0.4 * (g_base(arr, f"BY{n}") + g_base(arr, f"CK{n}"))/2 +
        0.6 * (g_base(arr, f"BS{n}") + g_base(arr, f"CE{n}"))/2
    )

def _com_y(arr, n: int) -> float:
    # 0.09*AD + 0.40*(AM+BB)/2 + 0.34*(I+L)/2 + 0.17*(BQ+CC)/2
    return (
        0.09 * g_base(arr, f"AD{n}") +
        0.40 * (g_base(arr, f"AM{n}") + g_base(arr, f"BB{n}"))/2 +
        0.34 * (g_base(arr, f"I{n}")  + g_base(arr, f"L{n}"))/2 +
        0.17 * (g_base(arr, f"BQ{n}") + g_base(arr, f"CC{n}"))/2
    )

def _com_z(arr, n: int) -> float:
    # ((무릎Z + 힙Z + 가슴Z + 머리Z)/4) - 발 Z
    knee  = (g_base(arr, f"BR{n}") + g_base(arr, f"CD{n}"))/2
    hip   = (g_base(arr, f"J{n}")  + g_base(arr, f"M{n}"))/2
    chest = (g_base(arr, f"AN{n}") + g_base(arr, f"BC{n}"))/2
    head  = g_base(arr, f"AE{n}")
    foot  = (g_base(arr, f"CA{n}") + g_base(arr, f"CM{n}"))/2
    return ((knee + hip + chest + head)/4) - foot

def _series_and_diffs(values: list[float]) -> tuple[list[float], list[float]]:
    vals  = [round(v, 2) for v in values]
    diffs = [None] + [round(vals[i] - vals[i-1], 2) for i in range(1, len(vals))]
    return vals, diffs

def _segment_changes(vals: list[float]) -> list[float]:
    return [round(vals[e] - vals[s], 2) for _, s, e in _SEGS]

def build_delta_x_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    pro = [ _com_x(base_pro, n) - _base_x(base_pro, n) for n in _FRAMES ]
    ama = [ _com_x(base_ama, n) - _base_x(base_ama, n) for n in _FRAMES ]
    pro_vals, pro_diff = _series_and_diffs(pro)
    ama_vals, ama_diff = _series_and_diffs(ama)

    seg1 = _segment_changes(pro_vals); seg2 = _segment_changes(ama_vals)
    tot1 = round(sum(abs(d) for d in pro_diff[1:]), 2)
    tot2 = round(sum(abs(d) for d in ama_diff[1:]), 2)

    idx = [str(i) for i in _FRAMES] + [label for label,_,_ in _SEGS] + ["Total"]
    df = pd.DataFrame({
        "프로":        pro_vals + [None]*5,
        "일반":        ama_vals + [None]*5,
        "프로 diff":   pro_diff  + seg1 + [tot1],
        "일반 diff":   ama_diff  + seg2 + [tot2],
    }, index=idx)
    df.index.name = "Frame"
    return df

def build_delta_y_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    pro = [ _com_y(base_pro, n) for n in _FRAMES ]
    ama = [ _com_y(base_ama, n) for n in _FRAMES ]
    pro_vals, pro_diff = _series_and_diffs(pro)
    ama_vals, ama_diff = _series_and_diffs(ama)

    seg1 = _segment_changes(pro_vals); seg2 = _segment_changes(ama_vals)
    tot1 = round(sum(abs(d) for d in pro_diff[1:]), 2)
    tot2 = round(sum(abs(d) for d in ama_diff[1:]), 2)

    idx = [str(i) for i in _FRAMES] + [label for label,_,_ in _SEGS] + ["Total"]
    df = pd.DataFrame({
        "프로":        pro_vals + [None]*5,
        "일반":        ama_vals + [None]*5,
        "프로 diff":   pro_diff  + seg1 + [tot1],
        "일반 diff":   ama_diff  + seg2 + [tot2],
    }, index=idx)
    df.index.name = "Frame"
    return df

def build_delta_z_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    pro = [ _com_z(base_pro, n) for n in _FRAMES ]
    ama = [ _com_z(base_ama, n) for n in _FRAMES ]
    pro_vals, pro_diff = _series_and_diffs(pro)
    ama_vals, ama_diff = _series_and_diffs(ama)

    seg1 = _segment_changes(pro_vals); seg2 = _segment_changes(ama_vals)
    tot1 = round(sum(abs(d) for d in pro_diff[1:]), 2)
    tot2 = round(sum(abs(d) for d in ama_diff[1:]), 2)

    idx = [str(i) for i in _FRAMES] + [label for label,_,_ in _SEGS] + ["Total"]
    df = pd.DataFrame({
        "프로":        pro_vals + [None]*5,
        "일반":        ama_vals + [None]*5,
        "프로 diff":   pro_diff  + seg1 + [tot1],
        "일반 diff":   ama_diff  + seg2 + [tot2],
    }, index=idx)
    df.index.name = "Frame"
    return df

def build_summary_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    dx = build_delta_x_table(base_pro, base_ama)
    dy = build_delta_y_table(base_pro, base_ama)
    dz = build_delta_z_table(base_pro, base_ama)

    segs = ["1-4","4-7","7-10","Total"]
    axes = [("X", dx, "프로 diff", "일반 diff"),
            ("Y", dy, "프로 diff", "일반 diff"),
            ("Z", dz, "프로 diff", "일반 diff")]

    rows=[]
    for ax, df, pc, gc in axes:
        for s in segs:
            rows.append([ax, s, df.at[s, pc], df.at[s, gc]])
    rows.append(["Total","", sum(r[2] for r in rows if r[0] in ["X","Y","Z"] and r[1]=="Total"),
                          sum(r[3] for r in rows if r[0] in ["X","Y","Z"] and r[1]=="Total")])

    out = pd.DataFrame(rows, columns=["Axis","Segment","프로","일반"])
    return out

def build_smdi_mrmi_table(base_pro: np.ndarray, base_ama: np.ndarray,
                          pro_label: str="Pro", ama_label: str="Ama") -> pd.DataFrame:
    summary = build_summary_table(base_pro, base_ama)
    pro_tot = {ax: float(summary[(summary.Axis==ax) & (summary.Segment=="Total")]["프로"]) for ax in ["X","Y","Z"]}
    ama_tot = {ax: float(summary[(summary.Axis==ax) & (summary.Segment=="Total")]["일반"]) for ax in ["X","Y","Z"]}

    mrmi = {ax: round((ama_tot[ax]-pro_tot[ax]) / pro_tot[ax] * 100, 2) for ax in ["X","Y","Z"]}
    scores = {ax: 100 - abs(ama_tot[ax]-pro_tot[ax]) / pro_tot[ax] * 100 for ax in ["X","Y","Z"]}
    smdi = round(sum(scores.values())/3, 2)

    df = pd.DataFrame(
        [
            {"SMDI":100, "MRMI X":0, "MRMI Y":0, "MRMI Z":0},
            {"SMDI":smdi, "MRMI X":mrmi["X"], "MRMI Y":mrmi["Y"], "MRMI Z":mrmi["Z"]},
        ],
        index=[pro_label, ama_label],
        columns=["SMDI","MRMI X","MRMI Y","MRMI Z"]
    )
    return df
