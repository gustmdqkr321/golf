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
    
_FRAMES = range(1, 11)

def _g(arr: np.ndarray, code: str) -> float:
    return g_base(arr, code)

# ───────────────────────── 공통 유틸 ─────────────────────────
def _delta_rows_table(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    com_func,  # def f(arr, n) -> np.array([x,y,z])
    pro_label: str = "Pro",
    ama_label: str = "Ama",
) -> pd.DataFrame:
    # 프레임 i→i+1 델타 (1-2 ... 9-10)
    def deltas(arr): return [com_func(arr, i+1) - com_func(arr, i) for i in range(1, 10)]
    d_pro = deltas(pro_arr)
    d_ama = deltas(ama_arr)

    idx = [f"{i}-{i+1}" for i in range(1, 10)]
    mov = pd.DataFrame(index=idx)

    # 값 채우기 (소수 2자리)
    for comp, label in [(d_pro, pro_label), (d_ama, ama_label)]:
        tmp = pd.DataFrame(comp, index=idx, columns=["ΔX", "ΔY", "ΔZ"]).round(2)
        for ax in ["X", "Y", "Z"]:
            mov[f"Δ{ax}_{label}"] = tmp[f"Δ{ax}"]

    # 부호 불일치 표시는 Ama 쪽에만 '!' 표시, Pro도 문자열 포맷 통일
    for ax in ["X", "Y", "Z"]:
        for k in idx:
            pr = float(mov.at[k, f"Δ{ax}_{pro_label}"])
            am = float(mov.at[k, f"Δ{ax}_{ama_label}"])
            mov.at[k, f"Δ{ax}_{pro_label}"] = f"{pr:.2f}"
            mov.at[k, f"Δ{ax}_{ama_label}"] = f"{am:.2f}!" if pr * am < 0 else f"{am:.2f}"

    # 구간 합(1-4, 4-7, 7-10)
    sections = {"1-4": (1, 4), "4-7": (4, 7), "7-10": (7, 10)}
    for sec, (a, b) in sections.items():
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        for label in [pro_label, ama_label]:
            for ax in ["X", "Y", "Z"]:
                col = f"Δ{ax}_{label}"
                vals = mov.loc[keys, col].astype(str).str.rstrip("!").astype(float)
                mov.at[sec, col] = round(vals.sum(), 2)

    # Total / TotalAbs
    step_keys = [f"{i}-{i+1}" for i in range(1, 10)]
    for label in [pro_label, ama_label]:
        for ax in ["X", "Y", "Z"]:
            col = f"Δ{ax}_{label}"
            vals = mov.loc[step_keys, col].astype(str).str.rstrip("!").astype(float)
            mov.at["Total", col]    = round(vals.sum(), 2)
            mov.at["TotalAbs", col] = round(vals.abs().sum(), 2)

    # TotalXYZ: 세 축 절대합(한 줄 요약) → ΔX 컬럼에만 표기
    for label in [pro_label, ama_label]:
        abs_cols = [f"Δ{ax}_{label}" for ax in ["X", "Y", "Z"]]
        total_xyz = mov.loc["TotalAbs", abs_cols].astype(float).sum()
        mov.at["TotalXYZ", f"ΔX_{label}"] = round(total_xyz, 2)

    return mov

# ───────────── 부위별 COM 정의 ─────────────
def _com_knee(arr, n):
    x = 0.5 * (_g(arr, f"BP{n}") + _g(arr, f"CB{n}"))
    y = 0.5 * (_g(arr, f"BQ{n}") + _g(arr, f"CC{n}"))
    z = 0.5 * (_g(arr, f"BR{n}") + _g(arr, f"CD{n}"))
    return np.array([x, y, z], dtype=float)

def _com_hips(arr, n):
    x = 0.5 * (_g(arr, f"H{n}") + _g(arr, f"K{n}"))
    y = 0.5 * (_g(arr, f"I{n}") + _g(arr, f"L{n}"))
    z = 0.5 * (_g(arr, f"J{n}") + _g(arr, f"M{n}"))
    return np.array([x, y, z], dtype=float)

def _com_shoulder(arr, n):
    x = 0.5 * (_g(arr, f"AL{n}") + _g(arr, f"BA{n}"))
    y = 0.5 * (_g(arr, f"AM{n}") + _g(arr, f"BB{n}"))
    z = 0.5 * (_g(arr, f"AN{n}") + _g(arr, f"BC{n}"))
    return np.array([x, y, z], dtype=float)

def _com_head(arr, n):
    # 단일 포인트(머리 중앙) – AC/AD/AE
    return np.array([_g(arr, f"AC{n}"), _g(arr, f"AD{n}"), _g(arr, f"AE{n}")], dtype=float)

# ───────────── 표 빌더 (부위별) ─────────────
def build_movement_table_knee(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _delta_rows_table(base_pro, base_ama, _com_knee)

def build_movement_table_hips(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _delta_rows_table(base_pro, base_ama, _com_hips)

def build_movement_table_shoulder(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _delta_rows_table(base_pro, base_ama, _com_shoulder)

def build_movement_table_head(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _delta_rows_table(base_pro, base_ama, _com_head)

# ───────────── 합계 표 ─────────────
def build_total_move(base_pro: np.ndarray, base_ama: np.ndarray,
                     pro_label: str = "Pro", ama_label: str = "Ama") -> pd.DataFrame:
    tables = {
        "무릎":   build_movement_table_knee(base_pro, base_ama),
        "골반":   build_movement_table_hips(base_pro, base_ama),
        "어깨":   build_movement_table_shoulder(base_pro, base_ama),
        "머리":   build_movement_table_head(base_pro, base_ama),
    }
    segments = ["1-4", "4-7", "7-10", "Total"]
    out = []
    for seg in segments:
        row = {"구간": seg}
        for part, df in tables.items():
            for label in [pro_label, ama_label]:
                if seg == "Total":
                    val = (df.at["TotalAbs", f"ΔX_{label}"]
                         + df.at["TotalAbs", f"ΔY_{label}"]
                         + df.at["TotalAbs", f"ΔZ_{label}"])
                else:
                    a, b = map(int, seg.split("-"))
                    keys = [f"{i}-{i+1}" for i in range(a, b)]
                    acc = 0.0
                    for ax in ["X", "Y", "Z"]:
                        ser = df.loc[keys, f"Δ{ax}_{label}"].astype(str).str.rstrip("!").astype(float)
                        acc += ser.abs().sum()
                    val = acc
                row[f"{part} 총 이동({label}, cm)"] = round(float(val), 2)
        out.append(row)

    cols = ["구간"]
    for part in ["무릎", "골반", "어깨", "머리"]:
        for label in [pro_label, ama_label]:
            cols.append(f"{part} 총 이동({label}, cm)")
    return pd.DataFrame(out)[cols]

def build_total_move_ratio(base_pro: np.ndarray, base_ama: np.ndarray,
                           pro_label: str = "Pro", ama_label: str = "Ama") -> pd.DataFrame:
    """
    구간별(1-4, 4-7, 7-10, Total)로 Pro/Ama 각각에 대해
    [무릎, 골반, 어깨, 머리] 절대 이동량의 합이 100%가 되도록 비율 계산.
    """
    # 1) 부위별 이동 테이블 생성
    tables = {
        "무릎":   build_movement_table_knee(base_pro, base_ama),
        "골반":   build_movement_table_hips(base_pro, base_ama),
        "어깨":   build_movement_table_shoulder(base_pro, base_ama),
        "머리":   build_movement_table_head(base_pro, base_ama),
    }
    segments = ["1-4", "4-7", "7-10", "Total"]
    labels   = [pro_label, ama_label]

    # 2) 구간·부위·사람별 절대이동량 수집 (분모 계산용)
    #    abs_sum[seg][label] = 해당 구간에서 4부위 절대이동량의 총합
    abs_vals = {seg: {part: {lbl: 0.0 for lbl in labels} for part in tables.keys()} for seg in segments}
    abs_sum  = {seg: {lbl: 0.0 for lbl in labels} for seg in segments}

    for seg in segments:
        for part, df in tables.items():
            for lbl in labels:
                if seg == "Total":
                    val = (float(df.at["TotalAbs", f"ΔX_{lbl}"])
                         + float(df.at["TotalAbs", f"ΔY_{lbl}"])
                         + float(df.at["TotalAbs", f"ΔZ_{lbl}"]))
                else:
                    a, b = map(int, seg.split("-"))
                    keys = [f"{i}-{i+1}" for i in range(a, b)]
                    acc = 0.0
                    for ax in ["X", "Y", "Z"]:
                        ser = df.loc[keys, f"Δ{ax}_{lbl}"].astype(str).str.rstrip("!").astype(float)
                        acc += float(ser.abs().sum())
                    val = acc
                abs_vals[seg][part][lbl] = float(val)
                abs_sum[seg][lbl] += float(val)

    # 3) 비율(%) 계산: part_abs / sum_abs * 100
    rows = []
    for seg in segments:
        row = {"구간": seg}
        for part in ["무릎", "골반", "어깨", "머리"]:
            for lbl in labels:
                denom = abs_sum[seg][lbl]
                num   = abs_vals[seg][part][lbl]
                pct   = (num / denom * 100) if denom not in (0.0, -0.0) else float("nan")
                row[f"{part} 이동비율({lbl},%)"] = round(pct, 2)
        rows.append(row)

    # 4) 출력 컬럼 순서 정리
    cols = ["구간"]
    for part in ["무릎", "골반", "어깨", "머리"]:
        for lbl in labels:
            cols.append(f"{part} 이동비율({lbl},%)")

    return pd.DataFrame(rows)[cols]
