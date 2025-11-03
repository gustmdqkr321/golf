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
    def deltas(arr): 
        return [com_func(arr, i+1) - com_func(arr, i) for i in range(1, 10)]

    d_pro = deltas(pro_arr)
    d_ama = deltas(ama_arr)

    step_idx = [f"{i}-{i+1}" for i in range(1, 10)]
    mov = pd.DataFrame(index=step_idx)

    # 값 채우기 (소수 2자리)
    for comp, label in [(d_pro, pro_label), (d_ama, ama_label)]:
        tmp = pd.DataFrame(comp, index=step_idx, columns=["ΔX", "ΔY", "ΔZ"]).round(2)
        for ax in ["X", "Y", "Z"]:
            mov[f"Δ{ax}_{label}"] = tmp[f"Δ{ax}"]

    # 부호 불일치 표시는 Ama 쪽에만 '!' 표시(웹에서 시각 구분용)
    for ax in ["X", "Y", "Z"]:
        for k in step_idx:
            pr = float(mov.at[k, f"Δ{ax}_{pro_label}"])
            am = float(mov.at[k, f"Δ{ax}_{ama_label}"])
            mov.at[k, f"Δ{ax}_{pro_label}"] = f"{pr:.2f}"
            mov.at[k, f"Δ{ax}_{ama_label}"] = f"{am:.2f}!" if pr * am < 0 else f"{am:.2f}"

    # ── 요약 구간 ─────────────────────────────────────────
    segs3 = [("1-4", 1, 4), ("4-7", 4, 7), ("7-10", 7, 10)]  # [시작,끝) 구간

    # 1) 일반합(연속 step 델타의 산술합)
    for seg_label, a, b in segs3:
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        for label in [pro_label, ama_label]:
            for ax in ["X", "Y", "Z"]:
                col = f"Δ{ax}_{label}"
                vals = mov.loc[keys, col].astype(str).str.rstrip("!").astype(float)
                mov.at[seg_label, col] = round(float(vals.sum()), 2)

    # Total(일반합의 전체)
    for label in [pro_label, ama_label]:
        for ax in ["X", "Y", "Z"]:
            col = f"Δ{ax}_{label}"
            vals = mov.loc[step_idx, col].astype(str).str.rstrip("!").astype(float)
            mov.at["Total", col] = round(float(vals.sum()), 2)

    # 2) 절대값합(각 step의 절대값 합)
    for seg_label, a, b in segs3:
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        abs_label = f"abs {seg_label}"
        for label in [pro_label, ama_label]:
            for ax in ["X", "Y", "Z"]:
                col = f"Δ{ax}_{label}"
                vals = mov.loc[keys, col].astype(str).str.rstrip("!").astype(float).abs()
                mov.at[abs_label, col] = round(float(vals.sum()), 2)

    # abs Total
    for label in [pro_label, ama_label]:
        for ax in ["X", "Y", "Z"]:
            col = f"Δ{ax}_{label}"
            vals = mov.loc[step_idx, col].astype(str).str.rstrip("!").astype(float).abs()
            mov.at["TotalAbs", col] = round(float(vals.sum()), 2)

    # 3) TotalXYZ: 세 축 절대합(요약 한 줄) → ΔX 컬럼에만 표기
    for label in [pro_label, ama_label]:
        abs_cols = [f"Δ{ax}_{label}" for ax in ["X", "Y", "Z"]]
        total_xyz = mov.loc["TotalAbs", abs_cols].astype(float).sum()
        mov.at["TotalXYZ", f"ΔX_{label}"] = round(float(total_xyz), 2)

    # 인덱스 정렬: step(1-2…9-10) → 일반합(1-4,4-7,7-10,Total) → 절대값합(abs 1-4,abs 4-7,abs 7-10,abs Total) → TotalXYZ
    desired_order = (
        step_idx +
        [lab for (lab, _, _) in segs3] + ["Total"] +
        [f"abs {lab}" for (lab, _, _) in segs3] + ["TotalAbs"] +
        ["TotalXYZ"]
    )
    mov = mov.reindex(desired_order)
    def _fmt2(x):
        # '1.23!' 같은 표식은 느낌표 유지한 채 두 자리로
        if isinstance(x, str) and x.endswith('!'):
            try:
                v = float(x[:-1])
                return f"{v:.2f}!"
            except Exception:
                return x
        # 빈값/NaN은 공백으로
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        try:
            return f"{float(x):.2f}"
        except Exception:
            return x

    for col in mov.columns:
        mov[col] = mov[col].map(_fmt2)
    mov.insert(0, "seg", mov.index.astype(str))
    
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
                    # 🔧 문자열일 수 있으므로 안전하게 float 캐스팅 후 합산
                    x = float(df.at["TotalAbs", f"ΔX_{label}"])
                    y = float(df.at["TotalAbs", f"ΔY_{label}"])
                    z = float(df.at["TotalAbs", f"ΔZ_{label}"])
                    val = x + y + z
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
    [무릎, 골반, 어깨, 머리] 절대 이동량 비율을 계산하고,
    반올림(소수 둘째자리) 이후에도 '합계=100.00%'가 정확히 되도록 보정한다.
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
    parts    = ["무릎", "골반", "어깨", "머리"]

    # 2) 구간·부위·사람별 절대이동량 수집
    abs_vals = {seg: {part: {lbl: 0.0 for lbl in labels} for part in parts} for seg in segments}
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
                        # 일부 셀에 '!' 같은 표식이 있다면 제거
                        ser = df.loc[keys, f"Δ{ax}_{lbl}"].astype(str).str.rstrip("!").astype(float)
                        acc += float(ser.abs().sum())
                    val = acc
                abs_vals[seg][part][lbl] = float(val)
                abs_sum[seg][lbl] += float(val)

    # 3) 비율(%) 계산 + 반올림 보정(합계=100.00)
    rows = []
    for seg in segments:
        row = {"구간": seg}
        for lbl in labels:
            denom = abs_sum[seg][lbl]

            # 분모가 0이면 전부 NaN
            if denom == 0.0:
                for part in parts:
                    row[f"{part} 이동비율({lbl},%)"] = float("nan")
                continue

            # (a) 소수점 2자리 반올림 전 비율
            raw = {part: (abs_vals[seg][part][lbl] / denom * 100.0) for part in parts}
            # (b) 두 자리 반올림
            rounded = {part: round(raw[part], 2) for part in parts}
            # (c) 합 보정: 100.00 - 합계 차이를 최대 잔여(remainder) 가진 항목에 더함
            sum_rounded = sum(rounded.values())
            diff = round(100.0 - sum_rounded, 2)  # diff는 -0.01 ~ +0.01 등 미세오차 가능

            if abs(diff) >= 0.01:  # 의미 있는 오차일 때만 보정
                # 각 항목의 소수점 아래 잔여(반올림 전 → 반올림 후)
                remainders = {part: (raw[part] - rounded[part]) for part in parts}
                # diff>0이면 소수부가 큰(내림에 가까운) 항목에 더하고, diff<0이면 소수부가 작은(올림에 가까운) 항목에서 뺀다
                target_part = max(remainders, key=remainders.get) if diff > 0 else min(remainders, key=remainders.get)
                rounded[target_part] = round(rounded[target_part] + diff, 2)

            # 최종 기록
            for part in parts:
                row[f"{part} 이동비율({lbl},%)"] = rounded[part]

            # 혹시라도 수치 안정용으로 마지막에 한 번 더 재합 보정(안전장치)
            # (여기서는 최대 0.01 오차 정도만 남을 수 있는데, 표시는 그대로 둔다)

        rows.append(row)

    # 4) 출력 컬럼 순서
    cols = ["구간"]
    for part in parts:
        for lbl in labels:
            cols.append(f"{part} 이동비율({lbl},%)")

    return pd.DataFrame(rows)[cols]