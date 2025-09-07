from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Union
import numpy as np
import pandas as pd

# ─────────────────────────── 유틸 ───────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def fmt(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

# 부위별 좌우 3축 컬럼(엑셀 컬럼명 기준)
PARTS: Dict[str, Dict[str, List[str]]] = {
    "knee":     {"L": ["BP", "BQ", "BR"], "R": ["CB", "CC", "CD"]},
    "pelvis":   {"L": ["H",  "I",  "J" ], "R": ["K",  "L",  "M" ]},
    "shoulder": {"L": ["AL", "AM", "AN"], "R": ["BA", "BB", "BC"]},
    "wrist":    {"L": ["AX", "AY", "AZ"], "R": ["BM", "BN", "BO"]},
}
FRAMES = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2"]  # 1..9

@dataclass
class ForceResult:
    table_main: pd.DataFrame
    table_opposite: pd.DataFrame
    table_same_top3: pd.DataFrame

# ─────────────────────── 헬퍼: 좌표/시간 ───────────────────────
def extract_times(arr: np.ndarray) -> np.ndarray:
    """B열 시간(초), 1..9프레임"""
    return np.array([g(arr, f"B{t}") for t in range(1,10)], dtype=float)

def _center_series(arr: np.ndarray, part: str, cm_to_m: float = 0.01) -> np.ndarray:
    """각 프레임(1..9)에서 (좌+우)/2 → ADD를 원점으로 평행이동, cm→m"""
    p = PARTS[part]
    C = []
    for t in range(1, 10):
        L = np.array([g(arr, f"{p['L'][0]}{t}"), g(arr, f"{p['L'][1]}{t}"), g(arr, f"{p['L'][2]}{t}")], dtype=float)
        R = np.array([g(arr, f"{p['R'][0]}{t}"), g(arr, f"{p['R'][1]}{t}"), g(arr, f"{p['R'][2]}{t}")], dtype=float)
        C.append((L + R) / 2.0)
    C = np.vstack(C) * cm_to_m     # cm → m
    C = C - C[0]                    # ADD 원점 이동(차분엔 영향 없음, 가독성용)
    return C  # (9,3)

# ─────────────────────── 핵심: 힘(시간반영) ───────────────────────
def _segment_forces(C: np.ndarray, times: np.ndarray | None, mass: float = 60.0) -> np.ndarray:
    """
    C: (9,3) m, times: (9,) sec or None
    v_s = (C_{s+1}-C_s) / Δt_s          (s=1..8)
    F_row1..8 = m * 0.5*(v_s + v_{s+1}) (마지막 FH2는 v_8 단독)
    반환: (9,3) — row0(ADD)=nan, row1..8=BH..FH2
    """
    n = C.shape[0]  # 9
    if times is None:
        dt = np.ones(n-1, dtype=float)
    else:
        times = times.astype(float)
        dt = np.diff(times)               # (8,)
        # 0으로 나누는 경우 방지
        dt[dt == 0] = 1.0

    v = (C[1:] - C[:-1]) / dt[:, None]    # (8,3)
    F = np.full((n, 3), np.nan, dtype=float)
    for s in range(1, n-1):               # 1..7
        v_avg = 0.5 * (v[s-1] + v[s])
        F[s] = mass * v_avg
    F[n-1] = mass * v[-1]                 # 마지막 FH2는 v8 단독
    return F

# ─────────────────────── 표 생성 ───────────────────────
def _mk_main_table(F_r: np.ndarray, F_h: np.ndarray, *, summary_mode: str = "mean") -> pd.DataFrame:
    """summary_mode: 'mean' (예시코드와 동일) | 'abs_sum' (절대합)"""
    rows = []
    for i, name in enumerate(FRAMES):
        r = F_r[i]; h = F_h[i]
        opp = np.sign(r) * np.sign(h) == -1
        def mark(v, flag): return f"{fmt(v)}{' ❗' if (isinstance(flag, np.ndarray) and flag.any()) or (isinstance(flag, (bool, np.bool_)) and flag) else ''}"
        rows.append([
            name,
            fmt(r[0]), fmt(r[1]), fmt(r[2]),
            mark(h[0], opp[0]), mark(h[1], opp[1]), mark(h[2], opp[2]),
            fmt(abs(r[0]-h[0])), fmt(abs(r[1]-h[1])), fmt(abs(r[2]-h[2])),
        ])
    df = pd.DataFrame(rows, columns=["Frame","Rory_X","Rory_Y","Rory_Z","Hong_X","Hong_Y","Hong_Z","Diff_X","Diff_Y","Diff_Z"])

    # 요약 행 계산
    idx = {name:i for i,name in enumerate(FRAMES)}
    segs = {
        "요약 1-4": [idx["BH"], idx["BH2"], idx["TOP"]],
        "요약 4-7": [idx["TR"], idx["DH"], idx["IMP"]],
        "요약 7-9": [idx["FH1"], idx["FH2"]],
    }

    def summarize(rows_idx: List[int]) -> List[str]:
        R = F_r[rows_idx]; H = F_h[rows_idx]
        if summary_mode == "mean":
            r_sum = np.nanmean(R, axis=0); h_sum = np.nanmean(H, axis=0)
        else:  # abs_sum
            r_sum = np.nansum(np.abs(R), axis=0); h_sum = np.nansum(np.abs(H), axis=0)
        d_sum = np.abs(r_sum - h_sum)
        return [fmt(r_sum[0]), fmt(r_sum[1]), fmt(r_sum[2]),
                fmt(h_sum[0]), fmt(h_sum[1]), fmt(h_sum[2]),
                fmt(d_sum[0]), fmt(d_sum[1]), fmt(d_sum[2])]

    for title, idxs in segs.items():
        df.loc[len(df)] = [title] + summarize(idxs)

    # 부호반대비율(요약 제외, BH..FH2의 8행×3축)
    total = 0; opposite = 0
    for i in range(1, 9):
        r = F_r[i]; h = F_h[i]
        for a in range(3):
            if np.isnan(r[a]) or np.isnan(h[a]): continue
            total += 1
            if np.sign(r[a]) * np.sign(h[a]) == -1:
                opposite += 1
    ratio = (opposite/total) if total else 0.0
    df.loc[len(df)] = ["부호반대비율", fmt(ratio), "", "", "", "", "", "", "", ""]

    # 요청: 1-7, 1-9는 "절대값 합" 고정
    seg_1_7 = [idx["BH"], idx["BH2"], idx["TOP"], idx["TR"], idx["DH"], idx["IMP"]]
    seg_1_9 = seg_1_7 + [idx["FH1"], idx["FH2"]]
    def abs_sum(idxs):
        R = np.nansum(np.abs(F_r[idxs]), axis=0); H = np.nansum(np.abs(F_h[idxs]), axis=0); D = np.abs(R-H)
        return [fmt(R[0]), fmt(R[1]), fmt(R[2]), fmt(H[0]), fmt(H[1]), fmt(H[2]), fmt(D[0]), fmt(D[1]), fmt(D[2])]
    df.loc[len(df)] = ["1-7"] + abs_sum(seg_1_7)
    df.loc[len(df)] = ["1-9"] + abs_sum(seg_1_9)
    return df

def _mk_opposite_table(F_r: np.ndarray, F_h: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(FRAMES):
        if name == "ADD": continue
        for axis, ax_name in enumerate(["X","Y","Z"]):
            r, h = F_r[i,axis], F_h[i,axis]
            if np.isnan(r) or np.isnan(h): continue
            if np.sign(r) * np.sign(h) == -1:
                rows.append([name, ax_name, float(r), float(h), abs(float(r)-float(h))])
    if not rows:
        return pd.DataFrame(columns=["Frame","Axis","Rory","Hong","|Diff|"])
    return (pd.DataFrame(rows, columns=["Frame","Axis","Rory","Hong","|Diff|"])
              .sort_values("|Diff|", ascending=False, ignore_index=True))

def _mk_same_top3(F_r: np.ndarray, F_h: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(FRAMES):
        if name == "ADD": continue
        for axis, ax_name in enumerate(["X","Y","Z"]):
            r, h = F_r[i,axis], F_h[i,axis]
            if np.isnan(r) or np.isnan(h): continue
            if np.sign(r) * np.sign(h) >= 0:
                rows.append([name, ax_name, float(r), float(h), abs(float(r)-float(h))])
    if not rows:
        return pd.DataFrame(columns=["Frame","Axis","Rory","Hong","|Diff|"])
    return (pd.DataFrame(rows, columns=["Frame","Axis","Rory","Hong","|Diff|"])
              .sort_values("|Diff|", ascending=False, ignore_index=True)
              .head(3))

# ─────────────────────── 외부 호출 API ───────────────────────
def build_all_tables(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    *,
    part: str = "knee",
    mass: float = 60.0,
    times_pro: np.ndarray | None = None,
    times_ama: np.ndarray | None = None,
    summary_mode: str = "mean",  # "mean"=예시코드, "abs_sum"=이전 규격
) -> ForceResult:

    C_r = _center_series(pro_arr, part)
    C_h = _center_series(ama_arr, part)

    if times_pro is None: times_pro = extract_times(pro_arr)
    if times_ama is None: times_ama = extract_times(ama_arr)

    F_r = _segment_forces(C_r, times_pro, mass=mass)
    F_h = _segment_forces(C_h, times_ama, mass=mass)

    main   = _mk_main_table(F_r, F_h, summary_mode=summary_mode)
    opp    = _mk_opposite_table(F_r, F_h)
    same3  = _mk_same_top3(F_r, F_h)
    return ForceResult(main, opp, same3)
