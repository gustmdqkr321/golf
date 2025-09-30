from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
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

# ─────────────────────────── 정의 ───────────────────────────
# 토크용 파트 정의: center(좌/우 3축), target(좌/우 3축)
#   무릎:   center=발목(BY,BZ,CA / CK,CL,CM),  target=무릎(BP,BQ,BR / CB,CC,CD)
#   골반:   center=무릎(...),                  target=골반(H,I,J / K,L,M)
#   어깨:   center=골반(...),                  target=어깨(AL,AM,AN / BA,BB,BC)
PARTS_TORQUE: Dict[str, Dict[str, List[str]]] = {
    "knee":     { "center": ["BY","BZ","CA","CK","CL","CM"], "target": ["BP","BQ","BR","CB","CC","CD"] },
    "pelvis":   { "center": ["BP","BQ","BR","CB","CC","CD"], "target": ["H","I","J","K","L","M"]       },
    "shoulder": { "center": ["H","I","J","K","L","M"],       "target": ["AL","AM","AN","BA","BB","BC"] },
    
}

# 표 라벨(ADD는 계산용 시간/좌표에만 쓰고 표는 BH..FH2 9행)
FRAMES = ["BH","BH2","TOP","TR","DH","IMP","FH1","FH2"]
FRAMES_FULL = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]  # 표 첫 줄에 ADD는 안 씀, 내부 인덱싱용

@dataclass
class TorqueResult:
    table_main: pd.DataFrame
    table_opposite: pd.DataFrame
    table_same_top3: pd.DataFrame

# ─────────────────────── 헬퍼: 좌표/시간 ───────────────────────
def extract_times_10(arr: np.ndarray) -> np.ndarray:
    """B열 시간(초), 1..10프레임(ADD~Finish) → (10,)"""
    return np.array([g(arr, f"B{t}") for t in range(1, 11)], dtype=float)

def _series_LR_mean(arr: np.ndarray, cols6: List[str], t_from=1, t_to=10, cm_to_m: float = 0.01) -> np.ndarray:
    """
    cols6: [Lx, Ly, Lz, Rx, Ry, Rz], t_from..t_to inclusive
    반환: (t_len, 3)  (좌/우 평균, cm→m, ADD 원점 이동 아님 — 토크 r에 원점이 필요없음)
    """
    Lx,Ly,Lz, Rx,Ry,Rz = cols6
    data = []
    for t in range(t_from, t_to+1):
        L = np.array([g(arr, f"{Lx}{t}"), g(arr, f"{Ly}{t}"), g(arr, f"{Lz}{t}")], dtype=float)
        R = np.array([g(arr, f"{Rx}{t}"), g(arr, f"{Ry}{t}"), g(arr, f"{Rz}{t}")], dtype=float)
        data.append((L + R) / 2.0)
    return np.vstack(data) * cm_to_m  # (N,3)

# ─────────────────────── 핵심: 토크(시간반영) ───────────────────────
def _torque_series(center: np.ndarray, target: np.ndarray, times10: np.ndarray, mass: float) -> np.ndarray:
    """
    center: (10,3) m, target: (10,3) m, times10: (10,) sec
    v_s = (target_{s+1}-target_s)/Δt_s,  s=1..9
    F_s = m * v_s
    r_s = target_s - center_s
    τ_s = r_s × F_s    → (9,3)  [BH..FH2]
    """
    # 안전한 Δt
    dt = np.diff(times10)  # (9,)
    dt[dt == 0] = 1.0

    v = (target[1:] - target[:-1]) / dt[:, None]  # (9,3)
    F = mass * v                                   # (9,3)
    r = (target[:-1] - center[:-1])                # (9,3), 시작 프레임 기준
    tau = np.cross(r, F)                           # (9,3)
    return tau

# ─────────────────────── 표 생성 ───────────────────────
def _mk_main_table_tau(T_r: np.ndarray, T_h: np.ndarray, *, summary_mode: str = "mean") -> pd.DataFrame:
    """
    T_r/T_h: (9,3) — BH..Finish
    """
    rows = []
    frame_labels = FRAMES_FULL[1:]  # BH..Finish (9개)
    for i, name in enumerate(frame_labels):
        r = T_r[i]; h = T_h[i]
        opp = np.sign(r) * np.sign(h) == -1
        def mark(v, flag):
            return f"{fmt(v)}{' ❗' if (isinstance(flag, np.ndarray) and flag.any()) or (isinstance(flag, (bool, np.bool_)) and flag) else ''}"
        rows.append([
            name,
            fmt(r[0]), fmt(r[1]), fmt(r[2]),
            mark(h[0], opp[0]), mark(h[1], opp[1]), mark(h[2], opp[2]),
            fmt(abs(r[0]-h[0])), fmt(abs(r[1]-h[1])), fmt(abs(r[2]-h[2])),
        ])
    df = pd.DataFrame(
        rows,
        columns=["Frame","Rory_X","Rory_Y","Rory_Z","Hong_X","Hong_Y","Hong_Z","Diff_X","Diff_Y","Diff_Z"]
    )

    # 요약 구간도 Finish까지 반영 (각 3개 구간)
    segs = {
        "요약 1-4": [0,1,2],    # BH, BH2, TOP
        "요약 4-7": [3,4,5],    # TR, DH, IMP
        "요약 7-10": [6,7,8],   # FH1, FH2, Finish
    }
    def summarize(idxs: List[int]) -> List[str]:
        R = T_r[idxs]; H = T_h[idxs]
        if summary_mode == "mean":
            r_sum = np.nanmean(R, axis=0); h_sum = np.nanmean(H, axis=0)
        else:
            r_sum = np.nansum(np.abs(R), axis=0); h_sum = np.nansum(np.abs(H), axis=0)
        d_sum = np.abs(r_sum - h_sum)
        return [fmt(r_sum[0]), fmt(r_sum[1]), fmt(r_sum[2]),
                fmt(h_sum[0]), fmt(h_sum[1]), fmt(h_sum[2]),
                fmt(d_sum[0]), fmt(d_sum[1]), fmt(d_sum[2])]
    for title, idxs in segs.items():
        df.loc[len(df)] = [title] + summarize(idxs)

    # 부호반대비율 (BH..Finish 9행×3축)
    total = 0; opposite = 0
    for i in range(0, 9):
        r = T_r[i]; h = T_h[i]
        for a in range(3):
            if np.isnan(r[a]) or np.isnan(h[a]): continue
            total += 1
            if np.sign(r[a]) * np.sign(h[a]) == -1:
                opposite += 1
    ratio = (opposite/total) if total else 0.0
    df.loc[len(df)] = ["부호반대비율", fmt(ratio), "", "", "", "", "", "", "", ""]
    return df

def _mk_opposite_table_tau(T_r: np.ndarray, T_h: np.ndarray) -> pd.DataFrame:
    rows = []
    frame_labels = FRAMES_FULL[1:]  # BH..FH2
    for i, name in enumerate(frame_labels):
        for axis, ax_name in enumerate(["X","Y","Z"]):
            r, h = T_r[i,axis], T_h[i,axis]
            if np.isnan(r) or np.isnan(h): continue
            if np.sign(r) * np.sign(h) == -1:
                rows.append([name, ax_name, float(r), float(h), abs(float(r)-float(h))])
    if not rows:
        return pd.DataFrame(columns=["Frame","Axis","Rory","Hong","|Diff|"])
    return (pd.DataFrame(rows, columns=["Frame","Axis","Rory","Hong","|Diff|"])
              .sort_values("|Diff|", ascending=False, ignore_index=True))

def _mk_same_top3_tau(T_r: np.ndarray, T_h: np.ndarray) -> pd.DataFrame:
    rows = []
    frame_labels = FRAMES_FULL[1:]
    for i, name in enumerate(frame_labels):
        for axis, ax_name in enumerate(["X","Y","Z"]):
            r, h = T_r[i,axis], T_h[i,axis]
            if np.isnan(r) or np.isnan(h): continue
            if np.sign(r) * np.sign(h) >= 0:
                rows.append([name, ax_name, float(r), float(h), abs(float(r)-float(h))])
    if not rows:
        return pd.DataFrame(columns=["Frame","Axis","Rory","Hong","|Diff|"])
    return (pd.DataFrame(rows, columns=["Frame","Axis","Rory","Hong","|Diff|"])
              .sort_values("|Diff|", ascending=False, ignore_index=True)
              .head(3))

# ─────────────────────── 외부 호출 API ───────────────────────
def build_torque_tables(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    *,
    part: str = "knee",        # "knee" | "pelvis" | "shoulder"
    mass: float = 60.0,
    summary_mode: str = "mean" # "mean" | "abs_sum"
) -> TorqueResult:
    if part not in PARTS_TORQUE:
        raise ValueError(f"지원하지 않는 part: {part} (knee|pelvis|shoulder)")

    spec = PARTS_TORQUE[part]
    times_p = extract_times_10(pro_arr)   # (10,)
    times_a = extract_times_10(ama_arr)   # (10,)

    center_p = _series_LR_mean(pro_arr, spec["center"], 1, 10)  # (10,3)
    target_p = _series_LR_mean(pro_arr, spec["target"], 1, 10)  # (10,3)
    center_a = _series_LR_mean(ama_arr, spec["center"], 1, 10)
    target_a = _series_LR_mean(ama_arr, spec["target"], 1, 10)

    # τ(토크) 시리즈 (BH..FH2 9행)
    T_p = _torque_series(center_p, target_p, times_p, mass)
    T_a = _torque_series(center_a, target_a, times_a, mass)

    main   = _mk_main_table_tau(T_p, T_a, summary_mode=summary_mode)
    opp    = _mk_opposite_table_tau(T_p, T_a)
    same3  = _mk_same_top3_tau(T_p, T_a)
    return TorqueResult(main, opp, same3)
