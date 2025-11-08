
# ─────────────────────────────────────────────────────────────────────────────
# sections/forces/features/_2torque.py  (patched for Excel numeric + yellow_rows)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

# ─────────────────────────── 정의 ───────────────────────────
# 토크용 파트 정의: center(좌/우 3축), target(좌/우 3축)
PARTS_TORQUE: Dict[str, Dict[str, List[str]]] = {
    # center=발목, target=무릎
    "knee":     { "center": ["BY","BZ","CA","CK","CL","CM"], "target": ["BP","BQ","BR","CB","CC","CD"] },
    # center=무릎, target=골반
    "pelvis":   { "center": ["BP","BQ","BR","CB","CC","CD"], "target": ["H","I","J","K","L","M"]       },
    # center=골반, target=어깨
    "shoulder": { "center": ["H","I","J","K","L","M"],       "target": ["AL","AM","AN","BA","BB","BC"] },
}

# 표 라벨
FRAMES = ["BH","BH2","TOP","TR","DH","IMP","FH1","FH2"]
FRAMES_FULL = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

@dataclass
class TorqueResult:
    table_main: pd.DataFrame
    table_opposite: pd.DataFrame
    table_same_top3: pd.DataFrame

# ─────────────────────── 헬퍼: 좌표/시간 ───────────────────────
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


def extract_times_10(arr: np.ndarray) -> np.ndarray:
    """B열 시간(초), 1..10프레임(ADD~Finish) → (10,)"""
    return np.array([g(arr, f"B{t}") for t in range(1, 11)], dtype=float)


def _series_LR_mean(arr: np.ndarray, cols6: List[str], t_from=1, t_to=10, cm_to_m: float = 0.01) -> np.ndarray:
    """
    cols6: [Lx, Ly, Lz, Rx, Ry, Rz], t_from..t_to inclusive
    반환: (t_len, 3)  (좌/우 평균, cm→m)
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
    τ_s = r_s × F_s    → (9,3)  [BH..FIN]
    """
    dt = np.diff(times10)           # (9,)
    dt[dt == 0] = 1.0

    v = (target[1:] - target[:-1]) / dt[:, None]  # (9,3)
    F = mass * v                                   # (9,3)
    r = (target[:-1] - center[:-1])                # (9,3)
    tau = np.cross(r, F)                           # (9,3)
    return tau

# ─────────────────────── 표 생성 ───────────────────────
def _mk_main_table_tau(T_r: np.ndarray, T_h: np.ndarray) -> pd.DataFrame:
    """
    T_r/T_h: (9,3) — BH..FIN
    요약은 모두 '절대값 합(abs sum)' 기준.
    """
    rows = []
    frame_labels = FRAMES_FULL[1:]  # BH..FIN (9개)
    for i, name in enumerate(frame_labels):
        r = T_r[i]; h = T_h[i]
        rows.append([
            name,
            float(r[0]) if np.isfinite(r[0]) else np.nan,
            float(r[1]) if np.isfinite(r[1]) else np.nan,
            float(r[2]) if np.isfinite(r[2]) else np.nan,
            float(h[0]) if np.isfinite(h[0]) else np.nan,
            float(h[1]) if np.isfinite(h[1]) else np.nan,
            float(h[2]) if np.isfinite(h[2]) else np.nan,
            float(abs(r[0]-h[0])) if np.isfinite(r[0]) and np.isfinite(h[0]) else np.nan,
            float(abs(r[1]-h[1])) if np.isfinite(r[1]) and np.isfinite(h[1]) else np.nan,
            float(abs(r[2]-h[2])) if np.isfinite(r[2]) and np.isfinite(h[2]) else np.nan,
        ])
    df = pd.DataFrame(
        rows,
        columns=["Frame","Pro_X","Pro_Y","Pro_Z","Ama_X","Ama_Y","Ama_Z","Diff_X","Diff_Y","Diff_Z"]
    )

    # 요약 구간(절대합): BH~TOP / TR~IMP / FH1~FIN
    segs = {
        "요약 1-4": [0,1,2],    # BH, BH2, TOP
        "요약 4-7": [3,4,5],    # TR, DH, IMP
        "요약 7-10": [6,7,8],   # FH1, FH2, FIN
    }

    def abs_sum_rows(idxs: List[int]) -> List[float]:
        R = np.nansum(np.abs(T_r[idxs]), axis=0)
        H = np.nansum(np.abs(T_h[idxs]), axis=0)
        D = np.abs(R - H)
        return [float(R[0]), float(R[1]), float(R[2]),
                float(H[0]), float(H[1]), float(H[2]),
                float(D[0]), float(D[1]), float(D[2])]

    for title, idxs in segs.items():
        df.loc[len(df)] = [title] + abs_sum_rows(idxs)

    # 부호반대비율 (BH..FIN 9행×3축)
    total = 0; opposite = 0
    for i in range(0, 9):
        r = T_r[i]; h = T_h[i]
        for a in range(3):
            if np.isnan(r[a]) or np.isnan(h[a]): continue
            total += 1
            if np.sign(r[a]) * np.sign(h[a]) == -1:
                opposite += 1
    ratio = (opposite/total) if total else 0.0
    df.loc[len(df)] = ["부호반대비율", float(ratio), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    # 추가 1: 1-7 (BH~IMP) 절대합 요약
    seg_1_7 = [0,1,2,3,4,5]          # BH..IMP
    df.loc[len(df)] = ["1-7"] + abs_sum_rows(seg_1_7)

    # 추가 2: 1-9 (BH~FIN) 절대합 요약
    seg_1_9 = [0,1,2,3,4,5,6,7,8]    # BH..FIN
    df.loc[len(df)] = ["1-9"] + abs_sum_rows(seg_1_9)

    # 추가 3: 1-9 XYZ — BH~FIN의 X+Y+Z 절대합을 축 합계로 스칼라 표시
    R_vec = np.nansum(np.abs(T_r[seg_1_9]), axis=0)  # [Σ|X|, Σ|Y|, Σ|Z|]
    H_vec = np.nansum(np.abs(T_h[seg_1_9]), axis=0)
    R_xyz = float(np.nansum(R_vec))                  # 프로 XYZ 합(스칼라)
    H_xyz = float(np.nansum(H_vec))                  # 일반 XYZ 합(스칼라)
    D_xyz = abs(R_xyz - H_xyz)

    df.loc[len(df)] = ["1-9 XYZ", R_xyz, np.nan, np.nan, H_xyz, np.nan, np.nan, D_xyz, np.nan, np.nan]

    return df


def _mk_opposite_table_tau(T_r: np.ndarray, T_h: np.ndarray) -> pd.DataFrame:
    rows = []
    frame_labels = FRAMES_FULL[1:]  # BH..FIN
    for i, name in enumerate(frame_labels):
        for axis, ax_name in enumerate(["X","Y","Z"]):
            r, h = T_r[i,axis], T_h[i,axis]
            if np.isnan(r) or np.isnan(h): continue
            if np.sign(r) * np.sign(h) == -1:
                rows.append([name, ax_name, float(r), float(h), abs(float(r)-float(h))])
    if not rows:
        return pd.DataFrame(columns=["Frame","Axis","Pro","Ama","|Diff|"])
    df = (pd.DataFrame(rows, columns=["Frame","Axis","Pro","Ama","|Diff|"])
            .sort_values("|Diff|", ascending=False, ignore_index=True))
    return df


def _mk_same_top3_tau(T_r: np.ndarray, T_h: np.ndarray) -> pd.DataFrame:
    rows = []
    frame_labels = FRAMES_FULL[1:]  # BH..FIN
    for i, name in enumerate(frame_labels):
        for axis, ax_name in enumerate(["X","Y","Z"]):
            r, h = T_r[i,axis], T_h[i,axis]
            if np.isnan(r) or np.isnan(h): continue
            if np.sign(r) * np.sign(h) >= 0:
                rows.append([name, ax_name, float(r), float(h), abs(float(r)-float(h))])
    if not rows:
        return pd.DataFrame(columns=["Frame","Axis","Pro","Ama","|Diff|"])
    df = (pd.DataFrame(rows, columns=["Frame","Axis","Pro","Ama","|Diff|"])
            .sort_values("|Diff|", ascending=False, ignore_index=True)
            .head(3))
    return df

# ─────────────────────── 외부 호출 API ───────────────────────
def build_torque_tables(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    *,
    part: str = "knee",        # "knee" | "pelvis" | "shoulder"
    mass: float = 60.0,
) -> TorqueResult:
    """
    요약은 전부 '절대값 합(abs sum)' 기준으로 계산됩니다.
    """
    if part not in PARTS_TORQUE:
        raise ValueError(f"지원하지 않는 part: {part} (knee|pelvis|shoulder)")

    spec = PARTS_TORQUE[part]
    times_p = extract_times_10(pro_arr)   # (10,)
    times_a = extract_times_10(ama_arr)   # (10,)

    center_p = _series_LR_mean(pro_arr, spec["center"], 1, 10)  # (10,3)
    target_p = _series_LR_mean(pro_arr, spec["target"], 1, 10)  # (10,3)
    center_a = _series_LR_mean(ama_arr, spec["center"], 1, 10)
    target_a = _series_LR_mean(ama_arr, spec["target"], 1, 10)

    # τ(토크) 시리즈 (BH..FIN 9행)
    T_p = _torque_series(center_p, target_p, times_p, mass)
    T_a = _torque_series(center_a, target_a, times_a, mass)

    main   = _mk_main_table_tau(T_p, T_a)   # ← abs sum 고정
    opp    = _mk_opposite_table_tau(T_p, T_a)
    same3  = _mk_same_top3_tau(T_p, T_a)

    # 엑셀 행 전체 하이라이트 전달
    if len(opp):
        opp.attrs["yellow_rows"] = list(range(len(opp)))
    if len(same3):
        same3.attrs["yellow_rows"] = list(range(len(same3)))

    return TorqueResult(main, opp, same3)
