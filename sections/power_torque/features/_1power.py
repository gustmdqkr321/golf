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

# 부위별 좌우 3축 컬럼(엑셀 컬럼명 기준)
PARTS: Dict[str, Dict[str, List[str]]] = {
    "knee":     {"L": ["BP", "BQ", "BR"], "R": ["CB", "CC", "CD"]},
    "pelvis":   {"L": ["H",  "I",  "J" ], "R": ["K",  "L",  "M" ]},
    "shoulder": {"L": ["AL", "AM", "AN"], "R": ["BA", "BB", "BC"]},
    "wrist":    {"L": ["AX", "AY", "AZ"], "R": ["BM", "BN", "BO"]},
    # 단일 포인트(클럽헤드) — 좌우 동일 컬럼 사용
    "clubhead": {"L": ["CN", "CO", "CP"], "R": ["CN", "CO", "CP"]},
}

# 테이블에 표시하는 프레임 라벨(ADD~FH2)
FRAMES = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2"]  # 9 labels

@dataclass
class ForceResult:
    table_main: pd.DataFrame
    table_opposite: pd.DataFrame
    table_same_top3: pd.DataFrame

# ─────────────────────── 헬퍼: 좌표/시간 ───────────────────────
def extract_times(arr: np.ndarray) -> np.ndarray:
    """B열 시간(초), 1..10 프레임(ADD~Finish)"""
    return np.array([g(arr, f"B{t}") for t in range(1, 11)], dtype=float)  # 10개

def _center_series(arr: np.ndarray, part: str, cm_to_m: float = 0.01) -> np.ndarray:
    """
    part이 좌/우(L/R)면 평균을, 단일(C)이면 그대로 좌표를 사용.
    1..10 프레임(ADD~Finish), cm→m, ADD를 원점으로 평행이동.
    """
    p = PARTS[part]
    C = []
    for t in range(1, 11):  # 10개 (ADD~Finish)
        if "L" in p and "R" in p:
            L = np.array([g(arr, f"{p['L'][0]}{t}"), g(arr, f"{p['L'][1]}{t}"), g(arr, f"{p['L'][2]}{t}")], dtype=float)
            R = np.array([g(arr, f"{p['R'][0]}{t}"), g(arr, f"{p['R'][1]}{t}"), g(arr, f"{p['R'][2]}{t}")], dtype=float)
            C.append((L + R) / 2.0)
        elif "C" in p:  # 단일 포인트
            P = np.array([g(arr, f"{p['C'][0]}{t}"), g(arr, f"{p['C'][1]}{t}"), g(arr, f"{p['C'][2]}{t}")], dtype=float)
            C.append(P)
        else:
            raise ValueError(f"PARTS['{part}'] 정의가 잘못되었습니다.")
    C = np.vstack(C) * cm_to_m      # (10,3)
    C = C - C[0]                    # ADD 원점 이동
    return C

# ─────────────────────── 핵심: 힘(시간반영) ───────────────────────
def _segment_forces(C: np.ndarray, times: np.ndarray | None, mass: float = 60.0) -> np.ndarray:
    """
    C: (10,3) m, times: (10,) sec or None
    v_s = (C_{s+1}-C_s) / Δt_s          (s=1..9)
    표용 F(0..8): 
      - row0(ADD)=nan
      - row1..7 = m * 0.5*(v_s + v_{s+1})  [s=1..7 → BH..IMP]
      - row8(FH2)= m * 0.5*(v_8 + v_9)     [Finish 반영]
    """
    if times is None:
        dt = np.ones(9, dtype=float)   # 9 간격
    else:
        times = times.astype(float)
        dt = np.diff(times)            # (9,)
        dt[dt == 0] = 1.0

    v = (C[1:] - C[:-1]) / dt[:, None]   # (9,3) v1..v9

    F = np.full((9, 3), np.nan, dtype=float)
    for s in range(1, 8):               # 1..7
        v_avg = 0.5 * (v[s-1] + v[s])
        F[s] = mass * v_avg
    F[8] = mass * 0.5 * (v[7] + v[8])
    return F

# ─────────────────────── 표 생성 ───────────────────────
def _mk_main_table(F_r: np.ndarray, F_h: np.ndarray) -> pd.DataFrame:
    """요약 행은 모두 '절대값 합(abs sum)' 기준."""
    rows = []
    for i, name in enumerate(FRAMES):
        r = F_r[i]; h = F_h[i]
        opp = np.sign(r) * np.sign(h) == -1
        def mark(v, flag):
            return f"{fmt(v)}{' ❗' if (isinstance(flag, np.ndarray) and flag.any()) or (isinstance(flag, (bool, np.bool_)) and flag) else ''}"
        rows.append([
            name,
            fmt(r[0]), fmt(r[1]), fmt(r[2]),
            mark(h[0], opp[0]), mark(h[1], opp[1]), mark(h[2], opp[2]),
            fmt(abs(r[0]-h[0])), fmt(abs(r[1]-h[1])), fmt(abs(r[2]-h[2])),
        ])
    df = pd.DataFrame(rows, columns=["Frame","pro_X","pro_Y","pro_Z","ama_X","ama_Y","ama_Z","Diff_X","Diff_Y","Diff_Z"])

    # 프레임 인덱스 맵
    idx = {name:i for i,name in enumerate(FRAMES)}

    # 요약 1-4 / 4-7 / 7-9 (절대합)
    segs = {
        "요약 1-4": [idx["BH"], idx["BH2"], idx["TOP"]],
        "요약 4-7": [idx["TR"], idx["DH"], idx["IMP"]],
        "요약 7-9": [idx["FH1"], idx["FH2"]],
    }

    def abs_sum_rows(rows_idx: List[int]) -> List[str]:
        R = np.nansum(np.abs(F_r[rows_idx]), axis=0)  # pro XYZ |.| 합
        H = np.nansum(np.abs(F_h[rows_idx]), axis=0)  # ama XYZ |.| 합
        D = np.abs(R - H)                              # (차이의 절대값) = |Σ|pro| - Σ|ama||
        return [fmt(R[0]), fmt(R[1]), fmt(R[2]),
                fmt(H[0]), fmt(H[1]), fmt(H[2]),
                fmt(D[0]), fmt(D[1]), fmt(D[2])]

    for title, idxs in segs.items():
        df.loc[len(df)] = [title] + abs_sum_rows(idxs)

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

    # ✅ 요청한 2줄 추가: 1-7, 1-9 (절대값 합산)
    seg_1_7 = [idx["BH"], idx["BH2"], idx["TOP"], idx["TR"], idx["DH"], idx["IMP"]]        # 2(BH)~7(IMP)
    seg_1_9 = seg_1_7 + [idx["FH1"], idx["FH2"]]                                            # 2(BH)~9(FH2)
    df.loc[len(df)] = ["1-7"] + abs_sum_rows(seg_1_7)
    df.loc[len(df)] = ["1-9"] + abs_sum_rows(seg_1_9)

    # ✅ 한 줄 더: 2~9행(X+Y+Z) 총합을 단일 값으로(프로/일반/차이)
    R_vec = np.nansum(np.abs(F_r[seg_1_9]), axis=0)   # pro의 [Σ|X|, Σ|Y|, Σ|Z|]
    H_vec = np.nansum(np.abs(F_h[seg_1_9]), axis=0)   # ama의 [Σ|X|, Σ|Y|, Σ|Z|]
    R_xyz = float(np.nansum(R_vec))                   # pro XYZ 총합(스칼라)
    H_xyz = float(np.nansum(H_vec))                   # ama XYZ 총합(스칼라)
    D_xyz = abs(R_xyz - H_xyz)                        # 총합 차이(스칼라)

    # 스칼라를 X열에만 배치하고 나머지 축 칸은 공란으로 둠
    df.loc[len(df)] = ["1-9 XYZ", fmt(R_xyz), "", "", fmt(H_xyz), "", "", fmt(D_xyz), "", ""]
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
        return pd.DataFrame(columns=["Frame","Axis","pro","ama","|Diff|"])
    return (pd.DataFrame(rows, columns=["Frame","Axis","pro","ama","|Diff|"])
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
        return pd.DataFrame(columns=["Frame","Axis","pro","ama","|Diff|"])
    return (pd.DataFrame(rows, columns=["Frame","Axis","pro","ama","|Diff|"])
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
) -> ForceResult:
    """
    요약은 전부 '절대값 합(abs sum)' 기준으로 계산됩니다.
    """
    C_r = _center_series(pro_arr, part)   # (10,3)
    C_h = _center_series(ama_arr, part)   # (10,3)

    if times_pro is None: times_pro = extract_times(pro_arr)  # (10,)
    if times_ama is None: times_ama = extract_times(ama_arr)  # (10,)

    F_r = _segment_forces(C_r, times_pro, mass=mass)  # (9,3) — ADD..FH2
    F_h = _segment_forces(C_h, times_ama, mass=mass)  # (9,3)

    main   = _mk_main_table(F_r, F_h)   # ← abs_sum 고정
    opp    = _mk_opposite_table(F_r, F_h)
    same3  = _mk_same_top3(F_r, F_h)
    return ForceResult(main, opp, same3)
