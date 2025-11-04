# sections/swing_error/features/_2BH.py
from __future__ import annotations
import numpy as np
import pandas as pd
import math

# ── 엑셀 좌표 유틸 (A..Z, AA.. → 0-index) ───────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def col_series(arr: np.ndarray, letters: str, start: int, end: int) -> np.ndarray:
    """행 start..end (1-based) 범위의 컬럼 값을 1D(np.float64)로 반환"""
    i = col_letters_to_index(letters)
    s = arr[start - 1 : end, i]
    return pd.to_numeric(pd.Series(s, dtype="object"), errors="coerce").astype(float).to_numpy()

# ── 중심점 (좌/우 평균) ──────────────────────────────────────────────────────
def _center_xyz(arr: np.ndarray, start: int, end: int,
                left: tuple[str, str, str], right: tuple[str, str, str]) -> np.ndarray:
    L = np.stack([col_series(arr, left[0], start, end),
                  col_series(arr, left[1], start, end),
                  col_series(arr, left[2], start, end)], axis=1)
    R = np.stack([col_series(arr, right[0], start, end),
                  col_series(arr, right[1], start, end),
                  col_series(arr, right[2], start, end)], axis=1)
    return (L + R) * 0.5  # (N, 3)

def _shoulder_center(arr, s, e):  # (AL,AM,AN) & (BA,BB,BC)
    return _center_xyz(arr, s, e, ("AL", "AM", "AN"), ("BA", "BB", "BC"))

def _pelvis_center(arr, s, e):    # (H,I,J) & (K,L,M)
    return _center_xyz(arr, s, e, ("H", "I", "J"), ("K", "L", "M"))

def _knee_center(arr, s, e):      # (BP,BQ,BR) & (CB,CC,CD)
    return _center_xyz(arr, s, e, ("BP", "BQ", "BR"), ("CB", "CC", "CD"))

# ── Body Hinge 각도 (어깨-골반 vs 무릎-골반 벡터 사이 각) ────────────────────
def _body_hinge_angles(arr: np.ndarray, start: int = 1, end: int = 10) -> np.ndarray:
    S = _shoulder_center(arr, start, end)  # (N,3)
    P = _pelvis_center(arr, start, end)    # (N,3)
    K = _knee_center(arr, start, end)      # (N,3)

    U = S - P  # upper (어깨→골반)
    L = K - P  # lower (무릎→골반)

    dot = np.einsum("ij,ij->i", U, L)
    nu  = np.linalg.norm(U, axis=1)
    nl  = np.linalg.norm(L, axis=1)
    denom = nu * nl
    denom = np.where(denom == 0, np.nan, denom)

    cos = np.clip(dot / denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))  # (N,)
    return ang

def _frame_labels(start: int, end: int, labels: list[str] | None = None) -> list[str]:
    frames = list(range(start, end + 1))
    if labels and len(labels) >= len(frames):
        return [f"{f} ({labels[i]})" for i, f in enumerate(frames)]
    return [str(f) for f in frames]

# ── 단일표 (예시 모양) ───────────────────────────────────────────────────────
def build_body_hinge_report(arr: np.ndarray, start: int = 1, end: int = 10,
                            labels: list[str] | None = None) -> pd.DataFrame:
    theta = _body_hinge_angles(arr, start, end)             # (N,)
    dseg = np.diff(theta, prepend=theta[0]); dseg[0] = 0.0

    fr_col = _frame_labels(start, end, labels)
    frame_nos = list(range(start, end + 1))

    rows = [[frame_nos[i], fr_col[i], round(theta[i], 2), round(dseg[i], 2)]
            for i in range(len(theta))]

    # 섹션 합계
    backswing = round(theta[4-1] - theta[1-1], 2)  # 1→4
    downswing = round(theta[7-1] - theta[4-1], 2)  # 4→7
    total_1_7 = round(theta[7-1] - theta[1-1], 2)  # 1→7

    # 요약 행은 큰 정렬번호를 부여해 맨 아래로
    rows += [
        [101, "Backswing (1-4)", np.nan, backswing],
        [102, "Downswing (4-7)", np.nan, downswing],
        [103, "Total (1-7)",     np.nan, total_1_7],
    ]

    df = pd.DataFrame(rows, columns=["_ord", "Frame", "Body Hinge (deg)", "Section Change (deg)"])
    df = df.sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)
    return df


# ── 프로/일반 비교표 ─────────────────────────────────────────────────────────
def build_body_hinge_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                             start: int = 1, end: int = 10,
                             labels: list[str] | None = None,
                             pro_name: str = "프로", ama_name: str = "일반") -> pd.DataFrame:
    df_p = build_body_hinge_report(pro_arr, start, end, labels)
    df_a = build_body_hinge_report(ama_arr, start, end, labels)

    # 정렬용 키 (요약행은 고정 번호)
    def _ord_from_frame(s: str) -> int:
        if s.startswith("Backswing"): return 101
        if s.startswith("Downswing"): return 102
        if s.startswith("Total"):     return 103
        try:
            return int(s.split()[0])
        except Exception:
            return 999

    df_p["_ord"] = df_p["Frame"].map(_ord_from_frame)

    df_p.columns = [
        "Frame",
        f"{pro_name} Body Hinge (deg)",
        f"{pro_name} Section Change (deg)",
        "_ord",
    ]
    df_a.columns = [
        "Frame",
        f"{ama_name} Body Hinge (deg)",
        f"{ama_name} Section Change (deg)",
    ]

    df = df_p.merge(df_a, on="Frame", how="outer")
    df = df.sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)

    # 차이(프로-일반)
    def _sub(a, b):
        return np.nan if (pd.isna(a) or pd.isna(b)) else float(a) - float(b)

    df["Body Hinge Δ(프로-일반)"] = [
        _sub(a, b) for a, b in zip(
            df[f"{pro_name} Body Hinge (deg)"], df[f"{ama_name} Body Hinge (deg)"]
        )
    ]
    df["Section Change Δ(프로-일반)"] = [
        _sub(a, b) for a, b in zip(
            df[f"{pro_name} Section Change (deg)"], df[f"{ama_name} Section Change (deg)"]
        )
    ]

    # 숫자 컬럼 강제 숫자화
    num_cols = [
        f"{pro_name} Body Hinge (deg)",
        f"{pro_name} Section Change (deg)",
        f"{ama_name} Body Hinge (deg)",
        f"{ama_name} Section Change (deg)",
        "Body Hinge Δ(프로-일반)",
        "Section Change Δ(프로-일반)",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── SD(1–8) 계산 (프레임 행만 대상) ───────────────────────────────
    # "1", "1 (ADD)" 같은 형식 모두 지원: 앞쪽 숫자만 추출해서 1~8 범위 필터
    mask_1_8 = df["Frame"].astype(str).str.extract(r'^(\d+)')[0].astype(float).between(1, 8, inclusive="both")

    sd_bh_pro   = float(np.nanstd(df.loc[mask_1_8, f"{pro_name} Body Hinge (deg)"],     ddof=0))
    sd_sc_pro   = float(np.nanstd(df.loc[mask_1_8, f"{pro_name} Section Change (deg)"], ddof=0))
    sd_bh_ama   = float(np.nanstd(df.loc[mask_1_8, f"{ama_name} Body Hinge (deg)"],     ddof=0))
    sd_sc_ama   = float(np.nanstd(df.loc[mask_1_8, f"{ama_name} Section Change (deg)"], ddof=0))
    sd_bh_delta = float(np.nanstd(df.loc[mask_1_8, "Body Hinge Δ(프로-일반)"],          ddof=0))
    sd_sc_delta = float(np.nanstd(df.loc[mask_1_8, "Section Change Δ(프로-일반)"],      ddof=0))

    # 맨 아래 SD 요약행 추가
    df.loc[len(df)] = [
        "SD (1-8)",
        sd_bh_pro,
        sd_sc_pro,
        sd_bh_ama,
        sd_sc_ama,
        sd_bh_delta,
        sd_sc_delta,
    ]

    return df
