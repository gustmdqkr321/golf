# sections/swing_error/features/_3LH.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 엑셀 좌표 유틸 ──────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def col_series(arr: np.ndarray, letters: str, start: int, end: int) -> np.ndarray:
    i = col_letters_to_index(letters)
    s = arr[start - 1 : end, i]
    return pd.to_numeric(pd.Series(s, dtype="object"), errors="coerce").astype(float).to_numpy()

# ── 중심점 (좌/우 평균) ──────────────────────────────────────────────────────
def _center_xyz(arr: np.ndarray, start: int, end: int,
                left: tuple[str,str,str], right: tuple[str,str,str]) -> np.ndarray:
    L = np.stack([col_series(arr, left[0], start, end),
                  col_series(arr, left[1], start, end),
                  col_series(arr, left[2], start, end)], axis=1)
    R = np.stack([col_series(arr, right[0], start, end),
                  col_series(arr, right[1], start, end),
                  col_series(arr, right[2], start, end)], axis=1)
    return (L + R) * 0.5  # (N,3)

# heel: (BY,BZ,CA) & (CK,CL,CM)
def _heel_center(arr, s, e):   return _center_xyz(arr, s, e, ("BY","BZ","CA"), ("CK","CL","CM"))
# knee: (BP,BQ,BR) & (CB,CC,CD)
def _knee_center(arr, s, e):   return _center_xyz(arr, s, e, ("BP","BQ","BR"), ("CB","CC","CD"))
# pelvis: (H,I,J) & (K,L,M)
def _pelvis_center(arr, s, e): return _center_xyz(arr, s, e, ("H","I","J"), ("K","L","M"))

# ── Leg Hinge 각도 (heel–knee vs pelvis–knee) ───────────────────────────────
def _leg_hinge_angles(arr: np.ndarray, start: int = 1, end: int = 10) -> np.ndarray:
    A = _heel_center(arr, start, end)     # lower
    B = _knee_center(arr, start, end)     # middle
    C = _pelvis_center(arr, start, end)   # upper

    v_low  = A - B
    v_up   = C - B

    dot = np.einsum("ij,ij->i", v_low, v_up)
    n1  = np.linalg.norm(v_low, axis=1)
    n2  = np.linalg.norm(v_up,  axis=1)
    denom = n1 * n2
    denom = np.where(denom == 0, np.nan, denom)

    cos = np.clip(dot / denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))      # 0~180 deg
    return ang

def _frame_labels(start: int, end: int, labels: list[str] | None = None) -> list[str]:
    nums = list(range(start, end + 1))
    if labels and len(labels) >= len(nums):
        return [f"{n} ({labels[i]})" for i, n in enumerate(nums)]
    return [str(n) for n in nums]

# ── 단일 리포트 표 ───────────────────────────────────────────────────────────
def build_leg_hinge_report(arr: np.ndarray, start: int = 1, end: int = 10,
                           labels: list[str] | None = None) -> pd.DataFrame:
    theta = _leg_hinge_angles(arr, start, end)                 # (N,)
    dseg  = np.diff(theta, prepend=theta[0]); dseg[0] = 0.0

    nums   = list(range(start, end + 1))
    frames = _frame_labels(start, end, labels)

    rows = [[nums[i], frames[i], round(theta[i], 2), round(dseg[i], 2)]
            for i in range(len(theta))]

    # 섹션 합계 (1→4, 4→7, 1→7)
    backswing = round(theta[4-1] - theta[1-1], 2)
    downswing = round(theta[7-1] - theta[4-1], 2)
    total_1_7 = round(theta[7-1] - theta[1-1], 2)

    rows += [
        [101, "Backswing (1-4)", np.nan, backswing],
        [102, "Downswing (4-7)", np.nan, downswing],
        [103, "Total (1-7)",     np.nan, total_1_7],
    ]

    df = pd.DataFrame(rows, columns=["_ord","Frame","Leg Hinge (deg)","Section Change (deg)"])
    df = df.sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True)
    return df

# ── 프로/일반 비교표 ─────────────────────────────────────────────────────────
def build_leg_hinge_compare(pro_arr: np.ndarray, ama_arr: np.ndarray,
                            start: int = 1, end: int = 10,
                            labels: list[str] | None = None,
                            pro_name: str = "프로", ama_name: str = "일반") -> pd.DataFrame:
    p = build_leg_hinge_report(pro_arr, start, end, labels)
    a = build_leg_hinge_report(ama_arr, start, end, labels)

    p.columns = ["Frame", f"{pro_name} Leg Hinge (deg)", f"{pro_name} Section Change (deg)"]
    a.columns = ["Frame", f"{ama_name} Leg Hinge (deg)", f"{ama_name} Section Change (deg)"]

    # 정렬키
    def _ord_key(s: str) -> int:
        if str(s).startswith("Backswing"): return 101
        if str(s).startswith("Downswing"): return 102
        if str(s).startswith("Total"):     return 103
        try:
            return int(str(s).split()[0])
        except:
            return 999

    p["_ord"] = p["Frame"].map(_ord_key)

    df = (
        p.merge(a, on="Frame", how="outer")
         .sort_values("_ord", kind="stable")
         .drop(columns="_ord")
         .reset_index(drop=True)
    )

    # 차이(프로-일반)
    def _sub(x, y):
        return np.nan if (pd.isna(x) or pd.isna(y)) else float(x) - float(y)

    df["Leg Hinge Δ(프로-일반)"] = [
        _sub(x, y) for x, y in zip(
            df[f"{pro_name} Leg Hinge (deg)"], df[f"{ama_name} Leg Hinge (deg)"]
        )
    ]
    df["Section Change Δ(프로-일반)"] = [
        _sub(x, y) for x, y in zip(
            df[f"{pro_name} Section Change (deg)"], df[f"{ama_name} Section Change (deg)"]
        )
    ]

    # 숫자 컬럼 강제 숫자화
    num_cols = [
        f"{pro_name} Leg Hinge (deg)",
        f"{pro_name} Section Change (deg)",
        f"{ama_name} Leg Hinge (deg)",
        f"{ama_name} Section Change (deg)",
        "Leg Hinge Δ(프로-일반)",
        "Section Change Δ(프로-일반)",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── SD(1–8) 계산: '1', '1 (ADD)' 등 숫자 앞부분 추출해서 1~8 범위 필터 ──
    frame_num = (
        df["Frame"].astype(str).str.extract(r'^(\d+)')[0].astype(float)
    )
    mask_1_8 = frame_num.between(1, 8, inclusive="both")

    sd_lh_pro   = float(np.nanstd(df.loc[mask_1_8, f"{pro_name} Leg Hinge (deg)"],     ddof=0))
    sd_sc_pro   = float(np.nanstd(df.loc[mask_1_8, f"{pro_name} Section Change (deg)"], ddof=0))
    sd_lh_ama   = float(np.nanstd(df.loc[mask_1_8, f"{ama_name} Leg Hinge (deg)"],     ddof=0))
    sd_sc_ama   = float(np.nanstd(df.loc[mask_1_8, f"{ama_name} Section Change (deg)"], ddof=0))
    sd_lh_delta = float(np.nanstd(df.loc[mask_1_8, "Leg Hinge Δ(프로-일반)"],          ddof=0))
    sd_sc_delta = float(np.nanstd(df.loc[mask_1_8, "Section Change Δ(프로-일반)"],      ddof=0))

    # SD 요약행 추가 (Frame 칸에 라벨)
    df.loc[len(df)] = [
        "SD (1-8)",
        sd_lh_pro,
        sd_sc_pro,
        sd_lh_ama,
        sd_sc_ama,
        sd_lh_delta,
        sd_sc_delta,
    ]

    return df
