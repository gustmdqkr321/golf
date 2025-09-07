# sections/swing/features/rotation.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 셀 헬퍼 ──────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def _get_xyz_series(arr: np.ndarray, cols3: list[str], max_frame: int = 10) -> np.ndarray:
    """
    프레임 1..max_frame 의 3D 좌표열 반환.
    데이터가 더 적으면 NaN이 포함될 수 있으나, 이후 계산은 안전하게 진행됨.
    """
    out = []
    for t in range(1, max_frame + 1):   # ← 1..10
        out.append([g(arr, f"{cols3[0]}{t}"),
                    g(arr, f"{cols3[1]}{t}"),
                    g(arr, f"{cols3[2]}{t}")])
    return np.asarray(out, float)

# ── 각도 계산 ────────────────────────────────────────────────────────────────
def _horizontal_angle(a, b, c, d) -> np.ndarray:
    """XZ 평면에서 좌우벡터의 회전각 (deg), 구간 t→t+1"""
    ab = b - a
    cd = d - c
    ab_p = ab[:, [0, 2]]
    cd_p = cd[:, [0, 2]]
    cross = ab_p[:, 0]*cd_p[:, 1] - ab_p[:, 1]*cd_p[:, 0]
    dot   = np.einsum("ij,ij->i", ab_p, cd_p)
    return np.degrees(np.arctan2(cross, dot))

def _vertical_angle_slope(a, b, c, d) -> np.ndarray:
    """XZ 대비 Y 기울기 각(=수직회전), 구간 차(deg)"""
    ab = b - a
    cd = d - c
    denom_ab = np.linalg.norm(ab[:, [0, 2]], axis=1)
    denom_cd = np.linalg.norm(cd[:, [0, 2]], axis=1)
    denom_ab[denom_ab == 0] = 1e-9
    denom_cd[denom_cd == 0] = 1e-9
    slope_ab = ab[:, 1] / denom_ab
    slope_cd = cd[:, 1] / denom_cd
    ang_ab = np.degrees(np.arctan(slope_ab))
    ang_cd = np.degrees(np.arctan(slope_cd))
    return ang_cd - ang_ab

def _apply_sign_rules(h_raw: np.ndarray, v_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """수평: 1–3 +, 4–… − / 수직: 첫 구간 +, 부호 전환 시 토글"""
    h = h_raw.copy()
    h[:3] = np.abs(h[:3])
    h[3:] = -np.abs(h[3:])

    v = np.zeros_like(v_raw)
    v[0] = abs(v_raw[0])
    sign = 1
    for i in range(1, len(v_raw)):
        if v_raw[i] * v_raw[i-1] < 0:
            sign *= -1
        v[i] = abs(v_raw[i]) * sign
    return h, v

# ── 공개 API ────────────────────────────────────────────────────────────────
PARTS = {
    "waist":   {"L": ["H", "I", "J"],   "R": ["K", "L", "M"]},      # 골반
    "shoulder":{"L": ["AL","AM","AN"],  "R": ["BA","BB","BC"]},     # 어깨
}

def compute_rotation_table(arr: np.ndarray, part: str = "waist") -> pd.DataFrame:
    """
    반환: 구간(1-2..9-10), 수평회전각도(°), 수직회전각도(°) DataFrame (소수2자리)
    프레임은 1..10 기준. (10프레임 데이터가 없으면 해당 구간 값은 NaN이 될 수 있음)
    """
    p = PARTS[part]
    L = _get_xyz_series(arr, p["L"], max_frame=10)   # ← 1..10
    R = _get_xyz_series(arr, p["R"], max_frame=10)

    # 유효 프레임 수 파악(뒤쪽 완전 NaN은 잘라냄)
    def _trim_nan_tail(X: np.ndarray) -> int:
        n = X.shape[0]
        while n > 1 and np.isnan(X[n-1]).all():
            n -= 1
        return n

    nL = _trim_nan_tail(L)
    nR = _trim_nan_tail(R)
    n  = max(min(nL, nR), 2)  # 최소 2프레임 필요
    L, R = L[:n], R[:n]

    # 구간 t→t+1
    h_raw = _horizontal_angle(L[:-1], R[:-1], L[1:], R[1:])
    v_raw = _vertical_angle_slope(L[:-1], R[:-1], L[1:], R[1:])
    h, v  = _apply_sign_rules(h_raw, v_raw)

    seg = [f"{i}-{i+1}" for i in range(1, n)]
    df = pd.DataFrame({
        "구간": seg,
        "수평회전각도(°)": np.round(h, 2),
        "수직회전각도(°)": np.round(v, 2),
    })
    return df

def _add_rollups_for_rotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    rotation.compute_rotation_table() 결과에
    1–4, 4–7, 7–10, Total 행을 '절댓값 합'으로 추가.
    (데이터가 부족하면 가능한 구간만 합산)
    """
    def parse_pair(s: str) -> tuple[int, int]:
        a, b = s.split("-")
        return int(a), int(b)

    pairs = df["구간"].apply(parse_pair).to_list()  # [(1,2), (2,3), ...]
    start_ends = [(1, 4), (4, 7), (7, 10)]
    cols = ["수평회전각도(°)", "수직회전각도(°)"]

    def abs_sum_between(start: int, end: int, col: str) -> float:
        # 포함: 시작>=start, 끝<=end
        mask = [(a >= start and b <= end) for (a, b) in pairs]
        vals = pd.to_numeric(df.loc[mask, col], errors="coerce").abs()  # ← 절댓값 합
        return float(vals.sum())

    rows = []
    labels = ["1-4", "4-7", "7-10"]
    for (s, e), lab in zip(start_ends, labels):
        rows.append({
            "구간": lab,
            cols[0]: abs_sum_between(s, e, cols[0]),
            cols[1]: abs_sum_between(s, e, cols[1]),
        })

    total_row = {
        "구간": "Total",
        cols[0]: sum(r[cols[0]] for r in rows),
        cols[1]: sum(r[cols[1]] for r in rows),
    }

    rollup_df = pd.DataFrame(rows + [total_row])
    rollup_df[cols] = rollup_df[cols].round(2)

    out = pd.concat([df, rollup_df], ignore_index=True)
    return out

# --- 없으면 추가: 구간합(부호 유지) ---
def segment_sum_signed(df: pd.DataFrame, start: int, end: int) -> tuple[float,float,float]:
    """[start,end] 구간의 수평합, 수직합, (수평+수직) 반환"""
    if "구간" not in df.columns:
        return 0.0, 0.0, 0.0
    seg_mask = df["구간"].astype(str).str.contains("-")
    work = df.loc[seg_mask].copy()
    ij = work["구간"].str.split("-", expand=True).astype(int)
    mask = (ij[0] >= start) & (ij[1] <= end)
    cols = ["수평회전각도(°)", "수직회전각도(°)"]
    vals = work.loc[mask, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    h_sum = float(vals[cols[0]].sum())
    v_sum = float(vals[cols[1]].sum())
    return h_sum, v_sum, h_sum + v_sum

# --- 새 함수: 설명 없이 프로/일반 두 값만 ---
def build_rotation_spec_table_simple(pro_arr: np.ndarray, ama_arr: np.ndarray, *, start: int = 1, end: int = 4) -> pd.DataFrame:
    """메인표(1~3행) — 1-4 WAI/SHO와 Total, 컬럼은 프로/일반만"""
    # 골반/어깨 회전 테이블
    pro_wai = compute_rotation_table(pro_arr, part="waist")
    ama_wai = compute_rotation_table(ama_arr, part="waist")
    pro_sho = compute_rotation_table(pro_arr, part="shoulder")
    ama_sho = compute_rotation_table(ama_arr, part="shoulder")

    # 1–4 구간 합(수평+수직)
    _, _, p_wai = segment_sum_signed(pro_wai, start, end)
    _, _, a_wai = segment_sum_signed(ama_wai, start, end)
    _, _, p_sho = segment_sum_signed(pro_sho, start, end)
    _, _, a_sho = segment_sum_signed(ama_sho, start, end)

    rows = [
        ["1-4 WAI", round(p_wai, 2), round(a_wai, 2)],
        ["1-4 SHO", round(p_sho, 2), round(a_sho, 2)],
        ["Total",   round(p_wai + p_sho, 2), round(a_wai + a_sho, 2)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반"])

def _com_y_series(arr: np.ndarray) -> list[float]:
    """프레임 1..10의 Y_COM 시리즈(소수 2자리)"""
    out = []
    for n in range(1, 11):
        val = (
            0.09 * g(arr, f"AD{n}") +
            0.40 * (g(arr, f"AM{n}") + g(arr, f"BB{n}"))/2 +
            0.34 * (g(arr, f"I{n}")  + g(arr, f"L{n}"))/2 +
            0.17 * (g(arr, f"BQ{n}") + g(arr, f"CC{n}"))/2
        )
        out.append(round(val, 2))
    return out

def _com_z_series(arr: np.ndarray) -> list[float]:
    """프레임 1..10의 Z laterality 시리즈(소수 2자리)"""
    out = []
    for n in range(1, 11):
        knee  = (g(arr, f"BR{n}") + g(arr, f"CD{n}"))/2
        hip   = (g(arr, f"J{n}")  + g(arr, f"M{n}"))/2
        chest = (g(arr, f"AN{n}") + g(arr, f"BC{n}"))/2
        head  = g(arr, f"AE{n}")
        foot  = (g(arr, f"CA{n}") + g(arr, f"CM{n}"))/2
        val = ((knee + hip + chest + head)/4) - foot
        out.append(round(val, 2))
    return out

def _sum_abs_step_changes(vals: list[float]) -> float:
    """연속 프레임 간 절대 변화량의 합: |Δ1-2|+|Δ2-3|+..."""
    a = np.asarray(vals, dtype=float)
    m = np.isfinite(a[:-1]) & np.isfinite(a[1:])
    return float(np.abs(a[1:][m] - a[:-1][m]).sum().round(2))

def build_abs_1_10_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    반환: 항목/프로/일반 3열
      - 1-10|abs Y : Σ|ΔY| (1→2..9→10)
      - 1-10|abs Z : Σ|ΔZ| (1→2..9→10)
    """
    y_pro = _com_y_series(pro_arr);  y_ama = _com_y_series(ama_arr)
    z_pro = _com_z_series(pro_arr);  z_ama = _com_z_series(ama_arr)

    y_abs_pro  = round(_sum_abs_step_changes(y_pro), 2)
    y_abs_ama  = round(_sum_abs_step_changes(y_ama), 2)
    z_abs_pro  = round(_sum_abs_step_changes(z_pro), 2)
    z_abs_ama  = round(_sum_abs_step_changes(z_ama), 2)

    rows = [
        ["1-10|abs Y", y_abs_pro, y_abs_ama],
        ["1-10|abs Z", z_abs_pro, z_abs_ama],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반"])
