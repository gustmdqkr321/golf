from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── A1 셀 접근 ───────────────────────────────────────────────────────
_CELL = re.compile(r"^([A-Za-z]+)(\d+)$")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g_base(arr: np.ndarray, addr: str) -> float:
    m = _CELL.match(addr.strip())
    if not m: return float("nan")
    c = _col_idx(m.group(1)); r = int(m.group(2)) - 1
    try:    return float(arr[r, c])
    except Exception: return float("nan")

# ── 좌표 스택(1..10 프레임) ───────────────────────────────────────────────
_FRAMES = 10  # 1..10 → (1-2)~(9-10) 구간

def _stack_xyz(arr: np.ndarray, X: str, Y: str, Z: str) -> np.ndarray:
    pts = []
    for i in range(1, _FRAMES+1):
        pts.append([g_base(arr, f"{X}{i}"), g_base(arr, f"{Y}{i}"), g_base(arr, f"{Z}{i}")])
    return np.asarray(pts, dtype=float)  # (10,3)

# ── 회전각(수평/수직) 공통 함수 ────────────────────────────────────────────
def _horizontal_angle_series(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: (N,3) 상위 관절, B: (N,3) 하위 관절
    각 프레임 벡터 AB를 XZ 평면에 투영하고, (t→t+1) 회전각(도) 계산.
    반환: 길이 N-1 (1-2 … 9-10)
    """
    AB = B - A
    ab_proj = AB[:-1, [0, 2]]
    cd_proj = AB[1:,  [0, 2]]
    cross = ab_proj[:, 0]*cd_proj[:, 1] - ab_proj[:, 1]*cd_proj[:, 0]
    dot   = (ab_proj * cd_proj).sum(axis=1)
    ang   = np.degrees(np.arctan2(cross, dot))  # [-180,180]
    return ang

def _vertical_angle_series(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    수직회전(기울기) 각도: tilt_t = atan( ΔY / ||(ΔX,ΔZ)|| )[deg]
    dtilt = tilt_{t+1} - tilt_t → 길이 N-1
    """
    AB = B - A
    denom = np.linalg.norm(AB[:, [0, 2]], axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    tilt  = np.degrees(np.arctan(AB[:, 1]/denom))
    return tilt[1:] - tilt[:-1]

def _apply_sign_rules(horizontal_angles: np.ndarray, vertical_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    - 수평: 1-4(앞 3개) +|·|, 4-5 이후 -|·|
    - 수직: 첫 구간 +|·|에서 시작, 인접 구간 값 부호가 바뀌면 누적 부호 토글
    """
    # 수평
    h = horizontal_angles.copy()
    k = min(3, h.size)
    h[:k] = np.abs(h[:k])
    if h.size > 3:
        h[3:] = -np.abs(h[3:])

    # 수직
    v = vertical_angles.copy()
    if v.size > 0:
        v_signed = np.zeros_like(v)
        sign = 1
        v_signed[0] = abs(v[0])
        for i in range(1, v.size):
            if v[i] * v[i-1] < 0:
                sign *= -1
            v_signed[i] = abs(v[i]) * sign
        v = v_signed
    return h, v

def _series_with_segments(arr: np.ndarray,
                          X0: str, Y0: str, Z0: str,
                          X1: str, Y1: str, Z1: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    A = _stack_xyz(arr, X0, Y0, Z0)
    B = _stack_xyz(arr, X1, Y1, Z1)
    h = _horizontal_angle_series(A, B)
    v = _vertical_angle_series(A, B)
    h_s, v_s = _apply_sign_rules(h, v)
    labels = [f"{i}-{i+1}" for i in range(1, _FRAMES)]
    return h_s, v_s, labels

def _merge_pro_ama_with_segments(pro_arr: np.ndarray, ama_arr: np.ndarray,
                                 X0: str, Y0: str, Z0: str, X1: str, Y1: str, Z1: str
                                 ) -> pd.DataFrame:
    # 각 시리즈(길이 9) 산출
    hp, vp, lbl = _series_with_segments(pro_arr, X0, Y0, Z0, X1, Y1, Z1)
    ha, va, _   = _series_with_segments(ama_arr, X0, Y0, Z0, X1, Y1, Z1)

    df = pd.DataFrame({
        "구간": lbl,
        "수평(Pro)": np.round(hp, 2),
        "수평(Ama)": np.round(ha, 2),
        "수직(Pro)": np.round(vp, 2),
        "수직(Ama)": np.round(va, 2),
    })

    # ── 구간합/전체합 추가 ─────────────────────────────────────────────────
    # 인덱스: 1-2 → 0, 2-3 → 1, 3-4 → 2, 4-5 → 3, 5-6 → 4, 6-7 → 5, 7-8 → 6, 8-9 → 7, 9-10 → 8
    segs = [("1-4", slice(0, 3)), ("4-7", slice(3, 6)), ("7-10", slice(6, 9))]
    rows = []

    # 구간합(부호 유지)
    for name, sl in segs:
        rows.append({
            "구간": name,
            "수평(Pro)": np.round(np.nansum(hp[sl]), 2),
            "수평(Ama)": np.round(np.nansum(ha[sl]), 2),
            "수직(Pro)": np.round(np.nansum(vp[sl]), 2),
            "수직(Ama)": np.round(np.nansum(va[sl]), 2),
        })

    # Total = Σ|…| (절댓값 합)
    rows.append({
        "구간": "Total",
        "수평(Pro)": np.round(np.nansum(np.abs(hp)), 2),
        "수평(Ama)": np.round(np.nansum(np.abs(ha)), 2),
        "수직(Pro)": np.round(np.nansum(np.abs(vp)), 2),
        "수직(Ama)": np.round(np.nansum(np.abs(va)), 2),
    })

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

# ── 공개 API ────────────────────────────────────────────────────────────
# 1) 왼팔(왼어깨→왼손목)
def build_left_arm_rotation_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _merge_pro_ama_with_segments(base_pro, base_ama,
                                        "AL","AM","AN",  # 상위: 왼어깨
                                        "AX","AY","AZ")  # 하위: 왼손목

# 2) 클럽(왼손목→클럽헤드)
def build_club_rotation_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _merge_pro_ama_with_segments(base_pro, base_ama,
                                        "AX","AY","AZ",  # 상위: 왼손목
                                        "CN","CO","CP")  # 하위: 클럽헤드

# 3) 무릎(왼무릎→오른무릎)
def build_knee_rotation_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _merge_pro_ama_with_segments(
        base_pro, base_ama,
        "BP","BQ","BR",   # 상위: 왼무릎
        "CB","CC","CD"    # 하위: 오른무릎
    )

# 4) 골반(왼엉덩이→오른엉덩이)
def build_hip_rotation_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _merge_pro_ama_with_segments(
        base_pro, base_ama,
        "H","I","J",    # 상위: 왼골반
        "K","L","M"     # 하위: 오른골반
    )

# 5) 어깨(왼어깨→오른어깨)
def build_shoulder_rotation_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    return _merge_pro_ama_with_segments(
        base_pro, base_ama,
        "AL","AM","AN",   # 상위: 왼어깨
        "BA","BB","BC"    # 하위: 오른어깨
    )