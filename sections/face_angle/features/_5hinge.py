from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# A1 셀 유틸
# ──────────────────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def _g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    try:
        return float(arr[num-1, _col_idx(letters)])
    except Exception:
        return float("nan")

# ──────────────────────────────────────────────────────────────────────
# Hinging 계산 (1프레임)
# θ = -arcsin( cross_z( AW_xy , WC_xy ) / (|AW|·|WC|) )
#   AW = W - A = (AX-AR, AY-AS, AZ-AT)
#   WC = C - W = (CN-AX, CO-AY, CP-AZ)
#   cross_z( (x1,y1),(x2,y2) ) = x1*y2 - y1*x2
# ──────────────────────────────────────────────────────────────────────
def _hinging_angle(AX, AY, AZ, AR, AS, AT, CN, CO, CP) -> float:
    num = (AX - AR) * (CO - AY) - (AY - AS) * (CN - AX)
    len_aw = math.sqrt((AX-AR)**2 + (AY-AS)**2 + (AZ-AT)**2)
    len_wc = math.sqrt((CN-AX)**2 + (CO-AY)**2 + (CP-AZ)**2)
    denom = (len_aw * len_wc) if (len_aw and len_wc) else 0.0
    ratio = (num / denom) if denom != 0.0 else 0.0
    ratio = max(-1.0, min(1.0, ratio))             # clamp
    return float(round(-math.degrees(math.asin(ratio)), 2))

def _compute_hinging_series(arr: np.ndarray) -> list[float]:
    out: list[float] = []
    for i in range(1, 11):
        AR, AS, AT = _g(arr, f"AR{i}"), _g(arr, f"AS{i}"), _g(arr, f"AT{i}")
        AX, AY, AZ = _g(arr, f"AX{i}"), _g(arr, f"AY{i}"), _g(arr, f"AZ{i}")
        CN, CO, CP = _g(arr, f"CN{i}"), _g(arr, f"CO{i}"), _g(arr, f"CP{i}")
        out.append(_hinging_angle(AX, AY, AZ, AR, AS, AT, CN, CO, CP))
    return out

# ──────────────────────────────────────────────────────────────────────
# 유지지수(서명 유지)
# ──────────────────────────────────────────────────────────────────────
def _signed_maintenance_score(top: float, dh: float) -> float:
    delta = dh - top
    drop_ratio = abs(delta)/abs(top) if top != 0 else 0.0
    if math.copysign(1.0, top) == math.copysign(1.0, dh) and abs(dh) <= abs(top):
        return float(round((1 - drop_ratio) * 100.0, 2))
    else:
        return float(round(-drop_ratio * 100.0, 2))

# ──────────────────────────────────────────────────────────────────────
# 유사도 (0~100) — 사용자가 준 hinging_sim 규칙과 동일
#   - 부호 다르면 차이에 alpha배 가중
#   - sim = 100 - ( total / (n * max_diff) ) * 100
# ──────────────────────────────────────────────────────────────────────
def _hinging_similarity(r_vals: np.ndarray, h_vals: np.ndarray, alpha: float = 2.0) -> float:
    R = np.asarray(r_vals, dtype=float)
    H = np.asarray(h_vals, dtype=float)

    # NaN 제거(공통 유효 구간만 비교)
    mask = ~(np.isnan(R) | np.isnan(H))
    R, H = R[mask], H[mask]
    if R.size == 0:
        return 100.0

    same = np.sign(R) == np.sign(H)
    diffs = np.abs(R - H)
    weighted = np.where(same, diffs, alpha * diffs)

    total = float(weighted.sum())
    maxd  = float(weighted.max()) if weighted.size > 0 else 0.0
    n     = int(weighted.size)

    if n == 0 or maxd == 0.0:
        return 100.0
    sim = 100.0 - (total / (n * maxd)) * 100.0
    return float(round(sim, 2))

# ──────────────────────────────────────────────────────────────────────
# 공개 함수: 프로/일반 배열로 힌징 비교표 생성 + 유사도
# ──────────────────────────────────────────────────────────────────────
def build_hinging_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray, alpha: float = 2.0) -> pd.DataFrame:
    """
    반환 형식 (index: 1..10, '1-4','4-6','Hinging_Maintenance','Hinging_Similarity'):
      columns = [
        'Rory Hinging(°)', 'ΔRory(°)',
        'Hong Hinging(°)', 'ΔHong(°)',
        'Similarity(0-100)'
      ]
    """
    r = _compute_hinging_series(pro_arr)
    h = _compute_hinging_series(ama_arr)

    # 프레임 간 Δ (첫 행 NaN)
    r_d = [np.nan] + [round(r[i] - r[i-1], 2) for i in range(1, 10)]
    h_d = [np.nan] + [round(h[i] - h[i-1], 2) for i in range(1, 10)]

    # 구간 Δ (1-4: ADD→TOP, 4-6: TOP→DH)  ※ 인덱스 0,3,5
    top_idx, dh_idx = 3, 5
    r_1_4 = float(round(r[top_idx] - r[0], 2))
    r_4_6 = float(round(r[dh_idx] - r[top_idx], 2))
    h_1_4 = float(round(h[top_idx] - h[0], 2))
    h_4_6 = float(round(h[dh_idx] - h[top_idx], 2))

    # 유지지수
    r_keep = _signed_maintenance_score(r[top_idx], r[dh_idx])
    h_keep = _signed_maintenance_score(h[top_idx], h[dh_idx])

    # 유사도(전체 1~10 힌징값)
    sim = _hinging_similarity(np.array(r, float), np.array(h, float), alpha=alpha)

    # 표 조립
    idx = list(range(1, 11)) + ["1-4", "4-6", "Hinging_Maintenance", "Hinging_Similarity"]
    data = {
        "pro Hinging(°)": r + [r_1_4, r_4_6, np.nan, np.nan],
        "Δpro(°)":        r_d + [np.nan, np.nan, r_keep, np.nan],
        "ama Hinging(°)": h + [h_1_4, h_4_6, np.nan, np.nan],
        "Δama(°)":        h_d + [np.nan, np.nan, h_keep, np.nan],
        "Similarity(0-100)": [np.nan]*10 + [np.nan, np.nan, np.nan, sim],
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Frame"
    return df
