from __future__ import annotations
import math
import numpy as np
import pandas as pd
import re

# ─────────────────────────────────────────────
# 엑셀(A1) 접근 유틸
# ─────────────────────────────────────────────
_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[0, col('AX')]
    """
    m = _CELL.match(code.strip())
    if not m:
        return float("nan")
    col = _col_letters_to_index(m.group(1))
    row = int(m.group(2)) - 1
    try:
        return float(arr[row, col])
    except Exception:
        return float("nan")

# ─────────────────────────────────────────────
# Cocking 유사도 (사용자 규칙)
#  - Δ 시퀀스 2개 비교
#  - ADD(첫 원소)는 제외하고 나머지 구간만
#  - 방향 일치 50점 + 크기 유사도 50점
# ─────────────────────────────────────────────
def cocking_sim(rory_seq: np.ndarray, hong_seq: np.ndarray) -> tuple[float, list[float]]:
    R = np.asarray(rory_seq, dtype=float)
    H = np.asarray(hong_seq, dtype=float)
    if R.shape != H.shape:
        raise ValueError("rory_seq, hong_seq 는 같은 길이여야 합니다.")

    # 첫 구간(ADD)은 제외
    deltas_r = R[1:]
    deltas_h = H[1:]
    scores: list[float] = []

    for dr, dh in zip(deltas_r, deltas_h):
        # 방향 일치 점수
        if math.copysign(1.0, dr) == math.copysign(1.0, dh):
            dir_score = 50.0
            # 크기 점수
            maxc = max(abs(dr), abs(dh))
            if maxc == 0:
                size_score = 50.0
            else:
                size_diff = abs(dr - dh)
                size_score = 50.0 - (size_diff / maxc) * 50.0
        else:
            dir_score = 0.0
            size_score = 0.0
        scores.append(dir_score + size_score)

    overall = float(np.round(np.mean(scores) if scores else 0.0, 2))
    return overall, [float(np.round(s, 2)) for s in scores]

# ─────────────────────────────────────────────
# 핵심 계산
# ─────────────────────────────────────────────
def compute_angles_from_array(arr: np.ndarray) -> list[float]:
    """
    n=1..10 에 대해
      v1 = (ARn−AXn, ASn−AYn, ATn−AZn)
      v2 = (CNn−AXn, COn−AYn, CPn−AZn)
      θ  = ∠(v1, v2) [deg]
    """
    out: list[float] = []
    for n in range(1, 11):
        AR, AS, AT = g(arr, f"AR{n}"), g(arr, f"AS{n}"), g(arr, f"AT{n}")
        AX, AY, AZ = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")
        CN, CO, CP = g(arr, f"CN{n}"), g(arr, f"CO{n}"), g(arr, f"CP{n}")

        v1 = np.array([AR-AX, AS-AY, AT-AZ], dtype=float)
        v2 = np.array([CN-AX, CO-AY, CP-AZ], dtype=float)

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            out.append(float("nan"))
            continue

        cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        theta = math.degrees(math.acos(cos_t))
        out.append(theta)
    return out

def compute_cocking_table_from_arrays(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
) -> pd.DataFrame:
    """
    반환 형식(인덱스 제거, seg 컬럼 추가):
      columns = [
        'seg','Pro ∠ABC','Ama ∠ABC','Pro Δ(°)','Ama Δ(°)','Similarity(0–100)'
      ]
      seg = ['ADD','BH','BH2','TOP','TR','DH','IMP','FH1','FH2','FIN',
             '1-4','4-6','Cocking_Maintenance','Similarity']
    """
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

    # 1) 각 프레임 각도
    ang_p = compute_angles_from_array(pro_arr)
    ang_a = compute_angles_from_array(ama_arr)

    # 2) 프레임 간 Δ (ADD 위치는 0.0로 채움 → 유사도는 [1:]만 사용)
    d_p = [0.0] + [ang_p[i] - ang_p[i-1] for i in range(1, 10)]
    d_a = [0.0] + [ang_a[i] - ang_a[i-1] for i in range(1, 10)]

    # 3) 구간 합산/지표
    d1_4_p = ang_p[3] - ang_p[0] if np.isfinite(ang_p[3]) and np.isfinite(ang_p[0]) else np.nan
    d4_6_p = ang_p[5] - ang_p[3] if np.isfinite(ang_p[5]) and np.isfinite(ang_p[3]) else np.nan
    d1_4_a = ang_a[3] - ang_a[0] if np.isfinite(ang_a[3]) and np.isfinite(ang_a[0]) else np.nan
    d4_6_a = ang_a[5] - ang_a[3] if np.isfinite(ang_a[5]) and np.isfinite(ang_a[3]) else np.nan

    def _cm(ang: list[float]) -> float:
        if not (np.isfinite(ang[3]) and np.isfinite(ang[5])) or ang[3] == 0:
            return np.nan
        return (1.0 - abs(ang[5] - ang[3]) / abs(ang[3])) * 100.0

    cm_p = _cm(ang_p)
    cm_a = _cm(ang_a)

    # 4) 코킹 유사도 (전체)
    overall_sim, _seg_scores = cocking_sim(np.array(d_p, float), np.array(d_a, float))

    # 5) DF 조립 (seg 컬럼으로 라벨 제공)
    segs_all = labels + ["1-4", "4-6", "Cocking_Maintenance", "Similarity"]
    data = {
        "seg":               segs_all,
        "Pro ∠ABC":          ang_p + [d1_4_p, d4_6_p, np.nan, np.nan],
        "Ama ∠ABC":          ang_a + [d1_4_a, d4_6_a, np.nan, np.nan],
        "Pro Δ(°)":          [np.nan] + d_p[1:] + [np.nan, np.nan, cm_p, np.nan],
        "Ama Δ(°)":          [np.nan] + d_a[1:] + [np.nan, np.nan, cm_a, np.nan],
        "Similarity(0–100)": [np.nan]*10 + [np.nan, np.nan, np.nan, overall_sim],
    }
    df = pd.DataFrame(data)

    # 숫자 컬럼은 숫자로 강제(스타일/연산 안정)
    num_cols = ["Pro ∠ABC","Ama ∠ABC","Pro Δ(°)","Ama Δ(°)","Similarity(0–100)"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
