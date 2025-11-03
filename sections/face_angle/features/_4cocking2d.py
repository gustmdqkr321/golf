from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# A1 셀 접근 유틸
# ─────────────────────────────────────────
_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    m = _CELL.match(code.strip())
    if not m:
        return float("nan")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    try:
        return float(arr[row, col])
    except Exception:
        return float("nan")

# ─────────────────────────────────────────
# YZ 평면 각도 계산 (프레임 1~10)
#   elbow→wrist(YZ) vs clubhead→wrist(YZ) 사이 각도(°)
# ─────────────────────────────────────────
def compute_yz_plane_angles_from_array(arr: np.ndarray) -> list[float]:
    """
    각 프레임 n=1..10에 대해 YZ 평면에서:
      v_elbow = (ASn-AYn, ATn-AZn)
      v_club  = (COn-AYn, CPn-AZn)
      angle   = arccos( dot(v1,v2) / (|v1||v2|) ) [deg]
    """
    out: list[float] = []
    for n in range(1, 11):
        AS, AT = g(arr, f"AS{n}"), g(arr, f"AT{n}")  # elbow Y,Z
        AY, AZ = g(arr, f"AY{n}"), g(arr, f"AZ{n}")  # wrist Y,Z
        CO, CP = g(arr, f"CO{n}"), g(arr, f"CP{n}")  # club Y,Z

        v1 = np.array([AS - AY, AT - AZ], dtype=float)
        v2 = np.array([CO - AY, CP - AZ], dtype=float)

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            out.append(float("nan"))
            continue

        cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = math.degrees(math.acos(cos_t))
        out.append(ang)
    return out

# ─────────────────────────────────────────
# 프로/일반 비교표
# ─────────────────────────────────────────
def build_yz_plane_compare_table(
    pro_arr: np.ndarray, ama_arr: np.ndarray
) -> pd.DataFrame:
    """
    반환: columns = ["seg","Frame","프로","일반","차이(프로-일반)"]
      seg 라벨: ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
    """
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

    p_list = compute_yz_plane_angles_from_array(pro_arr)  # 길이 10
    a_list = compute_yz_plane_angles_from_array(ama_arr)  # 길이 10

    rows = []
    for i in range(10):
        p = p_list[i] if i < len(p_list) else float("nan")
        a = a_list[i] if i < len(a_list) else float("nan")
        rows.append([labels[i], p, a, p - a])

    df = pd.DataFrame(rows, columns=["seg","프로","일반","차이(프로-일반)"])

    # 숫자 컬럼 강제 숫자화(스타일/연산 안정)
    for c in ["프로","일반","차이(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

