# sections/swing/features/_28_b4_b7.py
from __future__ import annotations
import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def build_b4_b7_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    항목: B4, (B7 - B4)
    컬럼: [항목, 프로, 일반, 차이(프로-일반)]
    """
    # 프로
    p_b4 = g(pro_arr, "B4")
    p_d  = g(pro_arr, "B7") - g(pro_arr, "B4")
    # 일반
    a_b4 = g(ama_arr, "B4")
    a_d  = g(ama_arr, "B7") - g(ama_arr, "B4")

    rows = [
        ["1/4",        round(p_b4, 2), round(a_b4, 2), round(p_b4 - a_b4, 2)],
        ["4/7",   round(p_d,  2), round(a_d,  2), round(p_d  - a_d,  2)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

def _swing_arc_length_polyline(CN, CO, CP, return_segment_lengths=False):
    CN = np.asarray(CN, dtype=float)
    CO = np.asarray(CO, dtype=float)
    CP = np.asarray(CP, dtype=float)
    if not (CN.shape == CO.shape == CP.shape):
        raise ValueError("CN, CO, CP 길이가 동일해야 합니다.")
    if CN.ndim != 1:
        raise ValueError("CN, CO, CP는 1차원 배열이어야 합니다.")
    if len(CN) < 2:
        return (0.0, np.array([])) if return_segment_lengths else 0.0

    dx = np.diff(CN); dy = np.diff(CO); dz = np.diff(CP)
    seg = np.sqrt(dx*dx + dy*dy + dz*dz)
    total = float(seg.sum())
    if return_segment_lengths:
        return total, seg
    return total

def _series(arr: np.ndarray, col: str, start: int = 1, end: int = 10) -> list[float]:
    return [g(arr, f"{col}{i}") for i in range(start, end+1)]

def _arc_total_and_segments(arr: np.ndarray, start: int = 1, end: int = 10):
    CN = _series(arr, "CN", start, end)
    CO = _series(arr, "CO", start, end)
    CP = _series(arr, "CP", start, end)
    total, seg = _swing_arc_length_polyline(CN, CO, CP, return_segment_lengths=True)
    return total, seg  # seg: length (end-start)

# ── 공개 API: 비교표(항목/프로/일반/차이) ───────────────────────────────────
def build_head_arc_polyline_table(pro_arr: np.ndarray,
                                  ama_arr: np.ndarray,
                                  *,
                                  start: int = 1,
                                  end: int = 10,
                                  include_segments: bool = True,
                                  ndigits: int = 2) -> pd.DataFrame:
    """
    클럽헤드(CN,CO,CP) 폴리라인 아크 길이 비교표 생성
      - 1행: 총 아크 길이 (start~end)
      - (옵션) 각 세그먼트 길이: 'i-(i+1)' 행들
    columns: ['항목','프로','일반','차이(프로-일반)']
    """
    p_total, p_seg = _arc_total_and_segments(pro_arr, start, end)
    a_total, a_seg = _arc_total_and_segments(ama_arr, start, end)

    rows = [
        ["총 아크 길이", round(p_total, ndigits), round(a_total, ndigits),
         round(p_total - a_total, ndigits)]
    ]

    if include_segments:
        for i in range(start, end):
            pi = float(p_seg[i-start])
            ai = float(a_seg[i-start])
            rows.append([f"{i}-{i+1}", round(pi, ndigits), round(ai, ndigits),
                         round(pi - ai, ndigits)])

    return pd.DataFrame(rows, columns=["항목","프로","일반","차이(프로-일반)"])