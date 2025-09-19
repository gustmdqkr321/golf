# sections/<섹션>/features/_pitch.py
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd

# ── A1 주소 → 인덱스 / 값 읽기 유틸 ─────────────────────────────────────────
_COL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    m = _COL_RE.match(code.strip())
    if not m:
        raise ValueError(f"잘못된 주소: {code}")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    try:
        return float(arr[row, col])
    except Exception:
        return float("nan")

# ── Pitch 계산 (예시식 그대로) ─────────────────────────────────────────────
def compute_pitch_angles_from_array(arr: np.ndarray,
                                    frames: range | list[int] = range(1, 11)
                                   ) -> list[float]:
    """
    Frame n마다
      A = midpoint of (ALn,AMn,ANn) & (BAn,BBn,BCn)
      B = midpoint of (AXn,AYn,AZn) & (BMn,BNn,BOn)
    Pitch = atan2( (yB-yA), sqrt((xB-xA)^2 + (zB-zA)^2) ) [deg]
    """
    vals: list[float] = []
    for n in frames:
        # A(mid shoulders)
        xA = (g(arr, f"AL{n}") + g(arr, f"BA{n}")) / 2.0
        yA = (g(arr, f"AM{n}") + g(arr, f"BB{n}")) / 2.0
        zA = (g(arr, f"AN{n}") + g(arr, f"BC{n}")) / 2.0
        # B(mid wrists)
        xB = (g(arr, f"AX{n}") + g(arr, f"BM{n}")) / 2.0
        yB = (g(arr, f"AY{n}") + g(arr, f"BN{n}")) / 2.0
        zB = (g(arr, f"AZ{n}") + g(arr, f"BO{n}")) / 2.0

        dx, dy, dz = xB - xA, yB - yA, zB - zA
        denom = math.hypot(dx, dz)  # sqrt(dx^2 + dz^2)
        pitch = math.degrees(math.atan2(dy, denom))
        vals.append(pitch)
    return vals

# alias (원본 함수명 느낌 원하면 사용)
compute_vertical_angles_from_array = compute_pitch_angles_from_array

def build_pitch_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray,
                              frames: range | list[int] = range(1, 11)
                             ) -> pd.DataFrame:
    """
    프로/일반 Pitch 비교표
    columns: ["Frame", "프로", "일반", "차이(프로-일반)"]
    """
    pro = compute_pitch_angles_from_array(pro_arr, frames=frames)
    ama = compute_pitch_angles_from_array(ama_arr, frames=frames)

    # frames가 range인지/리스트인지 상관없이 실제 프레임 번호 추출
    frame_numbers = list(frames) if not isinstance(frames, range) else list(frames)

    rows = []
    for f, p, a in zip(frame_numbers, pro, ama):
        rows.append([f, p, a, p - a])

    return pd.DataFrame(rows, columns=["Frame", "프로", "일반", "차이(프로-일반)"])
