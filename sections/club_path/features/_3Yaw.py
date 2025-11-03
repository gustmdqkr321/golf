# sections/<섹션>/features/_yaw.py
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 엑셀 A1 주소 → 인덱스 / 값 읽기 유틸
# ─────────────────────────────────────────
_COL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

def _col_idx(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, …, Z, AA, AB, …)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=AX]
    row = frame-1, col = 엑셀 열
    """
    m = _COL_RE.match(code.strip())
    if not m:
        raise ValueError(f"잘못된 주소: {code}")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    try:
        return float(arr[row, col])
    except Exception:
        return float("nan")

# ─────────────────────────────────────────
# Yaw 계산 (예시식 그대로)
# ─────────────────────────────────────────
def compute_yaw_angles_from_array(arr: np.ndarray, frames: range | list[int] = range(1, 11)) -> list[float]:
    """
    Frame n마다
      A = midpoint of (ALn,AMn,ANn) and (BAn,BBn,BCn)
      B = midpoint of (AXn,AYn,AZn) and (BMn,BNn,BOn)
    yaw = atan2( dz, sqrt(dx^2 + dy^2) ) [deg]
    """
    yaws: list[float] = []
    for n in frames:
        # A(mid of shoulders)
        xA = (g(arr, f"AL{n}") + g(arr, f"BA{n}")) / 2.0
        yA = (g(arr, f"AM{n}") + g(arr, f"BB{n}")) / 2.0
        zA = (g(arr, f"AN{n}") + g(arr, f"BC{n}")) / 2.0
        # B(mid of wrists)
        xB = (g(arr, f"AX{n}") + g(arr, f"BM{n}")) / 2.0
        yB = (g(arr, f"AY{n}") + g(arr, f"BN{n}")) / 2.0
        zB = (g(arr, f"AZ{n}") + g(arr, f"BO{n}")) / 2.0

        dx, dy, dz = xB - xA, yB - yA, zB - zA
        yaw_deg = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        yaws.append(yaw_deg)
    return yaws

def build_yaw_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray,
                            frames: range | list[int] = range(1, 11)) -> pd.DataFrame:
    """
    프로/일반 비교표 생성
    columns: ["Frame", "프로", "일반", "차이(프로-일반)"]
    """
    pro = compute_yaw_angles_from_array(pro_arr, frames=frames)
    ama = compute_yaw_angles_from_array(ama_arr, frames=frames)
    rows = []
    for i, (p, a) in enumerate(zip(pro, ama), start=frames[0] if isinstance(frames, range) else frames[0]):
        rows.append([str(i)+" Frame", p, a, p - a])
    df = pd.DataFrame(rows, columns=["Frame", "프로", "일반", "차이(프로-일반)"])
    return df
