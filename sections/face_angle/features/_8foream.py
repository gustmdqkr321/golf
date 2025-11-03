# sections/face_angle/features/_3aux_tables.py
from __future__ import annotations
import re
import math
import numpy as np
import pandas as pd

# ── A1 유틸 ───────────────────────────────────────────────────────────
_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col('AX')]
    """
    m = _CELL.fullmatch(code.strip())
    if not m: 
        return float("nan")
    r = int(m.group(2)) - 1
    c = _col_idx(m.group(1))
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def _round2_list(vals: list[float]) -> list[float]:
    return [float("nan") if (v is None or not np.isfinite(v)) else float(np.round(v, 2)) for v in vals]

# ──────────────────────────────────────────────────────────────────────
# 1) Tilt Numerators (프레임 1~9)
#   1,7: CPn - CSn
#   2~6: CQn - CNn
#   8~9: CNn - CQn
# ──────────────────────────────────────────────────────────────────────
def _tilt_numerators_from_array(arr: np.ndarray) -> list[float]:
    out = []
    for n in range(1, 10):
        if n in (1, 7):
            v = g(arr, f"CP{n}") - g(arr, f"CS{n}")
        elif 2 <= n <= 6:
            v = g(arr, f"CQ{n}") - g(arr, f"CN{n}")
        else:
            v = g(arr, f"CN{n}") - g(arr, f"CQ{n}")
        out.append(v)
    return _round2_list(out)

def build_tilt_numerators_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = _tilt_numerators_from_array(pro_arr)
    a = _tilt_numerators_from_array(ama_arr)
    d = _round2_list([pp - aa for pp, aa in zip(p, a)])
    idx = ["1(Z)","2(X)","3(X)","4(X)","5(X)","6(X)","7(Z)","8(X)","9(X)"]
    df = pd.DataFrame(
        {"seg":idx,"프로": p, "일반": a, "차이(프로-일반)": d},
        
    )
    return df

# ──────────────────────────────────────────────────────────────────────
# 2) AYn - BNn (프레임 1~9) + "STD(2–6)" 한 줄 삽입(7번째 위치)
# ──────────────────────────────────────────────────────────────────────
def _ay_bn_diffs_from_array(arr: np.ndarray) -> list[float]:
    diffs = [g(arr, f"AY{n}") - g(arr, f"BN{n}") for n in range(1, 10)]
    std_2_6 = float(np.std(diffs[1:6], ddof=0))  # 2~6 프레임
    diffs.insert(6, std_2_6)  # 7번째 위치에 삽입
    return _round2_list(diffs)

def build_ay_bn_diffs_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = _ay_bn_diffs_from_array(pro_arr)  # len=10 (…+STD 행 포함)
    a = _ay_bn_diffs_from_array(ama_arr)
    d = _round2_list([pp - aa for pp, aa in zip(p, a)])

    idx = ["1","2","3","4","5","6","STD(2–6)","7","8","9"]
    df = pd.DataFrame(
        {"seg":idx,"프로": p, "일반": a, "차이(프로-일반)": d},
        
    )
    return df

# ──────────────────────────────────────────────────────────────────────
# 3) ∠ABC (프레임 1..10)
#   A(AX,AY,AZ), B(BM,BN,BO), C(BM,BN,AZ)
# ──────────────────────────────────────────────────────────────────────
# --- 새로 교체: 서명(부호)까지 적용된 ABC 각도 계산 ---
def _abc_signed_angles_from_array(arr: np.ndarray) -> list[float]:
    out: list[float] = []
    for n in range(1, 11):
        AX, AY, AZ = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")
        BM, BN, BO = g(arr, f"BM{n}"), g(arr, f"BN{n}"), g(arr, f"BO{n}")

        # B = (BM, BN, BO)
        bx, by, bz = BM, BN, BO

        # C 점 선택
        if n in (1, 7, 10):
            cx, cy, cz = BM, BN, AZ          # C = (BM, BN, AZ)
        else:
            cx, cy, cz = AX, BN, AZ          # C = (AX, BN, AZ)

        # A = (AX, AY, AZ)
        ax, ay, az = AX, AY, AZ

        # 벡터 BA, BC
        vBA = np.array([ax - bx, ay - by, az - bz], dtype=float)
        vBC = np.array([cx - bx, cy - by, cz - bz], dtype=float)

        denom = np.linalg.norm(vBA) * np.linalg.norm(vBC)
        if not np.isfinite(denom) or denom == 0:
            out.append(float("nan"))
            continue

        cos_th = float(np.dot(vBA, vBC)) / denom
        cos_th = max(-1.0, min(1.0, cos_th))  # clamp
        th = math.degrees(math.acos(cos_th))   # [0,180]

        # 부호 적용: AY vs BN
        if AY < BN:
            th = -th

        out.append(float(np.round(th, 2)))
    return out

def build_abc_angles_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = _abc_signed_angles_from_array(pro_arr)   # len=10
    a = _abc_signed_angles_from_array(ama_arr)
    d = [float("nan") if (pp is None or a is None) else float(np.round(pp - aa, 2))
         for pp, aa in zip(p, a)]

    idx = ["1","2","3","4","5","6","STD(2–6)","7","8","9"]
    df = pd.DataFrame(
        {"seg":idx,"프로": p, "일반": a, "차이(프로-일반)": d},
        
    )
    return df
