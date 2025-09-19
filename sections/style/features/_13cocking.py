# 예: sections/swing/features/_angles.py (현재 top2 모듈)
import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = "".join(filter(str.isalpha, code))
    num     = int("".join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _angle_ABC_deg_at(arr: np.ndarray, frame: int) -> float:
    """frame에서 A=(AR,AS,AT), B=(AX,AY,AZ), C=(CN,CO,CP)의 ∠ABC [deg]"""
    f = str(frame)
    A = np.array([g(arr, "AR"+f), g(arr, "AS"+f), g(arr, "AT"+f)], dtype=float)
    B = np.array([g(arr, "AX"+f), g(arr, "AY"+f), g(arr, "AZ"+f)], dtype=float)
    C = np.array([g(arr, "CN"+f), g(arr, "CO"+f), g(arr, "CP"+f)], dtype=float)
    BA = A - B
    BC = C - B
    na = np.linalg.norm(BA); nc = np.linalg.norm(BC)
    if na == 0 or nc == 0:
        return float("nan")
    cosang = float(np.dot(BA, BC) / (na * nc))
    # 수치 안정화
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def build_frames_angle_ABC_table(pro_arr: np.ndarray, ama_arr: np.ndarray,
                                 frames: tuple[int, ...] = (4, 6, 8)) -> pd.DataFrame:
    """프레임 4/6/8의 ∠ABC를 한 표로 (항목/프로/일반/차이)"""
    rows = []
    for f in frames:
        p = _angle_ABC_deg_at(pro_arr, f)
        a = _angle_ABC_deg_at(ama_arr, f)
        rows.append([f"∠ {f}", round(p, 2), round(a, 2)])
    df = pd.DataFrame(rows, columns=["항목", "프로", "일반"])
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                          - pd.to_numeric(df["일반"], errors="coerce")).round(2)
    return df
