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
    
def _angle_ABC_deg_frame4(arr: np.ndarray) -> float:
    A = np.array([g(arr, "AL4"), g(arr, "AM4"), g(arr, "AN4")], dtype=float)
    B = np.array([g(arr, "AR4"), g(arr, "AS4"), g(arr, "AT4")], dtype=float)
    C = np.array([g(arr, "AX4"), g(arr, "AY4"), g(arr, "AZ4")], dtype=float)
    BA = A - B;  BC = C - B
    na = np.linalg.norm(BA); nc = np.linalg.norm(BC)
    if na == 0 or nc == 0:
        return float("nan")
    cosang = float(np.dot(BA, BC) / (na * nc))
    cosang = max(-1.0, min(1.0, cosang))  # 수치 안정화
    return float(np.degrees(np.arccos(cosang)))

def build_frame4_angle_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """프레임4 각도표: ∠ABC (deg) — 항목/프로/일반/차이"""
    angle_p = _angle_ABC_deg_frame4(pro_arr)
    angle_a = _angle_ABC_deg_frame4(ama_arr)
    df = pd.DataFrame([["L Elb Ang", round(angle_p, 2), round(angle_a, 2)]],
                      columns=["항목", "프로", "일반"])
    df["차이(프로-일반)"] = (pd.to_numeric(df["프로"], errors="coerce")
                           - pd.to_numeric(df["일반"], errors="coerce")).round(2)
    return df