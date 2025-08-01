# yz_angle.py

import math
import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy 2D 배열로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame-1, col = 엑셀 컬럼
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_yz_plane_angles(xlsx_path: Path) -> list[float]:
    """
    각 프레임 n=1..10에 대해 YZ 평면(정면)에서 팔꿈치→손목 벡터와
    클럽헤드→손목 벡터 사이의 각도(°)를 계산하여 리스트로 반환.
    
    - 팔꿈치 YZ: (ASn, ATn)
    - 손목   YZ: (AYn, AZn)
    - 클럽헤드 YZ: (COn, CPn)
    """
    arr = load_sheet(xlsx_path)
    angles = []
    for n in range(1, 11):
        AS = g(arr, f"AS{n}")  # elbow Y
        AT = g(arr, f"AT{n}")  # elbow Z
        AY = g(arr, f"AY{n}")  # wrist Y
        AZ = g(arr, f"AZ{n}")  # wrist Z
        CO = g(arr, f"CO{n}")  # clubhead Y
        CP = g(arr, f"CP{n}")  # clubhead Z
        
        # YZ평면 벡터
        v_elbow = np.array([AS - AY, AT - AZ])
        v_club  = np.array([CO - AY, CP - AZ])
        
        dot  = float(np.dot(v_elbow, v_club))
        norm = np.linalg.norm(v_elbow) * np.linalg.norm(v_club)
        # 0나누기 안전처리
        if norm == 0:
            angles.append(float('nan'))
            continue
        
        # 부동소수점 오류 방지용 clamp
        cosθ = max(-1.0, min(1.0, dot / norm))
        θ   = math.degrees(math.acos(cosθ))
        angles.append(θ)
    
    return angles

if __name__ == "__main__":
    # 테스트용
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    yz_angles = compute_yz_plane_angles(FILE)
    for i, a in enumerate(yz_angles, start=1):
        print(f"Frame {i}: {a:.1f}°")
