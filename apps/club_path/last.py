# distances.py

import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자 → 0-based 인덱스"""
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy array로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]"""
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_midpoint_distances(xlsx_path: Path) -> list[float]:
    """
    Frame n = 1..10 에 대해
      A = ((ALn+BAn)/2, (AMn+BBn)/2, (ANn+BCn)/2)
      B = ((AXn+BMn)/2, (AYn+BNn)/2, (AZn+BOn)/2)
    두 점 사이의 유클리드 거리 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    dists = []
    for n in range(1, 11):
        Ax = (g(arr, f'AL{n}') + g(arr, f'BA{n}')) / 2
        Ay = (g(arr, f'AM{n}') + g(arr, f'BB{n}')) / 2
        Az = (g(arr, f'AN{n}') + g(arr, f'BC{n}')) / 2

        Bx = (g(arr, f'AX{n}') + g(arr, f'BM{n}')) / 2
        By = (g(arr, f'AY{n}') + g(arr, f'BN{n}')) / 2
        Bz = (g(arr, f'AZ{n}') + g(arr, f'BO{n}')) / 2

        dx = Bx - Ax
        dy = By - Ay
        dz = Bz - Az

        dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        dists.append(dist)
    return dists

if __name__ == "__main__":
    SAMPLE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    print("Midpoint distances (1–10):", compute_midpoint_distances(SAMPLE))
