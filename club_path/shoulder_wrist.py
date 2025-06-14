# diffs4.py

import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자 → 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy array 로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_ax_minus_al(xlsx_path: Path) -> list[float]:
    """
    Frame 1~9에 대해 AXₙ - ALₙ 값을 계산해 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'AX{n}') - g(arr, f'AL{n}') for n in range(1, 10)]

def compute_bm_minus_ba(xlsx_path: Path) -> list[float]:
    """
    Frame 1~9에 대해 BMₙ - BAₙ 값을 계산해 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'BM{n}') - g(arr, f'BA{n}') for n in range(1, 10)]

if __name__ == '__main__':
    # 테스트용—경로를 실제 엑셀 파일로 바꿔 실행하세요
    sample = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    axal = compute_ax_minus_al(sample)
    bmba = compute_bm_minus_ba(sample)
    print("AX − AL (1–9):", axal)
    print("BM − BA (1–9):", bmba)
