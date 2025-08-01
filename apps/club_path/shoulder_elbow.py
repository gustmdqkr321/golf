# diffs3.py

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
    """헤더 없는 엑셀을 numpy array로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AR1' → arr[row=0, col=col_letters_to_index('AR')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_ar_minus_al(xlsx_path: Path) -> list[float]:
    """
    Frame 1~9에 대해 ARₙ − ALₙ 값을 계산해 반환
    """
    arr = load_sheet(xlsx_path)
    return [ g(arr, f'AR{n}') - g(arr, f'AL{n}') for n in range(1, 10) ]

def compute_bg_minus_ba(xlsx_path: Path) -> list[float]:
    """
    Frame 1~9에 대해 BGₙ − BAₙ 값을 계산해 반환
    """
    arr = load_sheet(xlsx_path)
    return [ g(arr, f'BG{n}') - g(arr, f'BA{n}') for n in range(1, 10) ]

if __name__ == '__main__':
    # 테스트용 경로를 실제 엑셀 파일로 바꿔주세요
    sample = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    ar_al = compute_ar_minus_al(sample)
    bg_ba = compute_bg_minus_ba(sample)
    print("AR−AL:", ar_al)
    print("BG−BA:", bg_ba)
