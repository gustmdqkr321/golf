import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx-1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_ax_minus_ar(xlsx_path: Path) -> list[float]:
    """
    Frame 1~9 에 대해 AXₙ - ARₙ 을 계산해 반환
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'AX{n}') - g(arr, f'AR{n}') for n in range(1, 10)]

def compute_bm_minus_bg(xlsx_path: Path) -> list[float]:
    """
    Frame 1~9 에 대해 BMₙ - BGₙ 을 계산해 반환
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'BM{n}') - g(arr, f'BG{n}') for n in range(1, 10)]

if __name__ == '__main__':
    # 테스트용: 실제 경로로 변경해주세요
    sample_file = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    
    ax_minus_ar = compute_ax_minus_ar(sample_file)
    bm_minus_bg = compute_bm_minus_bg(sample_file)
    
    print("AX - AR (Frame 1–9):", ax_minus_ar)
    print("BM - BG (Frame 1–9):", bm_minus_bg)