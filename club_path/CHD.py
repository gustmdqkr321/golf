# folder2/main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    예: 'CN4' → arr[row=0, col=col_letters_to_index('CN')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_cn_4_9(xlsx_path: Path) -> list[float]:
    """Frame 4~9에 대한 CNₙ 값을 리스트로 반환"""
    arr = load_sheet(xlsx_path)
    return [ g(arr, f'CN{n}') for n in range(4, 10) ]

def compute_bm_4_9(xlsx_path: Path) -> list[float]:
    """Frame 4~9에 대한 BMₙ 값을 리스트로 반환"""
    arr = load_sheet(xlsx_path)
    return [ g(arr, f'BM{n}') for n in range(4, 10) ]

if __name__ == '__main__':
    # 실제 파일 경로
    FILE = Path('/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx')
    cn_4_9 = compute_cn_4_9(FILE)
    bm_4_9 = compute_bm_4_9(FILE)
    print("CN 4-9:", cn_4_9)
    print("BM 4-9:", bm_4_9)
    # 플롯 예시
    plt.figure(figsize=(10, 5))
    plt.plot(range(4, 10), cn_4_9, marker='o', label='CN 4-9')
    plt.plot(range(4, 10), bm_4_9, marker='x', label='BM 4-9')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('CN and BM from Frame 4 to 9')
    plt.legend()
    plt.grid(True)
    plt.show()
#     plt.savefig('cn_bm_plot.png')
