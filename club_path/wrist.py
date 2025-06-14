# cnax.py

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

def compute_cn_minus_ax(xlsx_path: Path) -> list[float]:
    """
    Frame 1~10 에 대해 CNₙ - AXₙ 값을 계산해 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'CN{n}') - g(arr, f'AX{n}') for n in range(1, 11)]

if __name__ == '__main__':
    # 테스트용: 실제 경로로 변경해주세요
    sample_file = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    values = compute_cn_minus_ax(sample_file)
    
    # 결과 프린트
    print("CN - AX (Frame 1–10):")
    for i, v in enumerate(values, start=1):
        print(f" Frame {i}: {v:.4f}")
    
    # 간단 플롯
    frames = list(range(1, 11))
    plt.figure(figsize=(8, 4))
    plt.plot(frames, values, marker='o', linestyle='-')
    plt.xticks(frames)
    plt.xlabel('Frame')
    plt.ylabel('CN - AX (units)')
    plt.title('CNₙ – AXₙ (Frame 1–10)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
