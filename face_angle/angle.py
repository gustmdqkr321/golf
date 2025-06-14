# abc_segmented_signed.py

import numpy as np
import pandas as pd
import math
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자→0-based 인덱스"""
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx-1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy array 로드"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]"""
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_segmented_signed_ABC(xlsx_path: Path) -> list[float]:
    """
    1,7,10구간과 2-6,8-9구간을 나눠 BC 정의 후
    cosθ = (BA·BC)/(|BA||BC|), θ=arccos(cosθ),
    부호는 AY>BN → +, AY<BN → - 로 적용
    """
    arr = load_sheet(xlsx_path)
    thetas = []
    for n in range(1, 11):
        # 좌표
        AX = g(arr, f'AX{n}'); AY = g(arr, f'AY{n}'); AZ = g(arr, f'AZ{n}')
        BM = g(arr, f'BM{n}'); BN = g(arr, f'BN{n}'); BO = g(arr, f'BO{n}')

        # BA 벡터
        BA = np.array([AX-BM, AY-BN, AZ-BO])
        norm_BA = np.linalg.norm(BA)

        # 구간별 BC 정의
        if n in (1, 7, 10):
            BC = np.array([0.0, 0.0, AZ-BO])
        else:  # 2-6,8-9
            BC = np.array([AX-BM, 0.0, AZ-BO])

        norm_BC = np.linalg.norm(BC)

        # cosθ 계산
        dot = BA.dot(BC)
        cosθ = max(-1.0, min(1.0, dot / (norm_BA * norm_BC)))

        # θ (도)
        θ = math.degrees(math.acos(cosθ))

        # 부호 결정 (AY vs BN)
        if AY < BN:
            θ = -θ

        thetas.append(θ)

    return thetas

if __name__ == '__main__':
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    angles = compute_segmented_signed_ABC(FILE)
    frames = list(range(1, 11))

    # DataFrame 출력
    df = pd.DataFrame({
        "Frame":       frames,
        "Segmented ∠ABC": angles
    })
    print(df.to_markdown(index=False))

    # 간단 플롯
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,4))
    # plt.plot(frames, angles, marker='o')
    # plt.xticks(frames)
    # plt.xlabel('Frame n')
    # plt.ylabel('Signed ∠ABC (°)')
    # plt.title('Segmented & Signed ∠ABC per Frame')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
