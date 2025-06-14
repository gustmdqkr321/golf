import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없이 엑셀을 읽어 numpy array 반환"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    코드 예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_arcsin_angles(xlsx_path: Path):
    """
    θ = -arcsin(  (AX-AR)*(CO-AY) - (AY-AS)*(CN-AX)
                --------------------------------------
                sqrt((AX-AR)^2+(AY-AS)^2+(AZ-AT)^2)
                * sqrt((CN-AX)^2+(CO-AY)^2+(CP-AZ)^2)
                )
    를 1~10 프레임에 대해 계산해 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    angles = []
    for n in range(1, 11):
        # 왼어깨(A1) → (ARn, ASn, ATn), 왼손목(A1) → (AXn, AYn, AZn)
        AR, AS, AT = g(arr, f'AR{n}'), g(arr, f'AS{n}'), g(arr, f'AT{n}')
        AX, AY, AZ = g(arr, f'AX{n}'), g(arr, f'AY{n}'), g(arr, f'AZ{n}')
        # 클럽헤드 → (CNn, COn, CPn)
        CN, CO, CP = g(arr, f'CN{n}'), g(arr, f'CO{n}'), g(arr, f'CP{n}')

        # 분자
        num = (AX-AR)*(CO-AY) - (AY-AS)*(CN-AX)
        # 분모
        norm1 = math.sqrt((AX-AR)**2 + (AY-AS)**2 + (AZ-AT)**2)
        norm2 = math.sqrt((CN-AX)**2 + (CO-AY)**2 + (CP-AZ)**2)

        # 도메인 안전성 확보
        v = num / (norm1 * norm2)
        v = max(-1.0, min(1.0, v))

        # θ 계산 (–arcsin → 도 단위)
        theta = -math.degrees(math.asin(v))
        angles.append(theta)
    return angles

if __name__ == '__main__':
    # 실제 파일 경로로 수정하세요
    FILE1 = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")
    FILE2 = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    angles1 = compute_arcsin_angles(FILE1)
    angles2 = compute_arcsin_angles(FILE2)
    frames = list(range(1, 11))

    # 결과를 DataFrame 으로 출력
    df = pd.DataFrame({
        'Frame': frames,
        FILE1.stem: angles1,
        FILE2.stem: angles2
    })
    print(df.to_markdown(index=False))

    # matplotlib 으로 시각화
    plt.figure(figsize=(8,4))
    plt.plot(frames, angles1, marker='o', label=FILE1.stem)
    plt.plot(frames, angles2, marker='s', label=FILE2.stem)
    plt.xlabel('Frame')
    plt.ylabel('θ (°)')
    plt.title('Frame-wise θ Comparison (−arcsin 공식)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
