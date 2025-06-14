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
    """헤더 없는 엑셀을 읽어 numpy array 반환"""
    return pd.read_excel(xlsx_path, header=None).values


def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])


def compute_tilt_angles_excel(xlsx_path: Path) -> list[float]:
    """
    엑셀에서 CPn, CSn, CNn, CQn(프레임 1~9) 데이터를 g()로 읽어
    확정된 구간별 공식을 적용하여 tilt 각도를 계산한 뒤 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    angles = []
    for n in range(1, 10):  # Frame 1~9
        CP = g(arr, f'CP{n}')
        CS = g(arr, f'CS{n}')
        CN = g(arr, f'CN{n}')
        CQ = g(arr, f'CQ{n}')

        # 구간 1,7 (Frame 1,7)
        if n in [1, 7]:
            num = CP - CS
            den = CN - CQ
            theta = math.degrees(math.atan(abs(num) / abs(den)))
            if CP < CS:
                theta = -theta
        # 구간 2~6 (Frame 2~6)
        elif 2 <= n <= 6:
            num = CQ - CN
            den = CP - CS
            theta = math.degrees(math.atan(abs(num) / abs(den)))
            if num < 0:
                theta = -theta
        # 구간 8~9 (Frame 8~9), 부호 반대
        else:
            num = CQ - CN
            den = CP - CS
            theta = math.degrees(math.atan(abs(num) / abs(den)))
            if num >= 0:
                theta = -theta
        angles.append(theta)
    return angles

if __name__ == '__main__':
    # 실제 파일 경로로 수정하세요
    FILE = Path('/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx')

    # Tilt 각도 계산
    tilt_angles = compute_tilt_angles_excel(FILE)
    frames = list(range(1, len(tilt_angles) + 1))

    # DataFrame 출력
    df = pd.DataFrame({'Frame': frames, 'TiltAngle(°)': tilt_angles})
    print(df.to_markdown(index=False))

    # 플롯 및 저장
    plt.figure(figsize=(8, 4))
    plt.plot(frames, tilt_angles, marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Tilt Angle (°)')
    plt.title('Frame-wise Tilt Angle')
    plt.xticks(frames)
    plt.grid(True)
    plt.tight_layout()
    plt.show()