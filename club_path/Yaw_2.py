# yaw.py

import numpy as np
import pandas as pd
import math
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
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_yaw_angles(xlsx_path: Path) -> list[float]:
    """
    각 프레임 n=1..10에 대해
      A = midpoint of (ALn, AMn, ANn) & (BAn, BBn, BCn)
      B = midpoint of (AXn, AYn, AZn) & (BMn, BNn, BOn)
    사이의 수평회전각도(Yaw)를 계산하여 도 단위 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    yaws = []
    for n in range(1, 11):
        # A 점: (ALn, AMn, ANn) 과 (BAn, BBn, BCn) 의 중간
        xA = (g(arr, f'AL{n}') + g(arr, f'BA{n}')) / 2
        yA = (g(arr, f'AM{n}') + g(arr, f'BB{n}')) / 2
        zA = (g(arr, f'AN{n}') + g(arr, f'BC{n}')) / 2

        # B 점: (AXn, AYn, AZn) 과 (BMn, BNn, BOn) 의 중간
        xB = (g(arr, f'AX{n}') + g(arr, f'BM{n}')) / 2
        yB = (g(arr, f'AY{n}') + g(arr, f'BN{n}')) / 2
        zB = (g(arr, f'AZ{n}') + g(arr, f'BO{n}')) / 2

        # XY 평면 거리와 높이 차
        dx = xB - xA
        dy = yB - yA
        dz = zB - zA

        # Yaw = arctan( dz / sqrt(dx^2+dy^2) ) in degrees
        yaw = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        yaws.append(yaw)

    return yaws

if __name__ == '__main__':
    # 실제 파일 경로로 수정하세요
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    yaw_angles = compute_yaw_angles(FILE)
    print("Frame | Yaw (°)")
    for i, angle in enumerate(yaw_angles, start=1):
        print(f"  {i:2d}   | {angle:6.2f}")

    # 결과 간단 플롯
    import matplotlib.pyplot as plt
    frames = list(range(1, 11))
    plt.figure(figsize=(8,4))
    plt.plot(frames, yaw_angles, marker='o')
    plt.xticks(frames)
    plt.xlabel('Frame')
    plt.ylabel('Yaw Angle (°)')
    plt.title('Frame-wise Horizontal Yaw')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
