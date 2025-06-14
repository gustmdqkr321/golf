import math
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

def compute_vertical_angles(xlsx_path: Path) -> list[float]:
    """
    Frame 1~10에 대해
      A = midpoint of (ALn, AMn, ANn) & (BAn, BBn, BCn)
      B = midpoint of (AXn, AYn, AZn) & (BMn, BNn, BOn)
    사이의 수직회전각도(Pitch)를 계산하여 리스트로 반환.
    Pitch = arctan( (yB-yA) / sqrt((xB-xA)^2 + (zB-zA)^2) ) × 180/π
    """
    arr = load_sheet(xlsx_path)
    pitches = []
    for n in range(1, 11):
        # A점 = (AL,AM,AN) & (BA,BB,BC) 중간
        xA = (g(arr, f'AL{n}') + g(arr, f'BA{n}'))/2
        yA = (g(arr, f'AM{n}') + g(arr, f'BB{n}'))/2
        zA = (g(arr, f'AN{n}') + g(arr, f'BC{n}'))/2

        # B점 = (AX,AY,AZ) & (BM,BN,BO) 중간
        xB = (g(arr, f'AX{n}') + g(arr, f'BM{n}'))/2
        yB = (g(arr, f'AY{n}') + g(arr, f'BN{n}'))/2
        zB = (g(arr, f'AZ{n}') + g(arr, f'BO{n}'))/2

        dx = xB - xA
        dz = zB - zA
        dy = yB - yA

        denom = math.hypot(dx, dz)
        pitch = math.degrees(math.atan2(dy, denom))
        pitches.append(pitch)
    return pitches

if __name__ == '__main__':
    # 테스트용—경로를 실제 파일로 바꿔서 실행하세요
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    vals = compute_vertical_angles(FILE)
    for i,v in enumerate(vals,1):
        print(f"Frame {i}: Pitch = {v:.2f}°")
