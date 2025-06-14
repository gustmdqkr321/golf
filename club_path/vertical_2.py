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

def compute_vertical_rotations(xlsx_path: Path) -> list[float]:
    """
    Frame 1~10에 대해
      A = midpoint of (ALn,AMn,ANn)&(BAn,BBn,BCn)
      B = midpoint of (AXn,AYn,AZn)&(BMn,BNn,BOn)
    사이의 수직회전각(Vertical Rotation)을 계산하여 반환.
    Vertical = arctan((yB-yA) / sqrt((xB-xA)^2+(zB-zA)^2)) * 180/pi
    """
    arr = load_sheet(xlsx_path)
    vr = []
    for n in range(1, 11):
        # A midpoint
        xA = (g(arr,f'AL{n}')+g(arr,f'BA{n}'))/2
        yA = (g(arr,f'AM{n}')+g(arr,f'BB{n}'))/2
        zA = (g(arr,f'AN{n}')+g(arr,f'BC{n}'))/2
        # B midpoint
        xB = (g(arr,f'AX{n}')+g(arr,f'BM{n}'))/2
        yB = (g(arr,f'AY{n}')+g(arr,f'BN{n}'))/2
        zB = (g(arr,f'AZ{n}')+g(arr,f'BO{n}'))/2

        dy = yB - yA
        dxz = math.hypot(xB - xA, zB - zA)
        angle = math.degrees(math.atan2(dy, dxz))
        vr.append(angle)
    return vr

if __name__ == '__main__':
    # 테스트용: 실제 경로로 바꿔 주세요
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    vals = compute_vertical_rotations(FILE)
    for i,v in enumerate(vals,1):
        print(f"Frame {i}: Vertical Rotation = {v:.2f}°")
