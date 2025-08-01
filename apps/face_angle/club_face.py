# tilt_angle.py

from pathlib import Path
import math
import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy 2D 배열로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    code 에서 숫자는 프레임(1-based), 문자는 열 레이블(A…Z,AA…)
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    r = num - 1
    c = col_letters_to_index(letters)
    return float(arr[r, c])

def compute_tilt_angles(xlsx_path: Path) -> list[float]:
    """
    1~9 프레임에 대해 CP, CS, CN, CQ 좌표를 g()로 꺼내,
    구간별(1,7 / 2-6 / 8-9) 부호 규칙에 따라 tilt 각도를 계산하여 반환.
    """
    arr = load_sheet(xlsx_path)
    angles = []
    for n in range(1, 10):
        CP = g(arr, f"CP{n}")
        CS = g(arr, f"CS{n}")
        CN = g(arr, f"CN{n}")
        CQ = g(arr, f"CQ{n}")

        # 분자/분모
        if n in (1, 7):            # 구간1,7
            num = CP - CS
            den = CN - CQ
            θ = math.degrees(math.atan2(abs(num), abs(den)))
            θ = -θ if CP < CS else θ

        elif 2 <= n <= 6:          # 구간2~6
            num = CQ - CN
            den = CP - CS
            θ = math.degrees(math.atan2(abs(num), abs(den)))
            θ = -θ if num < 0 else θ

        else:                      # 구간8~9 (구간2~6과 부호 반대)
            num = CQ - CN
            den = CP - CS
            θ = math.degrees(math.atan2(abs(num), abs(den)))
            θ = θ if num < 0 else -θ

        angles.append(round(θ, 2))
    return angles

if __name__ == "__main__":
    FILE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    tilt = compute_tilt_angles(FILE)
    for i, a in enumerate(tilt, start=1):
        print(f"Frame {i}: {a:+.2f}°")
