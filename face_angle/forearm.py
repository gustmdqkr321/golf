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
    """헤더 없는 엑셀 파일을 numpy 2D 배열로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values


def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    code에서 숫자는 프레임(1-based), 문자는 열 레이블
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])


def compute_tilt_numerators(xlsx_path: Path) -> list[float]:
    """
    1~9 프레임에 대해 부호 규칙 제외, 각 프레임의 분자(좌표 차이) 계산:
      - 프레임 1,7: CPn - CSn
      - 프레임 2~6: CQn - CNn
      - 프레임 8~9: CNn - CQn
    """
    arr = load_sheet(xlsx_path)
    nums = []
    for n in range(1, 10):
        if n in (1, 7):
            nums.append(g(arr, f"CP{n}") - g(arr, f"CS{n}"))
        elif 2 <= n <= 6:
            nums.append(g(arr, f"CQ{n}") - g(arr, f"CN{n}"))
        else:
            nums.append(g(arr, f"CN{n}") - g(arr, f"CQ{n}"))
    return nums


def compute_ay_bn_diffs(xlsx_path: Path) -> list[float]:
    """
    1~9 프레임에 대해 AY n - BN n 차이를 계산하고,
    프레임 2~6 차이들의 표준편차를 7번째 항목으로 삽입한 10개 리스트 반환
    """
    arr = load_sheet(xlsx_path)
    diffs = [g(arr, f"AY{n}") - g(arr, f"BN{n}") for n in range(1, 10)]
    std_2_6 = float(np.std(diffs[1:6], ddof=0))
    diffs.insert(6, std_2_6)
    return diffs


def compute_abc_angles(xlsx_path: Path) -> list[float]:
    """
    각 프레임 1..10에 대해 A(AXn,AYn,AZn),
    B(BMn,BNn,BO n), C(BMn,BNn,AZn) 세 점으로 각 ∠ABC 계산하여 리스트 반환
    """
    arr = load_sheet(xlsx_path)
    angles = []
    for n in range(1, 11):
        ax, ay, az = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")
        bx, by, bz = g(arr, f"BM{n}"), g(arr, f"BN{n}"), g(arr, f"BO{n}")
        cx, cy, cz = g(arr, f"BM{n}"), g(arr, f"BN{n}"), g(arr, f"AZ{n}")
        vBA = np.array([ax-bx, ay-by, az-bz], dtype=float)
        vBC = np.array([cx-bx, cy-by, cz-bz], dtype=float)
        denom = np.linalg.norm(vBA) * np.linalg.norm(vBC)
        if denom == 0:
            angles.append(float('nan'))
            continue
        cosθ = np.dot(vBA, vBC) / denom
        cosθ = max(-1.0, min(1.0, cosθ))
        θ = math.degrees(math.acos(cosθ))
        angles.append(round(θ, 2))
    return angles

if __name__ == "__main__":
    file = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    print("Tilt numerators:", compute_tilt_numerators(file))
    print("AY-BN diffs:  ", compute_ay_bn_diffs(file))
    print("ABC angles:  ", compute_abc_angles(file))
