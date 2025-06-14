import numpy as np
import pandas as pd
import math
from pathlib import Path

# Utility to read Excel without header as numpy array
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])


def compute_cp_cs_cq_cn_diffs(xlsx_path: Path) -> list[float]:
    """
    Frame 1: CP1-CS1
    Frame 2-6: CQn-CNn
    Frame 7: CP7-CS7
    Frame 8-9: CNn-CQn
    반환 리스트 길이 9
    """
    arr = load_sheet(xlsx_path)
    vals = []
    for n in range(1, 10):
        if n == 1 or n == 7:
            v = g(arr, f'CP{n}') - g(arr, f'CS{n}')
        elif 2 <= n <= 6:
            v = g(arr, f'CQ{n}') - g(arr, f'CN{n}')
        else:  # n == 8 or 9
            v = g(arr, f'CN{n}') - g(arr, f'CQ{n}')
        vals.append(v)
    return vals


def compute_ay_bn_diffs(xlsx_path: Path) -> list[float]:
    arr = load_sheet(xlsx_path)
    # 1~9 프레임 AY−BN 차이
    diffs = [g(arr, f'AY{n}') - g(arr, f'BN{n}') for n in range(1, 10)]
    # 결과 리스트: frame1~6
    result = diffs[:6]
    # frame2~6 표준편차
    std_2_6 = float(np.std(diffs[1:6], ddof=0))
    result.append(std_2_6)
    # frame7~9
    result.extend(diffs[6:])
    return result


if __name__ == '__main__':
    # 실제 파일 경로
    FILE = Path('/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx')

    cp_cs_diffs = compute_cp_cs_cq_cn_diffs(FILE)
    ay_bn_diffs = compute_ay_bn_diffs(FILE)
    frames = list(range(1, 10))

    df = pd.DataFrame({
        'Frame': frames,
        'CP-CS / CQ-CN Diffs': cp_cs_diffs,
        'AY-BN Diffs': ay_bn_diffs
    })
    print(df.to_markdown(index=False))

    # (필요시 plot)
    # import matplotlib.pyplot as plt
    # plt.plot(frames, cp_cs_diffs, marker='o', label='CP-CS/CQ-CN')
    # plt.plot(frames, ay_bn_diffs, marker='s', label='AY-BN')
    # plt.xlabel('Frame')
    # plt.ylabel('Value')
    # plt.title('Frame-wise Differences')
    # plt.legend()
    # plt.grid(True)
    # plt.show()