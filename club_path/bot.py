# diffs5.py

import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_diff1(xlsx_path: Path) -> list[float]:
    """
    1) (AT1+BI1)/2 - (AN1+BC1)/2
    2)~9) (ARn+BGn)/2 - (ALn+BAn)/2
    """
    arr = load_sheet(xlsx_path)
    out = []
    for n in range(1,10):
        if n == 1:
            val = (g(arr, f'AT{n}') + g(arr, f'BI{n}'))/2 \
                - (g(arr, f'AN{n}') + g(arr, f'BC{n}'))/2
        else:
            val = (g(arr, f'AR{n}') + g(arr, f'BG{n}'))/2 \
                - (g(arr, f'AL{n}') + g(arr, f'BA{n}'))/2
        out.append(val)
    return out

def compute_diff2(xlsx_path: Path) -> list[float]:
    """
    1,7) (AZn+BOn)/2 - (ANn+BCn)/2
    else) (AXn+BMn)/2 - (ALn+BAn)/2
    """
    arr = load_sheet(xlsx_path)
    out = []
    for n in range(1,10):
        if n in (1,7):
            val = (g(arr, f'AZ{n}') + g(arr, f'BO{n}'))/2 \
                - (g(arr, f'AN{n}') + g(arr, f'BC{n}'))/2
        else:
            val = (g(arr, f'AX{n}') + g(arr, f'BM{n}'))/2 \
                - (g(arr, f'AL{n}') + g(arr, f'BA{n}'))/2
        out.append(val)
    return out

def compute_diff3(xlsx_path: Path) -> list[float]:
    """
    1) BO1-BC1 
    2~9) BMn-BAn
    """
    arr = load_sheet(xlsx_path)
    out = []
    for n in range(1,10):
        if n == 1:
            out.append(g(arr, f'BO{n}') - g(arr, f'BC{n}'))
        else:
            out.append(g(arr, f'BM{n}') - g(arr, f'BA{n}'))
    return out

def compute_diff4(xlsx_path: Path) -> list[float]:
    """
    ARn - ALn  (n=1..9)
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'AR{n}') - g(arr, f'AL{n}') for n in range(1,10)]

def compute_diff5(xlsx_path: Path) -> list[float]:
    """
    BGn - BAn  (n=1..9)
    """
    arr = load_sheet(xlsx_path)
    return [g(arr, f'BG{n}') - g(arr, f'BA{n}') for n in range(1,10)]


if __name__ == "__main__":
    SAMPLE = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    print("diff1:", compute_diff1(SAMPLE))
    print("diff2:", compute_diff2(SAMPLE))
    print("diff3:", compute_diff3(SAMPLE))
    print("diff4:", compute_diff4(SAMPLE))
    print("diff5:", compute_diff5(SAMPLE))
