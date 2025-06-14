import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx-1

def load_sheet(xlsx: Path) -> np.ndarray:
    return pd.read_excel(xlsx, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_global_normalized_linear_jerk(xlsx_path: Path):
    arr = load_sheet(xlsx_path)

    marker_groups = {
        'CHD':       ('CN','CO','CP'),
        'Wrist':     ('AX','AY','AZ','BM','BN','BO'),
        'Shoulder':  ('AL','AM','AN','BA','BB','BC'),
        'Waist':     ('H','I','J','K','L','M'),
    }
    segments = [(1,4), (4,7), (7,10)]

    # 1) 모든 구간의 저크값을 담을 리스트
    all_jerks = []
    by_group = { k: [] for k in marker_groups }

    for name, codes in marker_groups.items():
        for f1, f2 in segments:
            if name=='CHD':
                p1 = np.array([g(arr, f'{c}{f1}') for c in codes])
                p2 = np.array([g(arr, f'{c}{f2}') for c in codes])
            else:
                left1, right1 = np.array([g(arr, f'{codes[i]}{f1}') for i in range(3)]), \
                                np.array([g(arr, f'{codes[i+3]}{f1}') for i in range(3)])
                left2, right2 = np.array([g(arr, f'{codes[i]}{f2}') for i in range(3)]), \
                                np.array([g(arr, f'{codes[i+3]}{f2}') for i in range(3)])
                p1, p2 = right1-left1, right2-left2

            t1, t2 = g(arr, f'B{f1}'), g(arr, f'B{f2}')
            jerk = np.linalg.norm(p2 - p1) / (t2 - t1)
            all_jerks.append(jerk)
            by_group[name].append(jerk)

    # 2) 전체 최대값
    global_max = max(all_jerks) if all_jerks else 1.0

    # 3) 그룹별로 나누되, global_max로 정규화
    jerks_norm = { name: [j/global_max for j in lst]
                   for name, lst in by_group.items() }

    # 4) DataFrame 생성
    df = pd.DataFrame(
        jerks_norm,
        index=[f'{s}→{e}' for s,e in segments]
    )
    return df

if __name__=='__main__':
    PATH = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")
    df = compute_global_normalized_linear_jerk(PATH)
    print(df.to_markdown())
