# cocking_table.py

import math
import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀의 A, B, ..., Z, AA, AB... 형태의 컬럼명을 0부터 시작하는 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀 파일을 numpy 2D 배열로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """예: 'AX1' → arr[0, col('AX')]"""
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_angles(path: Path) -> list[float]:
    arr = load_sheet(path)
    out = []
    for n in range(1, 11):
        AR,AS,AT = g(arr,f"AR{n}"), g(arr,f"AS{n}"), g(arr,f"AT{n}")
        AX,AY,AZ = g(arr,f"AX{n}"), g(arr,f"AY{n}"), g(arr,f"AZ{n}")
        CN,CO,CP = g(arr,f"CN{n}"), g(arr,f"CO{n}"), g(arr,f"CP{n}")
        v1 = np.array([AR-AX, AS-AY, AT-AZ])
        v2 = np.array([CN-AX, CO-AY, CP-AZ])
        dot = float(v1.dot(v2))
        norm = np.linalg.norm(v1)*np.linalg.norm(v2)
        cosθ = max(-1, min(1, dot/norm))
        θ = math.degrees(math.acos(cosθ))
        out.append(θ)
    return out

def compute_table(path1: Path, path2: Path) -> pd.DataFrame:
    """
    두 파일에 대해 다음 표를 만들어 반환
      • index: ['ADD','BH','BH2','TOP','TR','DH','IMP','FH1','FH2','FIN',
                '1-4','4-6','Cocking_Maintenance']
      • columns: ['Rory_∠ABC','Hong_∠ABC','Rory_Δ(°)','Hong_Δ(°)']
    """
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
    # 1) 각 프레임 ∠ABC
    ang1 = compute_angles(path1)
    ang2 = compute_angles(path2)
    # 2) 프레임 간 Δ
    delta1 = [None]  # 첫 행은 빈칸
    delta2 = [None]
    for i in range(1,10):
        delta1.append(ang1[i] - ang1[i-1])
        delta2.append(ang2[i] - ang2[i-1])
    # 3) 구간별 Δ
    d1_4 = ang1[3] - ang1[0]
    d4_6 = ang1[5] - ang1[3]
    m1   = (1 - abs(ang1[5]-ang1[3]) / abs(ang1[3])) * 100
    d1_4_2 = ang2[3] - ang2[0]
    d4_6_2 = ang2[5] - ang2[3]
    m2     = (1 - abs(ang2[5]-ang2[3]) / abs(ang2[3])) * 100

    # 4) DataFrame 조립
    idx = labels + ["1-4","4-6","Cocking_Maintenance"]
    data = {
        "Rory_∠ABC":           ang1 + [d1_4, d4_6, None],
        "Hong_∠ABC":           ang2 + [d1_4_2, d4_6_2, None],
        "Rory_Δ(°)":           delta1 + [None, None, m1],
        "Hong_Δ(°)":           delta2 + [None, None, m2],
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Frame"
    return df

def main():
    # ─── 1) 엑셀 파일 경로 설정 ───
    path_rory = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    # ─── 2) 데이터 요약 ───
    df_cocking = compute_table(path_rory, path_rory)
    print("Cocking Data Summary:")
    print(df_cocking)

if __name__ == "__main__":
    main()