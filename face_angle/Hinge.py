# hinging_table.py

import math
import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_hinging(AX, AY, AZ, AR, AS, AT, CN, CO, CP) -> float:
    """
    한 프레임에 대해 음의 arcsin XY-투영 힌징 각도를 degree 로 반환합니다.

    θ = -arcsin(
          (AX-AR)*(CO-AY) - (AY-AS)*(CN-AX)
        -------------------------------------
        sqrt((AX-AR)^2 + (AY-AS)^2 + (AZ-AT)^2)
        × sqrt((CN-AX)^2 + (CO-AY)^2 + (CP-AZ)^2)
    )
    """
    # 1) 분자: XY 평면에서의 외적 z 성분
    num = (AX - AR) * (CO - AY) - (AY - AS) * (CN - AX)

    # 2) 분모: 두 3D 벡터 길이의 곱
    len_aw = math.sqrt((AX-AR)**2 + (AY-AS)**2 + (AZ-AT)**2)
    len_wc = math.sqrt((CN-AX)**2 + (CO-AY)**2 + (CP-AZ)**2)
    denom = len_aw * len_wc if len_aw and len_wc else 0.0

    # 3) 비율 및 클램핑
    ratio = num/denom if denom != 0.0 else 0.0
    ratio = max(-1.0, min(1.0, ratio))

    # 4) θ 계산 (–arcsin → 라디안, 도로 변환)
    theta = -math.degrees(math.asin(ratio))
    return round(theta, 2)

def signed_maintenance_score(top: float, dh: float) -> float:
    """TOP→DH 사이 signed 유지지수 (percent) 계산"""
    delta = dh - top
    drop_ratio = abs(delta)/abs(top) if top!=0 else 0.0
    # 방향 일치 & 과도하게 풀리지 않았으면 +
    if np.sign(top)==np.sign(dh) and abs(dh)<=abs(top):
        return round((1-drop_ratio)*100,2)
    else:
        return round(-drop_ratio*100,2)

def calculate_full_hinging_table(path_rory: Path, path_hong: Path) -> pd.DataFrame:
    """Rory/Hong 두 파일로 전체 힌징 테이블(1~10 프레임, Δ, 구간 Δ, 유지지수) 생성"""
    # 1) 각 프레임 힌징
    arr_r = load_sheet(path_rory)
    arr_h = load_sheet(path_hong)
    r_angles = []
    h_angles = []
    for i in range(1, 11):
        AR, AS, AT = g(arr_r, f"AR{i}"), g(arr_r, f"AS{i}"), g(arr_r, f"AT{i}")
        AX, AY, AZ = g(arr_r, f"AX{i}"), g(arr_r, f"AY{i}"), g(arr_r, f"AZ{i}")
        CN, CO, CP = g(arr_r, f"CN{i}"), g(arr_r, f"CO{i}"), g(arr_r, f"CP{i}")
        r_angles.append(compute_hinging(AX,AY,AZ,AR,AS,AT,CN,CO,CP))

        AR, AS, AT = g(arr_h, f"AR{i}"), g(arr_h, f"AS{i}"), g(arr_h, f"AT{i}")
        AX, AY, AZ = g(arr_h, f"AX{i}"), g(arr_h, f"AY{i}"), g(arr_h, f"AZ{i}")
        CN, CO, CP = g(arr_h, f"CN{i}"), g(arr_h, f"CO{i}"), g(arr_h, f"CP{i}")
        h_angles.append(compute_hinging(AX,AY,AZ,AR,AS,AT,CN,CO,CP))

    # 2) 프레임 간 Δ (첫은 빈칸)
    r_delta = [""] + [round(r_angles[i]-r_angles[i-1],2) for i in range(1,10)]
    h_delta = [""] + [round(h_angles[i]-h_angles[i-1],2) for i in range(1,10)]

    # 3) 구간 Δ1-4, Δ4-6, 유지지수 (index 0=Frame1, 3=TOP, 5=DH)
    top_idx, dh_idx = 3, 5
    r_1_4 = round(r_angles[top_idx] - r_angles[0], 2)
    r_4_6 = round(r_angles[dh_idx] - r_angles[top_idx], 2)
    h_1_4 = round(h_angles[top_idx] - h_angles[0], 2)
    h_4_6 = round(h_angles[dh_idx] - h_angles[top_idx], 2)

    r_maint = signed_maintenance_score(r_angles[top_idx], r_angles[dh_idx])
    h_maint = signed_maintenance_score(h_angles[top_idx], h_angles[dh_idx])

    # 4) DataFrame 결합
    rows = list(range(1,11)) + ["1-4","4-6","Hinging_Maintenance"]
    data = {
        "Rory Hinging (deg)":    r_angles + [r_1_4, r_4_6, ""],
        "ΔRory":                  r_delta  + ["",    "",    r_maint],
        "Hong Hinging (deg)":    h_angles + [h_1_4, h_4_6, ""],
        "ΔHong":                  h_delta  + ["",    "",    h_maint],
    }
    df = pd.DataFrame(data, index=rows)
    df.index.name = "Frame"
    return df

def main():
    # 파일 경로 설정
    file_rory = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    file_hong = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # 힌징 테이블 생성
    df_hinge = calculate_full_hinging_table(file_rory, file_hong)

    # 결과 출력
    print(df_hinge)
if __name__ == "__main__":
    main()