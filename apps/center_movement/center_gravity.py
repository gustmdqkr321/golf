# swing_delta.py

import numpy as np
import pandas as pd
import math
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy 2D 배열로 읽어들임"""
    # pandas 로 전체 시트를 header=None 으로 읽고 .values 반환
    import pandas as _pd
    return _pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    arr, 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    숫자는 1-based frame index, 문자는 컬럼 레이블
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_delta_x_table(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    두 파일에 대해 ΔX 변화를 표 형태로 반환
    • index: ['1','2',…,'10','1-4','4-7','7-10','1-7','Total']
    • columns: ['Pro ΔX','Golfer ΔX','Pro ΔX diff','Golfer ΔX diff']
    """
    # 1) 원본 불러오기
    arr1 = load_sheet(path_pro)
    arr2 = load_sheet(path_golfer)

    # 2) 각 프레임 ΔX 계산
    def com_x(arr, n):
        # COM 위치 (X축) 계산
        return (
            0.08 * g(arr, f"AC{n}") +
            0.35 * (g(arr, f"AL{n}") + g(arr, f"BA{n}"))/2 +
            0.30 * (g(arr, f"H{n}")  + g(arr, f"K{n}"))/2 +
            0.15 * (g(arr, f"BP{n}") + g(arr, f"CB{n}"))/2 +
            0.12 * (g(arr, f"BY{n}") + g(arr, f"CK{n}"))/2
        )
    def base_x(arr, n):
        # 발 중심 (60% 지점) X기준 계산 (간단화)
        return (
            0.4 * (g(arr, f"BY{n}") + g(arr, f"CK{n}"))/2 +
            0.6 * (g(arr, f"BS{n}") + g(arr, f"CE{n}"))/2
        )

    deltas1 = []
    deltas2 = []
    for i in range(1, 11):
        dx1 = round(com_x(arr1, i) - base_x(arr1, i), 2)
        dx2 = round(com_x(arr2, i) - base_x(arr2, i), 2)
        deltas1.append(dx1)
        deltas2.append(dx2)

    # 3) 프레임 간 변화 ΔΔX
    diff1 = [None]
    diff2 = [None]
    for i in range(1, 10):
        diff1.append(round(deltas1[i] - deltas1[i-1], 2))
        diff2.append(round(deltas2[i] - deltas2[i-1], 2))

    # 4) 구간별 ΔΔX (구간: 1-4, 4-7, 7-10, 1-7)
    segs = [("1-4", 0, 3), ("4-7", 3, 6), ("7-10", 6, 9), ("1-7", 0, 6)]
    seg1 = [round(deltas1[e] - deltas1[s], 2) for _, s, e in segs]
    seg2 = [round(deltas2[e] - deltas2[s], 2) for _, s, e in segs]

    # 5) 총 절대 변화량 (Total = Σ|ΔΔX|)
    total1 = round(sum(abs(d) for d in diff1[1:]), 2)
    total2 = round(sum(abs(d) for d in diff2[1:]), 2)

    # 6) DataFrame 조립
    idx = [str(i) for i in range(1, 11)] + [label for label,_,_ in segs] + ["Total"]
    data = {
        "Pro ΔX":            deltas1 + [None]*5,
        "Golfer ΔX":         deltas2 + [None]*5,
        "Pro ΔX diff":       diff1    + seg1 + [total1],
        "Golfer ΔX diff":    diff2    + seg2 + [total2],
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Frame"
    return df

def compute_delta_y_table(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • Y축 COM 높이(=Y_COM) 프레임별 값과 변화량을 테이블로 반환
    • columns: ['Pro Y','Golfer Y','Pro Y diff','Golfer Y diff']
    • index: ['1','2',…,'10','1-4','4-7','7-10','1-7','Total']
    """
    arr1 = load_sheet(path_pro)
    arr2 = load_sheet(path_golfer)

    # Y_COM 공식: 0.09*AD + 0.40*(AM+BB)/2 + 0.34*(I+L)/2 + 0.17*(BQ+CC)/2
    def com_y(arr, n):
        return round(
            0.09 * g(arr, f"AD{n}") +
            0.40 * (g(arr, f"AM{n}") + g(arr, f"BB{n}"))/2 +
            0.34 * (g(arr, f"I{n}")  + g(arr, f"L{n}"))/2 +
            0.17 * (g(arr, f"BQ{n}") + g(arr, f"CC{n}"))/2
        , 2)

    # 1~10 프레임 Y값
    y1 = [com_y(arr1, i) for i in range(1, 11)]
    y2 = [com_y(arr2, i) for i in range(1, 11)]

    # 프레임 간 변화 ΔY
    diff1 = [None] + [round(y1[i] - y1[i-1], 2) for i in range(1, 10)]
    diff2 = [None] + [round(y2[i] - y2[i-1], 2) for i in range(1, 10)]

    # 구간 변화 (1-4, 4-7, 7-10, 1-7)
    segs = [("1-4", 0, 3), ("4-7", 3, 6), ("7-10", 6, 9), ("1-7", 0, 6)]
    seg1 = [round(y1[e] - y1[s], 2) for _, s, e in segs]
    seg2 = [round(y2[e] - y2[s], 2) for _, s, e in segs]

    # Total 절대 변화량
    total1 = round(sum(abs(d) for d in diff1[1:]), 2)
    total2 = round(sum(abs(d) for d in diff2[1:]), 2)

    # DataFrame 조립
    idx = [str(i) for i in range(1, 11)] + [label for label,_,_ in segs] + ["Total"]
    data = {
        "Pro Y":         y1   + [None]*5,
        "Golfer Y":      y2   + [None]*5,
        "Pro Y diff":    diff1 + seg1 + [total1],
        "Golfer Y diff": diff2 + seg2 + [total2],
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Frame"
    return df

def compute_delta_z_table(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • Z축 COM 편측 정도(=laterality) 프레임별 값과 변화량을 테이블로 반환
    • 공식 ❶: ((BR+CD)/2 + (J+M)/2 + (AN+BC)/2 + AE)/4 - ((CA+CM)/2)
    • columns: ['Pro Z','Golfer Z','Pro Z diff','Golfer Z diff']
    • index 동일
    """
    arr1 = load_sheet(path_pro)
    arr2 = load_sheet(path_golfer)

    def com_z(arr, n):
        knee  = (g(arr, f"BR{n}") + g(arr, f"CD{n}"))/2
        hip   = (g(arr, f"J{n}")  + g(arr, f"M{n}"))/2
        chest = (g(arr, f"AN{n}") + g(arr, f"BC{n}"))/2
        head  = g(arr, f"AE{n}")
        foot  = (g(arr, f"CA{n}") + g(arr, f"CM{n}"))/2
        return round(((knee + hip + chest + head)/4) - foot, 2)

    # 1~10 프레임 Z값
    z1 = [com_z(arr1, i) for i in range(1, 11)]
    z2 = [com_z(arr2, i) for i in range(1, 11)]

    # 프레임 간 변화 ΔZ
    diff1 = [None] + [round(z1[i] - z1[i-1], 2) for i in range(1, 10)]
    diff2 = [None] + [round(z2[i] - z2[i-1], 2) for i in range(1, 10)]

    # 구간 변화 (1-4, 4-7, 7-10, 1-7)
    segs = [("1-4", 0, 3), ("4-7", 3, 6), ("7-10", 6, 9), ("1-7", 0, 6)]
    seg1 = [round(z1[e] - z1[s], 2) for _, s, e in segs]
    seg2 = [round(z2[e] - z2[s], 2) for _, s, e in segs]

    # Total 절대 변화량
    total1 = round(sum(abs(d) for d in diff1[1:]), 2)
    total2 = round(sum(abs(d) for d in diff2[1:]), 2)

    # DataFrame 조립
    idx = [str(i) for i in range(1, 11)] + [label for label,_,_ in segs] + ["Total"]
    data = {
        "Pro Z":         z1   + [None]*5,
        "Golfer Z":      z2   + [None]*5,
        "Pro Z diff":    diff1 + seg1 + [total1],
        "Golfer Z diff": diff2 + seg2 + [total2],
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Frame"
    return df

def compute_summary_table(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • X/Y/Z 각 축별 구간(1-4,4-7,7-10)과 Total ΔΔ 값을 모아
      프로 vs 골퍼 비교표( MultiIndex )를 반환
    • index.names = ['Axis','Segment'], columns = ['Pro','Golfer']
    """
    # 1) 각각의 Δ 테이블 불러오기
    df_x = compute_delta_x_table(path_pro, path_golfer)
    df_y = compute_delta_y_table(path_pro, path_golfer)
    df_z = compute_delta_z_table(path_pro, path_golfer)

    # 2) 사용할 세그먼트 리스트와 (DataFrame, 컬럼명) 매핑
    segs = ["1-4","4-7","7-10","Total"]
    axes = [
        ("X", df_x, "Pro ΔX diff",    "Golfer ΔX diff"),
        ("Y", df_y, "Pro Y diff",     "Golfer Y diff"),
        ("Z", df_z, "Pro Z diff",     "Golfer Z diff"),
    ]

    # 3) MultiIndex 인덱스 생성
    tuples = [(ax, seg) for ax, *_ in axes for seg in segs] + [("Total","")]
    idx = pd.MultiIndex.from_tuples(tuples, names=["Axis","Segment"])

    # 4) 빈 프레임 생성
    summary = pd.DataFrame(index=idx, columns=["Pro","Golfer"], dtype=float)

    # 5) 각 축·세그먼트별 값 채우기
    for axis, df, col_p, col_g in axes:
        for seg in segs:
            summary.loc[(axis, seg), "Pro"]    = df.at[seg, col_p]
            summary.loc[(axis, seg), "Golfer"] = df.at[seg, col_g]

    # 6) 맨 마지막 Total 행에는 각 축 Total 합계
    pro_tot   = sum(summary.loc[(ax,"Total"), "Pro"] for ax, *_ in axes)
    golfer_tot= sum(summary.loc[(ax,"Total"), "Golfer"] for ax, *_ in axes)
    summary.loc[("Total",""), "Pro"]    = pro_tot
    summary.loc[("Total",""), "Golfer"] = golfer_tot

    return summary

def compute_smdi_mrmi(
    pro_path: Path,
    golfer_path: Path,
    pro_label: str    = "Rory",
    golfer_label: str = "Hong"
) -> pd.DataFrame:
    """
    Returns a 2×4 DataFrame:

      index      SMDI    MRMI X   MRMI Y   MRMI Z
      ------------------------------------------------
      pro_label    100       0        0        0
      golfer_label  SMDI*   MRMIx   MRMIy   MRMIz

    where:
      • SMDI* = (score_x + score_y + score_z)/3  
        score_axis = 100 - (|GolferTotal - ProTotal|/ProTotal * 100)

      • MRMI_axis = (GolferTotal - ProTotal)/ProTotal * 100
    """
    # 1) 요약 테이블에서 'Total' 값 읽어오기
    summary = compute_summary_table(pro_path, golfer_path)
    axes = ["X","Y","Z"]
    pro_tot  = {ax: summary.at[(ax,"Total"),   "Pro"]    for ax in axes}
    gol_tot  = {ax: summary.at[(ax,"Total"), "Golfer"] for ax in axes}

    # 2) MRMI 계산
    mrmi = {
        ax: round((gol_tot[ax] - pro_tot[ax]) / pro_tot[ax] * 100, 2)
        for ax in axes
    }

    # 3) 편차율 점수 (Swing Movement Deviation Score)
    #    = 100 - (|실제 - 표준|/표준 *100)
    scores = {
        ax: 100 - abs(gol_tot[ax] - pro_tot[ax]) / pro_tot[ax] * 100
        for ax in axes
    }
    smdi = round(sum(scores.values()) / 3, 2)

    # 4) 결과 DataFrame
    df = pd.DataFrame(
        [
            {"SMDI":100,                  "MRMI X":0,         "MRMI Y":0,        "MRMI Z":0},
            {"SMDI":smdi, "MRMI X":mrmi["X"], "MRMI Y":mrmi["Y"], "MRMI Z":mrmi["Z"]},
        ],
        index=[pro_label, golfer_label],
        columns=["SMDI","MRMI X","MRMI Y","MRMI Z"]
    )
    return df

if __name__ == "__main__":
    # 테스트용 예시 경로
    pro = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golf = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # SMDI/MRMI 계산
    result = compute_smdi_mrmi(pro, golf, "Rory McIlroy", "Hong")
    print(result)