# bowing_table.py

import math
import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(예: 'AX')를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy 2D 배열로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    예: code='AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame_index = (숫자 - 1)
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_bowing_angles(path: Path) -> list[float]:
    """
    각 프레임 1..10에 대해 보잉/커핑 각도(°) 계산 (회전행렬 방식).
    """
    arr = load_sheet(path)
    angles = []
    for n in range(1,11):
        # 좌표 불러오기
        AR,AS,AT = g(arr, f"AR{n}"), g(arr, f"AS{n}"), g(arr, f"AT{n}")
        AX,AY,AZ = g(arr, f"AX{n}"), g(arr, f"AY{n}"), g(arr, f"AZ{n}")
        AL,AM,AN = g(arr, f"AL{n}"), g(arr, f"AM{n}"), g(arr, f"AN{n}")
        CN,CO,CP = g(arr, f"CN{n}"), g(arr, f"CO{n}"), g(arr, f"CP{n}")

        E = np.array([AR, AS, AT])  # 팔꿈치
        W = np.array([AX, AY, AZ])  # 손목
        S = np.array([AL, AM, AN])  # 어깨
        C = np.array([CN, CO, CP])  # 클럽헤드

        # 1) 손목→팔꿈치 벡터를 x축으로
        x_axis = W - E
        norm = np.linalg.norm(x_axis)
        if norm!=0: x_axis /= norm

        # 2) 어깨-손목 면과 클럽벡터의 수직방향을 y축으로
        y_axis = np.cross(W - S, C - W)
        norm = np.linalg.norm(y_axis)
        if norm!=0: y_axis /= norm

        # 3) z축 = x × y
        z_axis = np.cross(x_axis, y_axis)
        norm = np.linalg.norm(z_axis)
        if norm!=0: z_axis /= norm

        # 4) 회전행렬 R 구성
        R = np.vstack([x_axis, y_axis, z_axis]).T  # 3×3

        # 5) 클럽벡터를 로컬좌표로 투영
        local = R.T @ (C - W)

        # 6) X–Z 평면 기울기 → arctan2(v_z, v_x)
        theta = math.degrees(math.atan2(local[2], local[0]))
        angles.append(theta)
    return angles

def normalize_angles(angles: list[float]) -> list[float]:
    """
    ±90° 벗어나면 ±180° 보정
    """
    out = []
    for th in angles:
        if th > 90:
            out.append(180 - th)
        elif th < -90:
            out.append(-180 - th)
        else:
            out.append(th)
    return out

def signed_maintenance_score(top: float, dh: float) -> float:
    """
    TOP→DH 사이 signed 유지지수 (percent)
    """
    delta = dh - top
    ratio = abs(delta) / abs(top) if top!=0 else 0.0
    if np.sign(top)==np.sign(dh) and abs(dh)<=abs(top):
        return round((1 - ratio)*100, 2)
    else:
        return round(-ratio*100, 2)

def compute_table(path1: Path, path2: Path) -> pd.DataFrame:
    """
    두 파일에 대해 Bowing/Cupping 테이블을 생성하여 반환
      • index: ['ADD','BH','BH2','TOP','TR','DH','IMP','FH1','FH2','FIN',
                '1-4','4-6','Bowing_Maintenance']
      • columns: ['Rory Rel. Bowing (°)','Hong Rel. Bowing (°)',
                  'Rory ΔRel. Bowing','Hong ΔRel. Bowing']
    """
    # 1) 프레임 라벨
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

    # 2) 각 프레임 bowing 각도 계산 → 보정 → 상대각
    raw1 = normalize_angles(compute_bowing_angles(path1))
    raw2 = normalize_angles(compute_bowing_angles(path2))
    rel1 = np.array(raw1) - raw1[0]
    rel2 = np.array(raw2) - raw2[0]

    # 3) 변화량 ΔRel (첫 프레임은 빈칸)
    delta1 = [None] + [round(rel1[i] - rel1[i-1],2) for i in range(1, len(rel1))]
    delta2 = [None] + [round(rel2[i] - rel2[i-1],2) for i in range(1, len(rel2))]

    # 4) 구간별 Δ(1-4,4-6) 및 유지지수
    # 인덱스: ADD=0, TOP=3, DH=5
    d1_4_1 = round(rel1[3] - rel1[0],2)
    d4_6_1 = round(rel1[5] - rel1[3],2)
    m1     = signed_maintenance_score(rel1[3], rel1[5])

    d1_4_2 = round(rel2[3] - rel2[0],2)
    d4_6_2 = round(rel2[5] - rel2[3],2)
    m2     = signed_maintenance_score(rel2[3], rel2[5])

    # 5) DataFrame 조립
    idx = labels + ["1-4","4-6","Bowing_Maintenance"]
    data = {
        "Rory Rel. Bowing (°)":  list(map(lambda x: round(x,2), rel1)) + [d1_4_1, d4_6_1, None],
        "Hong Rel. Bowing (°)":  list(map(lambda x: round(x,2), rel2)) + [d1_4_2, d4_6_2, None],
        "Rory ΔRel. Bowing":     delta1 +       [None, None, m1],
        "Hong ΔRel. Bowing":     delta2 +       [None, None, m2],
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Frame"
    return df

if __name__=="__main__":
    # 간단 테스트
    path = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    df = compute_table(path, path)
    print(df)
