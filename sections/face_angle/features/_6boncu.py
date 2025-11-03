# bowing_table.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# A1 유틸
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# 보잉/커핑 계산
# ──────────────────────────────────────────────────────────────────────
def compute_bowing_angles_from_array(arr: np.ndarray) -> list[float]:
    """
    각 프레임 1..10에 대해 보잉/커핑 각도(°) 계산 (회전행렬 방식).
    """
    angles: list[float] = []
    for n in range(1, 11):
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
        nx = np.linalg.norm(x_axis)
        if nx != 0:
            x_axis = x_axis / nx

        # 2) 어깨-손목 면과 클럽벡터의 수직방향을 y축으로
        y_axis = np.cross(W - S, C - W)
        ny = np.linalg.norm(y_axis)
        if ny != 0:
            y_axis = y_axis / ny

        # 3) z축 = x × y
        z_axis = np.cross(x_axis, y_axis)
        nz = np.linalg.norm(z_axis)
        if nz != 0:
            z_axis = z_axis / nz

        # 4) 회전행렬
        R = np.vstack([x_axis, y_axis, z_axis]).T  # 3×3

        # 5) 클럽벡터를 로컬좌표로 투영
        local = R.T @ (C - W)

        # 6) X–Z 평면 기울기 → arctan2(v_z, v_x)
        theta = math.degrees(math.atan2(float(local[2]), float(local[0])))
        angles.append(theta)
    return angles

def compute_bowing_angles(path: Path) -> list[float]:
    """파일 경로 버전"""
    return compute_bowing_angles_from_array(load_sheet(path))

def normalize_angles(angles: list[float]) -> list[float]:
    """±90° 벗어나면 ±180° 보정"""
    out: list[float] = []
    for th in angles:
        if th > 90:
            out.append(180 - th)
        elif th < -90:
            out.append(-180 - th)
        else:
            out.append(th)
    return out

def signed_maintenance_score(top: float, dh: float) -> float:
    """TOP→DH 사이 signed 유지지수 (percent)"""
    delta = dh - top
    ratio = abs(delta) / abs(top) if top != 0 else 0.0
    if math.copysign(1.0, top) == math.copysign(1.0, dh) and abs(dh) <= abs(top):
        return round((1 - ratio) * 100.0, 2)
    else:
        return round(-ratio * 100.0, 2)

# ──────────────────────────────────────────────────────────────────────
# 유사도 (요청하신 bowing_sim 규칙)
#  - 방향 일치 점수: 50점 만점
#  - 크기 유사도 점수: 50점 만점 (차이 평균을 최대 절댓값으로 나눈 뒤 1-비율)
# ──────────────────────────────────────────────────────────────────────
def bowing_sim(pro_rel: np.ndarray, ama_rel: np.ndarray) -> float:
    R = np.asarray(pro_rel, dtype=float)
    H = np.asarray(ama_rel, dtype=float)
    if R.shape != H.shape:
        raise ValueError("pro_rel, ama_rel 는 같은 길이여야 합니다.")

    # 방향(부호) 일치 점수
    sign_match = (np.sign(R) == np.sign(H)).astype(float)
    sign_score = float(np.mean(sign_match) * 50.0)

    # 크기 유사도 점수
    diffs = np.abs(R - H)
    max_val = float(np.max(np.abs(R))) if np.max(np.abs(R)) != 0 else 1.0
    magnitude_score = float((1.0 - np.mean(diffs / max_val)) * 50.0)

    return round(sign_score + magnitude_score, 2)

# ──────────────────────────────────────────────────────────────────────
# 표 생성 (배열 입력 버전: 앱 컨텍스트에 맞춤)
# ──────────────────────────────────────────────────────────────────────
def build_bowing_table_from_arrays(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    Bowing/Cupping 테이블 생성 (유사도 포함)
      • index: ['ADD','BH','BH2','TOP','TR','DH','IMP','FH1','FH2','FIN',
                '1-4','4-6','Bowing_Maintenance','Similarity']
      • columns: ['Pro Rel. Bowing (°)','Ama Rel. Bowing (°)',
                  'Pro ΔRel. Bowing','Ama ΔRel. Bowing','Similarity']
    """
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

    raw_pro = normalize_angles(compute_bowing_angles_from_array(pro_arr))
    raw_ama = normalize_angles(compute_bowing_angles_from_array(ama_arr))
    rel_pro = np.array(raw_pro, dtype=float) - float(raw_pro[0])
    rel_ama = np.array(raw_ama, dtype=float) - float(raw_ama[0])

    # Δ (첫 프레임 NaN → 포맷 에러 방지)
    d_pro = [np.nan] + [rel_pro[i] - rel_pro[i-1] for i in range(1, len(rel_pro))]
    d_ama = [np.nan] + [rel_ama[i] - rel_ama[i-1] for i in range(1, len(rel_ama))]

    # 구간 Δ 및 유지지수 (ADD=0, TOP=3, DH=5)
    pro_1_4 = rel_pro[3] - rel_pro[0]
    pro_4_6 = rel_pro[5] - rel_pro[3]
    ama_1_4 = rel_ama[3] - rel_ama[0]
    ama_4_6 = rel_ama[5] - rel_ama[3]
    pro_maint = signed_maintenance_score(rel_pro[3], rel_pro[5])
    ama_maint = signed_maintenance_score(rel_ama[3], rel_ama[5])

    # 유사도
    sim = bowing_sim(rel_pro, rel_ama)

    # 소수점 둘째자리로 반올림
    r2 = lambda x: float(np.round(x, 2)) if isinstance(x, (int, float, np.floating)) and not np.isnan(x) else x
    fmt = np.vectorize(r2)

    idx = labels + ["1-4","4-6","Bowing_Maintenance","Similarity"]
    data = {
        "seg":                 idx,
        "Pro Rel. Bowing (°)":  list(fmt(rel_pro)) + [r2(pro_1_4), r2(pro_4_6), np.nan, np.nan],
        "Ama Rel. Bowing (°)":  list(fmt(rel_ama)) + [r2(ama_1_4), r2(ama_4_6), np.nan, np.nan],
        "Pro ΔRel. Bowing":     list(fmt(d_pro))   + [np.nan,      np.nan,      r2(pro_maint), np.nan],
        "Ama ΔRel. Bowing":     list(fmt(d_ama))   + [np.nan,      np.nan,      r2(ama_maint), np.nan],
        "Similarity":           [np.nan]*len(labels) + [np.nan, np.nan, np.nan, r2(sim)],
    }
    df = pd.DataFrame(data)
    return df

# ──────────────────────────────────────────────────────────────────────
# 표 생성 (파일 경로 버전)
# ──────────────────────────────────────────────────────────────────────
def compute_table(path_pro: Path, path_ama: Path) -> pd.DataFrame:
    arr_pro = load_sheet(path_pro)
    arr_ama = load_sheet(path_ama)
    return build_bowing_table_from_arrays(arr_pro, arr_ama)

# ──────────────────────────────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    path = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    df = compute_table(path, path)
    print(df)
