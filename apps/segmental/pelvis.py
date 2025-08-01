# pelvis_analysis.py

import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy 2D 배열로 읽어들임"""
    import pandas as _pd
    return _pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    arr, 'H1' → arr[row=0, col=col_letters_to_index('H')]
    숫자는 1-based frame index, 문자는 컬럼 레이블
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def _pelvis_center(arr: np.ndarray, i: int) -> np.ndarray:
    """
    프레임 i 에서 골반 중심 좌표 (cm) 반환
    왼쪽 골반: H_i, I_i, J_i
    오른쪽 골반: K_i, L_i, M_i
    """
    L = np.array([g(arr, f"H{i}"), g(arr, f"I{i}"), g(arr, f"J{i}")])
    R = np.array([g(arr, f"K{i}"), g(arr, f"L{i}"), g(arr, f"M{i}")])
    return (L + R) / 2

def _pelvis_vector(arr: np.ndarray, i: int) -> np.ndarray:
    """
    프레임 i 에서 양측 골반 벡터 R - L 반환
    """
    L = np.array([g(arr, f"H{i}"), g(arr, f"I{i}"), g(arr, f"J{i}")])
    R = np.array([g(arr, f"K{i}"), g(arr, f"L{i}"), g(arr, f"M{i}")])
    return R - L

def compute_pelvis_tdd_table(
    pro_path: Path,
    golfer_path: Path,
    deg2m: float = 0.01
) -> pd.DataFrame:
    """
    골반 TDD 분석 테이블 반환
    • index: ['1-2','2-3',…,'9-10','1-4','4-7','7-10','Total']
    • columns:
        Distance_pro(m), Rotation_pro(m), Angle_pro(°), TDD_pro(m),
        Distance_golfer(m), Rotation_golfer(m), Angle_golfer(°), TDD_golfer(m)
    """
    arr1 = load_sheet(pro_path)
    arr2 = load_sheet(golfer_path)
    N = 10

    # 1–9 프레임 간 key 생성
    seg_keys = [f"{i}-{i+1}" for i in range(1, N)]
    D1, D2, θ1, θ2 = [], [], [], []

    for i in range(1, N):
        # 이동 거리 (cm→m)
        d1 = np.linalg.norm(_pelvis_center(arr1, i+1) - _pelvis_center(arr1, i)) / 100
        d2 = np.linalg.norm(_pelvis_center(arr2, i+1) - _pelvis_center(arr2, i)) / 100
        D1.append(d1); D2.append(d2)

        # 회전 각 (degree)
        def ang(a, b):
            c = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            return np.degrees(np.arccos(np.clip(c, -1, 1)))

        v1, v1p = _pelvis_vector(arr1, i),   _pelvis_vector(arr1, i+1)
        v2, v2p = _pelvis_vector(arr2, i),   _pelvis_vector(arr2, i+1)
        θ1.append(ang(v1, v1p))
        θ2.append(ang(v2, v2p))

    # 회전 거리 & TDD
    R1 = [t * deg2m for t in θ1]
    R2 = [t * deg2m for t in θ2]
    T1 = [d + r for d, r in zip(D1, R1)]
    T2 = [d + r for d, r in zip(D2, R2)]

    # 기본 DataFrame
    df = pd.DataFrame({
        "Distance_pro(m)":     [round(x,2) for x in D1],
        "Rotation_pro(m)":     [round(x,2) for x in R1],
        "Angle_pro(°)":        [round(x,2) for x in θ1],
        "TDD_pro(m)":          [round(x,2) for x in T1],
        "Distance_golfer(m)":  [round(x,2) for x in D2],
        "Rotation_golfer(m)":  [round(x,2) for x in R2],
        "Angle_golfer(°)":     [round(x,2) for x in θ2],
        "TDD_golfer(m)":       [round(x,2) for x in T2],
    }, index=seg_keys)

    # 구간 합계 (1-4,4-7,7-10)
    sections = {
        "1-4": seg_keys[0:3],
        "4-7": seg_keys[3:6],
        "7-10": seg_keys[6:9],
    }
    for seg, keys in sections.items():
        df.loc[seg] = df.loc[keys].sum().round(2)

    # Total 행 (세 구간만 합산)
    df.loc["Total"] = df.loc[list(sections.keys())].sum().round(2)
    return df

def compute_pelvis_rotation_table(
    pro_path: Path,
    golfer_path: Path
) -> pd.DataFrame:
    """
    골반 수평/수직 회전각 분석 테이블 반환
    • index: ['1-2','2-3',…,'9-10','1-4','4-7','7-10','Total']
    • columns:
        Horizontal Angle_pro(°), Horizontal Angle_golfer(°),
        Vertical Angle_pro(°),   Vertical Angle_golfer(°)
    """
    arr_p = load_sheet(pro_path)
    arr_h = load_sheet(golfer_path)
    N = 10

    seg_keys = [f"{i}-{i+1}" for i in range(1, N)]
    H_p, H_h = [], []
    raw_V_p, raw_V_h = [], []

    for i in range(1, N):
        # 수평 회전각 (XZ 평면)
        vp1 = _pelvis_vector(arr_p, i)[[0,2]]
        vp2 = _pelvis_vector(arr_p, i+1)[[0,2]]
        vh1 = _pelvis_vector(arr_h, i)[[0,2]]
        vh2 = _pelvis_vector(arr_h, i+1)[[0,2]]
        def horiz(a,b):
            dot   = a.dot(b)
            cross = a[0]*b[1] - a[1]*b[0]
            return np.degrees(np.arctan2(cross, dot))
        ang_p, ang_h = horiz(vp1,vp2), horiz(vh1,vh2)
        H_p.append(+abs(ang_p) if i<4 else -abs(ang_p))
        H_h.append(+abs(ang_h) if i<4 else -abs(ang_h))

        # 수직 회전각 (Y축 tilt)
        def slope_ang(arr, idx):
            v = _pelvis_vector(arr, idx)
            return np.degrees(np.arctan2(v[1], np.hypot(v[0],v[2])))
        s1p, s2p = slope_ang(arr_p,i), slope_ang(arr_p,i+1)
        s1h, s2h = slope_ang(arr_h,i), slope_ang(arr_h,i+1)
        raw_V_p.append(s2p - s1p)
        raw_V_h.append(s2h - s1h)

    # 수직 각 부호 처리
    V_p, V_h = [], []
    sign_p, sign_h = 1, 1
    for idx in range(len(raw_V_p)):
        if idx==0:
            V_p.append(abs(raw_V_p[0]))
            V_h.append(abs(raw_V_h[0]))
        else:
            if raw_V_p[idx]*raw_V_p[idx-1] < 0: sign_p *= -1
            if raw_V_h[idx]*raw_V_h[idx-1] < 0: sign_h *= -1
            V_p.append(sign_p * abs(raw_V_p[idx]))
            V_h.append(sign_h * abs(raw_V_h[idx]))

    # DataFrame + 구간 합계 + Total
    df = pd.DataFrame({
        "Horizontal Angle_pro(°)":    np.round(H_p,2),
        "Horizontal Angle_golfer(°)": np.round(H_h,2),
        "Vertical Angle_pro(°)":      np.round(V_p,2),
        "Vertical Angle_golfer(°)":   np.round(V_h,2),
    }, index=seg_keys)

    sections = {
        "1-4": seg_keys[0:3],
        "4-7": seg_keys[3:6],
        "7-10": seg_keys[6:9],
    }
    for seg, keys in sections.items():
        df.loc[seg] = df.loc[keys].sum().round(2)

    df.loc["Total"] = df.loc[list(sections.keys())].abs().sum().round(2)
    return df

if __name__ == "__main__":
    pro  = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golf = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    print("▶ 골반 TDD 분석 ▶")
    print(compute_pelvis_tdd_table(pro, golf).to_string(), "\n")
    print("▶ 골반 회전각 분석 ▶")
    print(compute_pelvis_rotation_table(pro, golf).to_string())
