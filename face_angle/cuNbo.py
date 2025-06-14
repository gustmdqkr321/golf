import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없이 엑셀을 읽어 numpy array 반환"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    예: 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    row = frame-1, col = 엑셀 열
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_cupping_bowing_angles(xlsx_path: Path):
    """
    단계별 공식에 따라 로컬 좌표계 생성 후,
    클럽 벡터의 X,Z 성분으로 cupping/bowing 각도(°)를 계산하여 리스트로 반환
    """
    arr = load_sheet(xlsx_path)
    angles = []
    for n in range(1, 11):
        # 관절/부위 좌표 추출
        AL, AM, AN = g(arr, f'AL{n}'), g(arr, f'AM{n}'), g(arr, f'AN{n}')  # 원어깨
        AR, AS, AT = g(arr, f'AR{n}'), g(arr, f'AS{n}'), g(arr, f'AT{n}')  # 왼팔꿈치
        AX, AY, AZ = g(arr, f'AX{n}'), g(arr, f'AY{n}'), g(arr, f'AZ{n}')  # 왼손목
        BM, BN, BO = g(arr, f'BM{n}'), g(arr, f'BN{n}'), g(arr, f'BO{n}')  # 오른손목
        CN, CO, CP = g(arr, f'CN{n}'), g(arr, f'CO{n}'), g(arr, f'CP{n}')  # 클럽헤드

        # ① 벡터 설정
        forearm     = np.array([AX-AR, AY-AS, AZ-AT])   # 팔꿈치→손목
        club        = np.array([CN-AX, CO-AY, CP-AZ])   # 손목→클럽헤드
        shoulder_w  = np.array([AX-AL, AY-AM, AZ-AN])   # 어깨→손목
        wrist_lr    = np.array([BM-AX, BN-AY, BO-AZ])   # 왼손목→오른손목

        # ② 로컬 좌표축 정의
        # X_local: forearm 방향 정규화
        X_local = forearm / np.linalg.norm(forearm)

        # Y_local: (shoulder_w × wrist_lr) 정규화
        Y_cross = np.cross(shoulder_w, wrist_lr)
        Y_local = Y_cross / np.linalg.norm(Y_cross)

        # Z_local: X_local × Y_local, 정규화
        Z_cross = np.cross(X_local, Y_local)
        Z_local = Z_cross / np.linalg.norm(Z_cross)

        # ③ 클럽 벡터를 로컬 좌표로 사영
        LocalClubX = club.dot(X_local)
        LocalClubZ = club.dot(Z_local)

        # ④ cupping/bowing 각도 계산 (°)
        theta = math.degrees(math.atan2(LocalClubZ, LocalClubX))
        angles.append(theta)

    return angles

if __name__ == '__main__':
    # 실제 경로로 수정하세요
    FILE1 = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")
    FILE2 = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    bowing1 = compute_cupping_bowing_angles(FILE1)
    bowing2 = compute_cupping_bowing_angles(FILE2)
    frames = list(range(1, 11))

    # 결과 DataFrame 출력
    df = pd.DataFrame({
        'Frame': frames,
        FILE1.stem: bowing1,
        FILE2.stem: bowing2
    })
    print(df.to_markdown(index=False))

    # 시각화
    plt.figure(figsize=(8,4))
    plt.plot(frames, bowing1, marker='o', label=FILE1.stem)
    plt.plot(frames, bowing2, marker='s', label=FILE2.stem)
    plt.xlabel('Frame')
    plt.ylabel('Cupping/Bowing (°)')
    plt.title('Frame-wise Cupping/Bowing Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
