import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy array 로 읽어들임"""
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """예: code='AX1' → arr[row=0, col=col_letters_to_index('AX')]"""
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_frame_thetas(xlsx_path: Path) -> list[float]:
    """
    각 프레임 n=1..10 에 대해
      F = (AXn−ALn, AYn−AMn, AZn−ANn)
      W = (BMn−AXn, BNn−AYn, BOn−AZn)
    사이의 각 θₙ = atan2(||F×W||, F·W) 를 도 단위로 반환
    """
    arr = load_sheet(xlsx_path)
    thetas = []
    for n in range(1, 11):
        F = np.array([
            g(arr, f'AX{n}') - g(arr, f'AL{n}'),
            g(arr, f'AY{n}') - g(arr, f'AM{n}'),
            g(arr, f'AZ{n}') - g(arr, f'AN{n}')
        ])
        W = np.array([
            g(arr, f'BM{n}') - g(arr, f'AX{n}'),
            g(arr, f'BN{n}') - g(arr, f'AY{n}'),
            g(arr, f'BO{n}') - g(arr, f'AZ{n}')
        ])
        dot   = np.dot(F, W)
        cross = np.cross(F, W)
        θ     = math.degrees(math.atan2(np.linalg.norm(cross), dot))
        thetas.append(θ)
    return thetas

def plot_thetas(frames: list[int],
                series: dict[str, list[float]],
                xlabel: str = 'Frame',
                ylabel: str = 'θ (°)',
                title: str = 'Frame-wise θ Comparison'):
    """
    여러 시리즈를 한 번에 플롯해 주는 범용 함수.
    - frames: x축 값 리스트 (예: [1,2,…,10])
    - series: {'label1': [θ1…θ10], 'label2': […], …}
    """
    plt.figure(figsize=(8,4))
    for label, thetas in series.items():
        plt.plot(frames, thetas, marker='o', label=label)
    plt.xticks(frames)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_consecutive_diffs(thetas: list[float]) -> list[float]:
    """θ2-θ1, θ3-θ2, …, θ10-θ9"""
    return [thetas[i] - thetas[i-1] for i in range(1, len(thetas))]

def compute_base_diffs(thetas: list[float]) -> list[float]:
    """θ2-θ1, θ3-θ1, …, θ10-θ1"""
    return [thetas[i] - thetas[0] for i in range(1, len(thetas))]

if __name__ == '__main__':
    # 파일 경로들
    file1 = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")
    file2 = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    # 1) 각 프레임 θ 계산
    θ1 = compute_frame_thetas(file1)
    θ2 = compute_frame_thetas(file2)
    frames = list(range(1, 11))

    # 2) 플롯 (재사용 가능한 함수 호출)
    plot_thetas(
        frames,
        {
            file1.stem: θ1,
            file2.stem: θ2
        },
        xlabel='Frame n',
        ylabel='θₙ (°)',
        title='Wrist Rolling Angle'
    )

    # 3) 구간별 차이값 출력
    print("=== 구간별 차이 (n → n+1) ===")
    for name, thetas in ((file1.stem, θ1), (file2.stem, θ2)):
        diffs = compute_consecutive_diffs(thetas)
        print(f"\n[{name}]")
        for i, d in enumerate(diffs, start=1):
            print(f" Frame {i}→{i+1}: {d:.2f}°")

    # 4) 1프레임 기준 차이값 출력
    print("\n=== 1프레임 기준 차이 (1 → n) ===")
    for name, thetas in ((file1.stem, θ1), (file2.stem, θ2)):
        diffs = compute_base_diffs(thetas)
        print(f"\n[{name}]")
        for i, d in enumerate(diffs, start=2):
            print(f" Frame 1→{i}: {d:.2f}°")
