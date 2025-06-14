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

def cocking3d(xlsx_path: Path) -> list[float]:
    """
    각 프레임 n=1..10 에 대해
      A⃗ = (ARn−AXn, ASn−AYn, ATn−AZn)
      C⃗ = (CNn−AXn, COn−AYn, CPn−AZn)
    사이의 각 θₙ = acos( (A·C) / (‖A‖‖C‖) ) 를 도 단위로 반환
    """
    arr = load_sheet(xlsx_path)
    thetas = []
    for n in range(1, 11):
        A = np.array([
            g(arr, f'AR{n}') - g(arr, f'AX{n}'),
            g(arr, f'AS{n}') - g(arr, f'AY{n}'),
            g(arr, f'AT{n}') - g(arr, f'AZ{n}')
        ])
        C = np.array([
            g(arr, f'CN{n}') - g(arr, f'AX{n}'),
            g(arr, f'CO{n}') - g(arr, f'AY{n}'),
            g(arr, f'CP{n}') - g(arr, f'AZ{n}')
        ])
        dot   = np.dot(A, C)
        normA = np.linalg.norm(A)
        normC = np.linalg.norm(C)
        cosθ  = max(-1.0, min(1.0, dot/(normA*normC)))
        θ     = math.degrees(math.acos(cosθ))
        thetas.append(θ)
    return thetas

def compute_consecutive_diffs(thetas: list[float]) -> list[float]:
    """[θ2-θ1, θ3-θ2, …, θ10-θ9]"""
    return [thetas[i] - thetas[i-1] for i in range(1, len(thetas))]

def cocking2d(xlsx_path: Path):
    """
    이미지 공식 θ = cos⁻¹(
      (AS−AY)(CO−AY)+(AT−AZ)(CP−AZ)
      ——————————————————————————————
      √((AS−AY)²+(AT−AZ)²) × √((CO−AY)²+(CP−AZ)²)
    )
    를 1..10 프레임에 대해 계산해 프린트만 수행
    """
    arr = load_sheet(xlsx_path)
    thetas = []
    for n in range(1, 11):
        dAY = g(arr, f'AY{n}')
        v1y = g(arr, f'AS{n}') - dAY
        v1z = g(arr, f'AT{n}') - g(arr, f'AZ{n}')
        v2y = g(arr, f'CO{n}') - dAY
        v2z = g(arr, f'CP{n}') - g(arr, f'AZ{n}')

        num = v1y * v2y + v1z * v2z
        den = math.sqrt(v1y**2 + v1z**2) * math.sqrt(v2y**2 + v2z**2)
        cosθ = max(-1.0, min(1.0, num/den))
        θ    = math.degrees(math.acos(cosθ))
        thetas.append(θ)

    return thetas


if __name__ == '__main__':
    file1 = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")
    file2 = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")

    # 1) θ 리스트 계산
    θ1 = cocking3d(file1)
    θ2 = cocking3d(file2)
    frames = list(range(1, 11))

    # 2) 기존 플롯 유지
    plt.figure(figsize=(8,4))
    plt.plot(frames, θ1, marker='o', label=file1.stem)
    plt.plot(frames, θ2, marker='s', label=file2.stem)
    plt.xticks(frames)
    plt.xlabel('Frame n')
    plt.ylabel('θₙ (°)')
    plt.title('각 프레임별 A–C 벡터 간 θₙ (acos 공식)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) 구간별 차이(1→2, 2→3, …) 프린트
    print("\n=== 구간별 차이 (Frame n → n+1) ===")
    for name, thetas in ((file1.stem, θ1), (file2.stem, θ2)):
        diffs = compute_consecutive_diffs(thetas)
        print(f"\n[{name}]")
        for i, d in enumerate(diffs, start=1):
            print(f" {i}→{i+1}: {d:.2f}°")

    # 4) 2D Cos⁻¹ 공식 결과 프린트
    cocking2d(file1)
    cocking2d(file2)
