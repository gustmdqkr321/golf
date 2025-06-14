import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# ---------------------------------------
# 설정
ROOT_DIR = Path("/Users/park_sh/Desktop/sim_pro")
GENERAL_QUERY = Path("/Users/park_sh/Desktop/sim_pro/test/kys.xlsx")  # 일반인 엑셀 경로
TOP_N = 5  # 상위 N명
# ---------------------------------------

def extract_features(xlsx: Path) -> np.ndarray:
    """
    셀 코드(예: 'CB1', 'AD3')를 이용해 데이터 배열에서 직접 값 추출.
    열-행 인덱스 방식으로 좌표 데이터를 읽고,
    g('B#') 로 시간(Time(sec))도 포함하여 계산.
    """
    # 전체 시트 읽기(header=None)
    df = pd.read_excel(xlsx, header=None)
    arr = df.values  # header=None 사용, 첫 번째 데이터 행부터 로드

    # 엑셀 열 문자를 0-based 인덱스로 변환 (A->0, B->1, ...)
    def col_letters_to_index(letters: str) -> int:
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
        return idx - 1

    # 셀 코드(예: 'CB1')에서 열 문자와 프레임 번호 추출해 값 리턴
    def g(code: str) -> float:
        letters = ''.join(filter(str.isalpha, code))
        num = int(''.join(filter(str.isdigit, code)))
        col = col_letters_to_index(letters)
        row = num - 1  # header=None 시트에서 프레임 번호와 일치하도록
        return float(arr[row, col])

    # 1) knee vs ankle 거리
    f1 = np.linalg.norm([
        g('CB1') - g('CK1'),
        g('CC1') - g('CL1'),
        g('CD1') - g('CM1')
    ])
    # 2) 선분1 거리: K1-BA1, BB1-L1
    f2 = np.hypot(g('K1') - g('BA1'), g('BB1') - g('L1'))
    # 3) 선분2 거리: BA1-AC1, AD1-BB1
    f3 = np.hypot(g('BA1') - g('AC1'), g('AD1') - g('BB1'))
    # 4) 합산 + CL1
    f4 = f1 + f2 + f3 + g('CL1')
    # 5) 선분3 거리: BB1-BH1, BA1-BG1
    f5 = np.hypot(g('BB1') - g('BH1'), g('BA1') - g('BG1'))
    # 6) 선분4a 거리: BH1-BM1, BG1-BM1
    f6 = np.hypot(g('BH1') - g('BM1'), g('BG1') - g('BM1'))
    # 7) 선분4b 거리: BM1-CN1
    f7 = abs(g('BM1') - g('CN1'))
    # 8) f6+f7+f1
    f8 = f6 + f7 + f1
    # 9) 절대합: AN1+BC1
    f9 = abs(g('AN1')) + abs(g('BC1'))

    # 10-16) 시간(Time(sec)) 프레임1,4,7,10
    b1 = g('B1')
    b4 = g('B4')
    b7 = g('B7')
    b10 = g('B10')
    f10 = b10
    f11 = b4 / (b10 - b4)
    f12 = b7
    f13 = b4
    f14 = b7 - b4
    f15 = b4 / (b7 - b4)
    f16 = (b10 - b7) / (b7 - b4)

    # 17-18) BM4-BM1, BN4-BN1
    f17 = g('BM4') - g('BM1')
    f18 = g('BN4') - g('BN1')
    # 19-21) AX4-AC4, AY4-AD4, AZ4-AE4
    f19 = g('AX4') - g('AC4')
    f20 = g('AY4') - g('AD4')
    f21 = g('AZ4') - g('AE4')

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9,
                     f10, f11, f12, f13, f14, f15, f16,
                     f17, f18, f19, f20, f21])


def load_player_data(root: Path):
    """
    driver 폴더만 탐색하여
    각 선수의 first_data_transition.xlsx로부터 피처 추출
    """
    driver_dir = root / 'driver'
    if not driver_dir.exists():
        raise FileNotFoundError(f'driver 디렉터리가 없습니다: {driver_dir}')
    names, feats = [], []
    for pdir in driver_dir.iterdir():
        if not pdir.is_dir() or pdir.name.lower() == 'test':
            continue
        xlsx = pdir / 'first_data_transition.xlsx'
        if xlsx.exists():
            names.append(pdir.name)
            feats.append(extract_features(xlsx))
    return names, np.vstack(feats)


def find_similar_vectors(names, feats, query_vec, top_n):
    nn = NearestNeighbors(n_neighbors=top_n, metric='cosine').fit(feats)
    dist, idx = nn.kneighbors(query_vec.reshape(1, -1))
    return dist[0], idx[0]


if __name__ == '__main__':
    # 데이터 로드
    player_names, player_feats = load_player_data(ROOT_DIR)
    if not GENERAL_QUERY.exists():
        raise FileNotFoundError(f'일반인 파일이 없습니다: {GENERAL_QUERY}')
    query_feats = extract_features(GENERAL_QUERY)

    # 유사도 계산
    dist, idx = find_similar_vectors(player_names, player_feats, query_feats, TOP_N)
    sims = [(player_names[i], float(dist[j])) for j, i in enumerate(idx)]

    # 결과 출력
    print(f"=== 유사 선수 Top {TOP_N} ===")
    for rank, (name, d) in enumerate(sims, start=1):
        print(f"{rank}: {name} (거리={d:.4f})")

    # === PCA 시각화 ===
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(player_feats)
    query_2d = pca.transform(query_feats.reshape(1, -1))

    plt.figure(figsize=(8, 6))
    plt.scatter(all_2d[:,0], all_2d[:,1], alpha=0.4, label='Players')
    plt.scatter(query_2d[0,0], query_2d[0,1], marker='*', s=200, c='red', label='Query')
    for i in idx:
        plt.scatter(all_2d[i,0], all_2d[i,1], edgecolors='black', s=100)
        plt.text(all_2d[i,0]+0.01, all_2d[i,1]+0.01, player_names[i], fontsize=9)

    plt.title('Player Features PCA Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
