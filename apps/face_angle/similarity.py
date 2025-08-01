import numpy as np


def rolling_sim(
    rory_rolls: np.ndarray,
    hong_rolls: np.ndarray,
    alpha: float = 2.0
) -> float:
    """
    로리와 홍의 순수 롤링값 배열을 받아 부호가 다르면 차이를 alpha배 가중하고
    0~100 사이의 유사도 점수를 반환합니다.
    """
    R = np.asarray(rory_rolls, dtype=float)
    H = np.asarray(hong_rolls, dtype=float)
    if R.shape != H.shape:
        raise ValueError("rory_rolls, hong_rolls 는 같은 길이여야 합니다.")

    # 1) 부호 일치 여부
    same_sign = np.sign(R) == np.sign(H)

    # 2) 절댓값 차이
    diffs = np.abs(R - H)

    # 3) 부호 다를 때 alpha 배 가중
    weighted = np.where(same_sign, diffs, alpha * diffs)

    # 4) 총합/최댓값/갯수
    total_diff = weighted.sum()
    max_diff   = weighted.max() if weighted.size > 0 else 0.0
    n          = weighted.size

    # 5) 유사도 계산 (0~100)
    if max_diff == 0 or n == 0:
        return 100.0  # 완전 동일 혹은 데이터 없음 → 100%
    sim = 100.0 - (total_diff / (n * max_diff)) * 100.0

    return float(np.round(sim, 2))


def cocking_sim(
    rory_seq: np.ndarray,
    hong_seq: np.ndarray
) -> tuple[float, list[float]]:
    """
    코킹 변화량(Δ 값) 시퀀스 두 개를 받아, 각 구간별 유사도(0~100)와
    전체 평균 유사도(0~100)를 반환합니다.

    * ADD 구간(첫 원소)은 제외하고, 2~마지막 구간(인덱스 1~)만 비교합니다.
    * 방향 일치 → 50점, 불일치 → 0점
    * 크기 유사도 = 50 - (|r - h| / max(|r|,|h|)) * 50
      (둘 다 0이면 50점 고정)
    * 구간별 합 = 방향점 + 크기점
    * 전체 유사도 = 구간별 합들의 평균
    """
    R = np.asarray(rory_seq, dtype=float)
    H = np.asarray(hong_seq, dtype=float)
    if R.shape != H.shape:
        raise ValueError("rory_seq, hong_seq 는 같은 길이여야 합니다.")

    # 첫 구간(ADD)은 제외하고 나머지 Δ만 비교
    deltas_r = R[1:]
    deltas_h = H[1:]
    scores = []

    for dr, dh in zip(deltas_r, deltas_h):
        # 1) 방향 일치 점수
        if np.sign(dr) == np.sign(dh):
            dir_score = 50.0
            # 2) 크기 점수
            maxc = max(abs(dr), abs(dh))
            if maxc == 0:
                size_score = 50.0
            else:
                size_diff = abs(dr - dh)
                size_score = 50.0 - (size_diff / maxc) * 50.0
        else:
            dir_score = 0.0
            size_score = 0.0

        scores.append(dir_score + size_score)

    # 3) 전체 평균 유사도
    overall = float(np.round(np.mean(scores) if scores else 0.0, 2))
    return overall, [float(np.round(s, 2)) for s in scores]


def hinging_sim(
    rory_vals: np.ndarray,
    hong_vals: np.ndarray,
    alpha: float = 2.0
) -> float:
    """
    절대 힌징값 배열 두 개로 유사도 계산.
    부호 다를 때는 차이에 alpha 배 가중.
    0~100 사이 float 반환.
    """
    R = np.asarray(rory_vals, dtype=float)
    H = np.asarray(hong_vals, dtype=float)
    if R.shape != H.shape:
        raise ValueError("rory_vals, hong_vals 는 같은 길이여야 합니다.")

    same = np.sign(R) == np.sign(H)
    diffs = np.abs(R - H)
    weighted = np.where(same, diffs, alpha * diffs)

    total, maxd, n = weighted.sum(), weighted.max() if weighted.size>0 else 0.0, weighted.size
    if n == 0 or maxd == 0:
        return 100.0
    sim = 100.0 - (total / (n * maxd)) * 100.0
    return float(np.round(sim, 2))


def bowing_sim(
    rory_rel: np.ndarray,
    hong_rel: np.ndarray
) -> float:
    """
    보잉/커핑 상대각도 배열 두 개를 받아 유사도(0~100)를 계산합니다.
    - 방향 일치 점수: 50점 만점 (부호가 같으면 1, 아니면 0)
    - 크기 유사도 점수: 50점 만점 (차이의 평균을 최댓값으로 나눈 뒤 1-비율)
    """
    R = np.asarray(rory_rel, dtype=float)
    H = np.asarray(hong_rel, dtype=float)
    if R.shape != H.shape:
        raise ValueError("rory_rel, hong_rel 는 같은 길이여야 합니다.")

    # 1) 부호(방향) 일치 점수
    sign_match = (np.sign(R) == np.sign(H)).astype(float)
    sign_score = np.mean(sign_match) * 50.0

    # 2) 크기 유사도 점수
    diffs = np.abs(R - H)
    max_val = np.max(np.abs(R)) if np.max(np.abs(R)) != 0 else 1.0
    magnitude_score = (1.0 - np.mean(diffs / max_val)) * 50.0

    # 3) 총합 및 반올림
    total = sign_score + magnitude_score
    return float(np.round(total, 2))


def club_sim(
    rory_vals: np.ndarray,
    hong_vals: np.ndarray,
    alpha: float = 2.0,
    scale: float = 90.0
) -> float:
    """
    일반화된 평균 오차 기반 유사도 계산 (0~100%).
    1) 부호 일치: error = |r-h|, 부호 다름: alpha*|r-h|
    2) MeanError = 평균(error)
    3) Similarity = 100 - (MeanError/scale)*100
    """
    R = np.asarray(rory_vals, dtype=float)
    H = np.asarray(hong_vals, dtype=float)
    if R.shape != H.shape:
        raise ValueError("rory_vals, hong_vals 는 같은 길이여야 합니다.")

    # 오차 계산
    same_sign = np.sign(R) == np.sign(H)
    diff = np.abs(R - H)
    adjusted = np.where(same_sign, diff, alpha * diff)

    mean_error = np.mean(adjusted)
    similarity = 100.0 - (mean_error / scale) * 100.0
    return float(np.round(similarity, 2))


if __name__ == "__main__":
    # 예시 시퀀스 (Tilt 1~9)
    rory_tilt = np.array([-2.61, -40.23,  88.59, -53.93, -35.74,  -8.95,   4.62,  15.12, -18.11])
    hong_tilt = np.array([-5.13, -31.15,  -4.96, -68.16, -20.49,   11.54, -20.36,  -0.48,  -7.32])
    sim = compute_mean_error_similarity(rory_tilt, hong_tilt)
    print("Mean Error Similarity:", sim)
