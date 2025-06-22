import numpy as np

def compute_absolute_hinging_similarity(rory_values, hong_values, alpha=2):
    """
    절대 힌징값을 기준으로 유사도 계산.
    부호가 다르면 차이값에 alpha 배 가중치를 적용.

    Parameters:
        rory_values (list or np.array): 로리의 힌징값 (deg)
        hong_values (list or np.array): 홍의 힌징값 (deg)
        alpha (float): 부호 다를 때의 가중치 (default = 2)

    Returns:
        similarity (float): 유사도 (0~100 사이)
    """
    rory_values = np.array(rory_values)
    hong_values = np.array(hong_values)

    same_sign = np.sign(rory_values) == np.sign(hong_values)
    diff = np.abs(rory_values - hong_values)
    adjusted_diff = np.where(same_sign, diff, alpha * diff)

    total_diff = np.sum(adjusted_diff)
    max_diff = np.max(adjusted_diff)
    n = len(rory_values)

    similarity = 100 - (total_diff / (n * max_diff)) * 100
    return round(similarity, 2)

rory = [30.40, 4.66, -6.80, -5.28, -12.50, -22.25, 10.25, 10.45, -31.72, -13.49]
hong = [39.66, -10.50, -17.85, -6.23, -13.07, -13.78, 17.55, 0.11, -6.72, -42.75]

similarity = compute_absolute_hinging_similarity(rory, hong, alpha=2)
print(f"절대 힌징값 기반 유사도: {similarity}%")
