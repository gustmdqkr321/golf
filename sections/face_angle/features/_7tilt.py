# tilt_angle.py (업데이트)
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

# ── A1 유틸 ───────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    r = num - 1
    c = col_letters_to_index(letters)
    return float(arr[r, c])

# ── Tilt 계산 ─────────────────────────────────────────────────────────
def compute_tilt_angles_from_array(arr: np.ndarray) -> list[float]:
    """
    프레임 1~9: CP, CS, CN, CQ로 틸트 계산
    - 1,7:  sign by CP vs CS
    - 2~6:  sign by (CQ-CN)
    - 8~9:  2~6의 부호 반전
    """
    angles: list[float] = []
    for n in range(1, 10):
        CP = g(arr, f"CP{n}")
        CS = g(arr, f"CS{n}")
        CN = g(arr, f"CN{n}")
        CQ = g(arr, f"CQ{n}")

        if n in (1, 7):
            num = CP - CS
            den = CN - CQ
            th = math.degrees(math.atan2(abs(num), abs(den)))
            th = -th if CP < CS else th
        elif 2 <= n <= 6:
            num = CQ - CN
            den = CP - CS
            th = math.degrees(math.atan2(abs(num), abs(den)))
            th = -th if num < 0 else th
        else:
            num = CQ - CN
            den = CP - CS
            th = math.degrees(math.atan2(abs(num), abs(den)))
            th = th if num < 0 else -th

        angles.append(round(th, 2))
    return angles

def compute_tilt_angles(xlsx_path: Path) -> list[float]:
    arr = load_sheet(xlsx_path)
    return compute_tilt_angles_from_array(arr)

# ── 유사도(일반화 평균 오차 기반) ──────────────────────────────────────
def _tilt_similarity(pro_seq, ama_seq, alpha: float = 2.0, scale: float = 90.0) -> float:
    """
    부호 다르면 오차에 alpha 배 가중 → Mean Error → 0~100로 환산
    """
    R = np.asarray(pro_seq, dtype=float)
    H = np.asarray(ama_seq, dtype=float)
    if R.shape != H.shape:
        raise ValueError("pro_seq, ama_seq 길이가 같아야 합니다.")

    same  = np.sign(R) == np.sign(H)
    diff  = np.abs(R - H)
    adj   = np.where(same, diff, alpha * diff)

    mean_err = float(np.nanmean(adj)) if adj.size else 0.0
    if not np.isfinite(mean_err) or scale == 0:
        return 100.0
    sim = 100.0 - (mean_err / scale) * 100.0
    return float(np.round(np.clip(sim, 0, 100), 2))

# ── Pro/Ama 비교표 + 유사도 ───────────────────────────────────────────
def build_tilt_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    index: Frame 1..9 + '6/7' + 'Similarity'
    columns: ["frame", "Pro Tilt (°)", "Ama Tilt (°)", "Similarity (%)"]
    - '6/7' 행에는 7번값-6번값을 넣는다.
    - 유사도는 마지막 행에만 넣는다.
    """
    pro  = compute_tilt_angles_from_array(pro_arr)   # 길이 9, 인덱스 0..8 => 1..9 frame
    ama  = compute_tilt_angles_from_array(ama_arr)
    sim  = _tilt_similarity(pro, ama, alpha=2.0, scale=90.0)

    frames = [f"{i} Frame" for i in range(1, 10)]
    df = pd.DataFrame(
        {
            "frame": frames,
            "Pro Tilt (°)"   : pro,
            "Ama Tilt (°)"   : ama,
            "Similarity (%)" : [np.nan]*9,
        }
    )

    # ── 6/7 행 추가 (7번값 - 6번값) ─────────────────────────
    # pro[5] → 6번 프레임, pro[6] → 7번 프레임
    pro_6_7 = float(np.round(pro[6] - pro[5], 2))
    ama_6_7 = float(np.round(ama[6] - ama[5], 2))
    df.loc["6/7"] = ["6/7", pro_6_7, ama_6_7, np.nan]

    # ── 마지막 Similarity 행 ───────────────────────────────
    df.loc["Similarity"] = ["Similarity", np.nan, np.nan, sim]

    return df
