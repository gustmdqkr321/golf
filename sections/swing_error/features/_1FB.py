# sections/swing_error/features/frontal_bend.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _mid(a: float, b: float) -> float:
    return 0.5 * (a + b)

# ── Step 1. 중심점: 골반 P, 어깨 S (프레임 row) ─────────────────────────────
def center_P(arr: np.ndarray, row: int) -> tuple[float, float, float]:
    # Left Waist:  H,I,J | Right Waist: K,L,M
    Xp = _mid(g(arr, f"H{row}"),  g(arr, f"K{row}"))
    Yp = _mid(g(arr, f"I{row}"),  g(arr, f"L{row}"))
    Zp = _mid(g(arr, f"J{row}"),  g(arr, f"M{row}"))
    return Xp, Yp, Zp

def center_S(arr: np.ndarray, row: int) -> tuple[float, float, float]:
    # Left Shoulder: AL,AM,AN | Right Shoulder: BA,BB,BC
    Xs = _mid(g(arr, f"AL{row}"), g(arr, f"BA{row}"))
    Ys = _mid(g(arr, f"AM{row}"), g(arr, f"BB{row}"))
    Zs = _mid(g(arr, f"AN{row}"), g(arr, f"BC{row}"))
    return Xs, Ys, Zs

# ── Step 2. Frontal Bend(θ) : 수직을 0°로 정의 ──────────────────────────────
def frontal_bend_angle(arr: np.ndarray, row: int) -> tuple[float, float, float]:
    """
    반환: (θ[deg], dX, dY)
      dX = Xs - Xp, dY = Ys - Yp
      θ  = 90° - atan2(dY, dX) * 180/π
         (앞으로 숙여질수록 +, 수직=0°)
    """
    Xp, Yp, _ = center_P(arr, row)
    Xs, Ys, _ = center_S(arr, row)
    dX = Xs - Xp
    dY = Ys - Yp
    theta = 90.0 - math.degrees(math.atan2(dY, dX))
    return float(theta), float(dX), float(dY)

# ── 단일 배열용 표 생성 ──────────────────────────────────────────────────────
def build_frontal_bend_table(arr: np.ndarray, start: int = 1, end: int = 10,
                             address_frame: int = 1) -> pd.DataFrame:
    """
    각 프레임의 절대 각(θ), 어드레스 대비 변화(θ_i-θ_0), 구간 변화(θ_i-θ_{i-1})를 계산.
    """
    rows = []
    theta0, _, _ = frontal_bend_angle(arr, address_frame)
    prev_theta = None
    for r in range(start, end + 1):
        theta, dx, dy = frontal_bend_angle(arr, r)
        delta_addr = theta - theta0
        delta_seg  = (theta - prev_theta) if prev_theta is not None else float("nan")
        rows.append([r, dx, dy, theta, delta_addr, delta_seg])
        prev_theta = theta
    return pd.DataFrame(rows, columns=[
        "Frame", "Xs-Xp", "Ys-Yp", "FrontalBend(°)", "Δ(θ-θ_addr)", "Δseg(θ_i-θ_{i-1})"
    ])

# ── 프로/일반 비교 표 ────────────────────────────────────────────────────────
def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray,
                        start: int = 1, end: int = 10, address_frame: int = 1) -> pd.DataFrame:
    p = build_frontal_bend_table(pro_arr, start, end, address_frame)[["Frame", "FrontalBend(°)", "Δ(θ-θ_addr)"]]
    a = build_frontal_bend_table(ama_arr, start, end, address_frame)[["Frame", "FrontalBend(°)", "Δ(θ-θ_addr)"]]
    p.columns = ["Frame", "프로 θ", "프로 Δ"]
    a.columns = ["Frame", "일반 θ", "일반 Δ"]
    df = p.merge(a, on="Frame", how="outer")
    for c in ["프로 θ", "일반 θ", "프로 Δ", "일반 Δ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["θ 차(프로-일반)"] = (df["프로 θ"] - df["일반 θ"])
    df["Δ 차(프로-일반)"] = (df["프로 Δ"] - df["일반 Δ"])
    return df

def build_fb_report_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray,
                                  start: int = 1, end: int = 10) -> pd.DataFrame:
    """
    프레임 1~10의 Frontal Bend(θ)와 Δseg(= θ_i - θ_{i-1})를
    프로/일반/차이로 한 표에 결합하고, 아래에 구간 합계도 함께 제공.

    컬럼:
      Frame | 프로 θ | 일반 θ | θ 차(프로-일반) | 프로 Δseg | 일반 Δseg | Δseg 차(프로-일반)
    섹션 행(Backswing/Downswing/Total)은 θ 열은 NaN, Δseg 합계만 채움.
    """
    # 1) 단일 배열용 θ, Δseg 계산 헬퍼 (frame1의 Δseg=0.0)
    def _thetas_deltas(arr: np.ndarray, start: int, end: int) -> tuple[list[float], list[float]]:
        thetas: list[float] = []
        for r in range(start, end + 1):
            th, _, _ = frontal_bend_angle(arr, r)
            thetas.append(float(th))
        deltas: list[float] = [0.0]
        for i in range(1, len(thetas)):
            deltas.append(thetas[i] - thetas[i - 1])
        return thetas, deltas

    p_theta, p_delta = _thetas_deltas(pro_arr, start, end)
    a_theta, a_delta = _thetas_deltas(ama_arr, start, end)

    # 2) 프레임 행 구성
    rows: list[list[object]] = []
    frames = list(range(start, end + 1))
    for i, fr in enumerate(frames):
        rows.append([
            fr,
            p_theta[i], a_theta[i], (p_theta[i] - a_theta[i]),
            p_delta[i], a_delta[i], (p_delta[i] - a_delta[i]),
        ])

    # 3) 섹션 합계(Δseg 합만 의미 있음)
    def sum_delta(deltas: list[float], a: int, b: int) -> float:
        # Δseg는 (a+1 .. b)의 합
        i0 = (a - start) + 1
        i1 = (b - start) + 1
        return float(sum(deltas[i0:i1]))

    backswing_p = sum_delta(p_delta, 1, 4)
    backswing_a = sum_delta(a_delta, 1, 4)
    downswing_p = sum_delta(p_delta, 4, 7)
    downswing_a = sum_delta(a_delta, 4, 7)
    total17_p   = sum_delta(p_delta, 1, 7)
    total17_a   = sum_delta(a_delta, 1, 7)

    rows.extend([
        ["Backswing (1-4)", np.nan, np.nan, np.nan,
         backswing_p, backswing_a, backswing_p - backswing_a],
        ["Downswing (4-7)", np.nan, np.nan, np.nan,
         downswing_p, downswing_a, downswing_p - downswing_a],
        ["Total (1-7)",     np.nan, np.nan, np.nan,
         total17_p, total17_a, total17_p - total17_a],
    ])

    df = pd.DataFrame(rows, columns=[
        "Frame",
        "프로 θ", "일반 θ", "θ 차(프로-일반)",
        "프로 Δseg", "일반 Δseg", "Δseg 차(프로-일반)",
    ])

    # 숫자 컬럼 강제 숫자화(포매팅 에러 방지)
    for c in ["프로 θ", "일반 θ", "θ 차(프로-일반)", "프로 Δseg", "일반 Δseg", "Δseg 차(프로-일반)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
