# sections/club_hand/features/_6accel.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Sequence

# ====== 기본 설정(디폴트) ======
DEFAULT_TIME_COL = "B"
DEFAULT_LEFT_WRIST  = ("AX", "AY", "AZ")   # 왼손목 좌표
DEFAULT_RIGHT_WRIST = ("BM", "BN", "BO")   # 오른손목 좌표
DEFAULT_CLUB_HEAD   = ("CN", "CO", "CP")   # 클럽헤드 좌표
DEFAULT_ROUND = 2


# ====== 내부 유틸 ======
def _to_seconds(t: np.ndarray) -> np.ndarray:
    """시간열(ms 또는 s)을 자동으로 초 단위로 변환"""
    t = np.asarray(t, dtype=float)
    # 프레임 간 간격의 중앙값이 10초(?) 보다 크면 ms 단위라고 보기 어렵다.
    # 원래 예시 로직(>10)을 유지: 10보다 크면 ms로 간주하여 /1000
    if t.size >= 2 and np.median(np.diff(t)) > 10:
        return t / 1000.0
    return t

def _segment_center(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """좌우 좌표 평균으로 중심점 계산"""
    return (L + R) / 2.0

def _calc_segment_accel(coords: np.ndarray, t_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    좌표 기반 프레임별 가속도 계산
    - coords: (N, 3) [cm 단위 가정]
    - t_sec : (N,)  [sec]
    반환:
      a: (N-2,)  프레임 경계 가속도 [m/s^2]
      v: (N-1,)  세그먼트 속도 [m/s]
      t_mid: (N-1,) 세그먼트 중앙 시각 [sec]
    """
    # 세그먼트 변위(m) — 입력 좌표(cm) 가정 → /100
    disp = np.linalg.norm(np.diff(coords, axis=0), axis=1) / 100.0
    # 세그 중간 시각
    t_mid = (t_sec[:-1] + t_sec[1:]) / 2.0
    dt = np.diff(t_sec)

    # 속도 (세그먼트 기준)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt != 0, disp / dt, np.nan)

    # 가속도 (연속 세그먼트 속도 변화율)
    dv = np.diff(v)
    dt_a = np.diff(t_mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(dt_a != 0, dv / dt_a, np.nan)

    return a, v, t_mid

def _as_numeric_block(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """지정한 열들을 float 블록으로 변환 (결측/문자 → NaN 처리)"""
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"필수 열이 없습니다: '{c}'")
    block = df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    return block


# ====== 공개 API ======
def build_hand_club_accel_table(
    df_pro: pd.DataFrame,
    df_ama: pd.DataFrame,
    *,
    time_col: str = DEFAULT_TIME_COL,
    left_wrist: Sequence[str] = DEFAULT_LEFT_WRIST,
    right_wrist: Sequence[str] = DEFAULT_RIGHT_WRIST,
    club_head: Sequence[str] = DEFAULT_CLUB_HEAD,
    ndigits: int = DEFAULT_ROUND,
    pro_label: str = "로리",
    ama_label: str = "홍",
) -> pd.DataFrame:
    """
    프로/일반 DataFrame에서 손 중심(양손목 평균)과 클럽헤드의 프레임별 가속도를 계산해 표로 반환.

    반환 컬럼:
      ['구간',
       f'손 가속도(m/s²) - {pro_label}',
       f'손 가속도(m/s²) - {ama_label}',
       f'클럽 가속도(m/s²) - {pro_label}',
       f'클럽 가속도(m/s²) - {ama_label}']

    구간 라벨은 (N-2) 길이: 1-2 ~ (N-2)-(N-1)
    """

    # ----- Pro -----
    if time_col not in df_pro.columns:
        raise KeyError(f"프로 DF에 시간열('{time_col}')이 없습니다.")
    t_pro = _to_seconds(pd.to_numeric(df_pro[time_col], errors="coerce").values)

    Lp = _as_numeric_block(df_pro, left_wrist)
    Rp = _as_numeric_block(df_pro, right_wrist)
    Cp = _as_numeric_block(df_pro, club_head)

    center_p = _segment_center(Lp, Rp)
    a_hand_p, _, _ = _calc_segment_accel(center_p, t_pro)
    a_club_p, _, _ = _calc_segment_accel(Cp, t_pro)

    # ----- Ama -----
    if time_col not in df_ama.columns:
        raise KeyError(f"일반 DF에 시간열('{time_col}')이 없습니다.")
    t_ama = _to_seconds(pd.to_numeric(df_ama[time_col], errors="coerce").values)

    La = _as_numeric_block(df_ama, left_wrist)
    Ra = _as_numeric_block(df_ama, right_wrist)
    Ca = _as_numeric_block(df_ama, club_head)

    center_a = _segment_center(La, Ra)
    a_hand_a, _, _ = _calc_segment_accel(center_a, t_ama)
    a_club_a, _, _ = _calc_segment_accel(Ca, t_ama)

    # 길이 정합(혹시 서로 프레임 수가 다르면 공통 최소 길이로 맞춤)
    k = min(len(a_hand_p), len(a_hand_a), len(a_club_p), len(a_club_a))
    a_hand_p = np.round(a_hand_p[:k], ndigits)
    a_hand_a = np.round(a_hand_a[:k], ndigits)
    a_club_p = np.round(a_club_p[:k], ndigits)
    a_club_a = np.round(a_club_a[:k], ndigits)

    # 구간 라벨: 1-2 … (k)-(k+1)
    labels = [f"{i+1}-{i+2}" for i in range(k)]

    tbl = pd.DataFrame({
        "구간": labels,
        f"손 가속도(m/s²) - {pro_label}": a_hand_p,
        f"손 가속도(m/s²) - {ama_label}": a_hand_a,
        f"클럽 가속도(m/s²) - {pro_label}": a_club_p,
        f"클럽 가속도(m/s²) - {ama_label}": a_club_a,
    })

    return tbl
