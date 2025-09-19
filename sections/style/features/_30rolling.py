# sections/swing/features/_24_summary_roll.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ── 엑셀 셀 헬퍼 ──────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ── 핵심 계산(배열 입력) ──────────────────────────────────────────────────
def _compute_wrist_elbow(arr: np.ndarray):
    wrist = [g(arr, f"AY{n}") - g(arr, f"BN{n}") for n in range(1, 10)]
    elbow = [g(arr, f"AS{n}") - g(arr, f"BH{n}") for n in range(1, 10)]
    return wrist, elbow

def _compute_pure_rolling(wrist: list[float], elbow: list[float]) -> list[float]:
    pr = [math.nan]  # 프레임1은 변화량 없음
    for i in range(1, len(wrist)):
        pr.append((wrist[i]-wrist[i-1]) - (elbow[i]-elbow[i-1]))
    return pr

def _segment_sums(pr: list[float]):
    # 1–4(2~4), 4–7(4~7), 7–9(7~9) — 네 로직 그대로
    return sum(pr[1:4]), sum(pr[3:7]), sum(pr[6:9])

def _cocking_maintenance(s14: float, s47: float) -> float:
    # 부호 같으면 s47 - s14, 다르면 s47 + s14
    return (s47 - s14) if s14*s47 >= 0 else (s47 + s14)

def summarize_from_arr(arr: np.ndarray) -> dict:
    wrist, elbow = _compute_wrist_elbow(arr)
    pure  = _compute_pure_rolling(wrist, elbow)
    s14, s47, s79 = _segment_sums(pure)
    diff17 = _cocking_maintenance(s14, s47)
    stdp   = float(np.nanstd(pure))
    total  = sum(abs(x) for x in pure if not math.isnan(x))
    return {
        "wrist": wrist,
        "elbow": elbow,
        "pure_roll": pure,
        "sum1_4": s14,
        "sum4_7": s47,
        "sum7_9": s79,
        "diff1_7": diff17,
        "std": stdp,
        "total_delta": total,
    }

# ── 표 생성 API ───────────────────────────────────────────────────────────
def build_summary_10_11_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    10, 11번 핵심값만 보여주는 요약 표:
      10) 1-4 순수 롤링 합 (sum1_4)
      11) 코킹 유지량(1-7) (diff1_7)
    """
    P = summarize_from_arr(pro_arr)
    A = summarize_from_arr(ama_arr)

    rows = [
        [" 1-4 ", round(P["sum1_4"], 2), round(A["sum1_4"], 2), round(P["sum1_4"] - A["sum1_4"], 2)],
        ["4-7", round(P["diff1_7"], 2), round(A["diff1_7"], 2), round(P["diff1_7"] - A["diff1_7"], 2)],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

def build_full_tables(pro_arr: np.ndarray, ama_arr: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 표(선택사항): 프레임별 wrist/elbow/pure_roll + 요약행
    반환: (프로용 DF, 일반용 DF)
    """
    def _mk(person: dict, title: str) -> pd.DataFrame:
        idx = [str(i) for i in range(1, 10)] + ["1-4 합(순수)","4-7 합(순수)","7-9 합(순수)","코킹 유지량(1-7)","표준편차","총 변화 절대합"]
        wrist = [round(x,2) if isinstance(x,(int,float)) else x for x in person["wrist"]] + [None]*6
        elbow = [round(x,2) if isinstance(x,(int,float)) else x for x in person["elbow"]] + [None]*6
        pure  = [round(x,2) if isinstance(x,(int,float)) else x for x in person["pure_roll"]] + \
                [round(person["sum1_4"],2), round(person["sum4_7"],2), round(person["sum7_9"],2),
                 round(person["diff1_7"],2), round(person["std"],2), round(person["total_delta"],2)]
        return pd.DataFrame({"Frame/요약": idx, "Wrist": wrist, "Elbow": elbow, "PureRolling": pure})

    P = summarize_from_arr(pro_arr)
    A = summarize_from_arr(ama_arr)
    return _mk(P, "프로"), _mk(A, "일반")
