# sections/swing/features/swing_tempo.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Union

# ──────────────── 공통 유틸 ────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def _safe_div(a: float, b: float) -> float:
    return np.nan if b == 0 else a / b

def classify_relative(value: float, standard: float, tol: float,
                      low_is_fast: bool = True) -> str:
    """
    표준±tol 이내면 'middle'. 그 외는 작으면/크면 fast/slow.
    """
    if any(map(np.isnan, [value, standard, tol])):
        return "undefined"
    if abs(value - standard) <= tol:
        return "middle"
    return "fast" if (value < standard) == bool(low_is_fast) else "slow"

# ──────────────── 한 시트에서 5개 값 추출 ────────────────
def extract_values(arr: np.ndarray,
                   row_b4: int = 4, row_b7: int = 7, row_b10: int = 10):
    """
    1) TOTAL SWING TIME : B10
    2) 1/7              : B7
    3) 1-4/4-7          : B4 / (B7 - B4)
    4) 1/4              : B4
    5) 4/7              : B7 - B4
    """
    v1 = g(arr, f"B{row_b10}")
    v2 = g(arr, f"B{row_b7}")
    b4 = g(arr, f"B{row_b4}")
    v5 = v2 - b4
    v3 = _safe_div(b4, v5)
    v4 = b4
    return v1, v2, v3, v4, v5

# ──────────────── 본 계산 (메인 호출 방식과 100% 호환) ────────────────
def compute_tempo_rhythm(
    arr: np.ndarray,
    *,
    # 메인에서 넘겨주는 4개 인자(초기 프로 기준)
    tempo_std: float = 1.14,
    rhythm_std: float = 2.80,
    tempo_tol: float = 0.05,
    rhythm_tol: float = 0.20,
    # 4,5번 기준(입력 안 하면 합리적 디폴트)
    back_std: float | None = None,   # 4번(1/4) 기준. 기본 0.84
    down_std: float | None = None,   # 5번(4/7) 기준. 기본 tempo_std - back_std (=0.30)
    back_tol: float = 0.05,
    down_tol: float = 0.05,
) -> dict:
    """
    메인 코드가 넘기는 4개 파라미터만으로도 동작.
    4·5번 기준은 주지 않으면 back_std=0.84, down_std=(tempo_std - back_std)을 사용.
    """
    v1, v2, v3, v4, v5 = extract_values(arr)

    if back_std is None:
        back_std = 0.84
    if down_std is None:
        down_std = tempo_std - back_std  # 예: 1.14 - 0.84 = 0.30

    tempo_style  = classify_relative(v2, tempo_std,  tempo_tol,  low_is_fast=True)  # 2번
    rhythm_style = classify_relative(v3, rhythm_std, rhythm_tol, low_is_fast=True)  # 3번
    back_style   = classify_relative(v4, back_std,   back_tol,   low_is_fast=True)  # 4번
    down_style   = classify_relative(v5, down_std,   down_tol,   low_is_fast=True)  # 5번

    return {
        # 값 (표에서 사용)
        "1_value": v1, "2_value": v2, "3_value": v3, "4_value": v4, "5_value": v5,
        # 기준/허용오차 (표시용)
        "tempo_std": tempo_std,   "tempo_tol": tempo_tol,
        "rhythm_std": rhythm_std, "rhythm_tol": rhythm_tol,
        "back_std": back_std,     "back_tol": back_tol,
        "down_std": down_std,     "down_tol": down_tol,
        # 스타일
        "tempo_style":  tempo_style,
        "rhythm_style": rhythm_style,
        "back_style":   back_style,
        "down_style":   down_style,
    }

# ──────────────── 표(DataFrame) 생성 ────────────────
def build_tempo_rhythm_table(metrics: dict) -> pd.DataFrame:
    """
    컬럼: No, 검사항목, 수식/셀, 값, 표준(기준), 스타일
    """
    rows = [
        ["1", "TOTAL SWING TIME", "B10",        metrics["1_value"], np.nan,                 ""],
        ["2", "1/7",              "B7",         metrics["2_value"], metrics["tempo_std"],   metrics["tempo_style"]],
        ["3", "1-4/4-7",          "B4/(B7-B4)", metrics["3_value"], metrics["rhythm_std"],  metrics["rhythm_style"]],
        ["4", "1/4",              "B4",         metrics["4_value"], metrics["back_std"],    metrics["back_style"]],
        ["5", "4/7",              "B7-B4",      metrics["5_value"], metrics["down_std"],    metrics["down_style"]],
    ]
    return pd.DataFrame(rows, columns=["No", "검사항목", "수식/셀", "값", "표준(기준)", "스타일"])

def build_tempo_rhythm_compare(pro: dict, ama: dict) -> pd.DataFrame:
    """
    프로/일반 비교표 (5개 항목 모두, 스타일 포함)
    """
    rows = [
        [ "TOTAL SWING TIME", pro["1_value"], ama["1_value"], pro["1_value"] - ama["1_value"], ""],
        [ "1/7",              pro["2_value"], ama["2_value"], pro["2_value"] - ama["2_value"],
            f"P:{pro['tempo_style']}, A:{ama['tempo_style']}"],
        [ "1-4/4-7",          pro["3_value"], ama["3_value"], pro["3_value"] - ama["3_value"],
            f"P:{pro['rhythm_style']}, A:{ama['rhythm_style']}"],
        [ "1/4",              pro["4_value"], ama["4_value"], pro["4_value"] - ama["4_value"],
            f"P:{pro['back_style']}, A:{ama['back_style']}"],
        [ "4/7",              pro["5_value"], ama["5_value"], pro["5_value"] - ama["5_value"],
            f"P:{pro['down_style']}, A:{ama['down_style']}"],
    ]
    return pd.DataFrame(rows, columns=["검사항목", "프로", "일반", "차이(프로-일반)", "스타일"])
