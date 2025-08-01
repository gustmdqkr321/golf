# summary.py

import math
import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    return pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

def compute_wrist_elbow(arr: np.ndarray):
    wrist = [g(arr, f"AY{n}") - g(arr, f"BN{n}") for n in range(1,10)]
    elbow = [g(arr, f"AS{n}") - g(arr, f"BH{n}") for n in range(1,10)]
    return wrist, elbow

def compute_pure_rolling(wrist: list[float], elbow: list[float]) -> list[float]:
    pr = [math.nan]
    for i in range(1, len(wrist)):
        pr.append((wrist[i]-wrist[i-1]) - (elbow[i]-elbow[i-1]))
    return pr

def segment_sums(pr: list[float]):
    return sum(pr[1:4]), sum(pr[3:7]), sum(pr[6:9])

def cocking_maintenance(s14: float, s47: float) -> float:
    return (s47 - s14) if s14*s47 >= 0 else (s47 + s14)

def summarize_player(xlsx_path: Path) -> dict[str, list[float] | float]:
    """
    주어진 파일에 대해 아래 키를 가진 dict 반환:
      wrist       : list[9]   (frames 1..9)
      pure_roll   : list[9]
      sum1_4, sum4_7, sum7_9 : float
      diff1_7, std, total_delta : float
    """
    arr = load_sheet(xlsx_path)
    wrist, elbow = compute_wrist_elbow(arr)
    pure  = compute_pure_rolling(wrist, elbow)
    s14, s47, s79 = segment_sums(pure)
    diff17 = cocking_maintenance(s14, s47)
    stdp   = float(np.nanstd(pure))
    total  = sum(abs(x) for x in pure if not math.isnan(x))
    return {
        "wrist": wrist,
        "pure_roll": pure,
        "sum1_4": s14,
        "sum4_7": s47,
        "sum7_9": s79,
        "diff1_7": diff17,
        "std": stdp,
        "total_delta": total,
    }
