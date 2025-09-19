# sections/face_angle/features/_2rolling.py  (요지 부분만)
from __future__ import annotations
import math, numpy as np, pandas as pd

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx-1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    try:
        return float(arr[num-1, _col_idx(letters)])
    except Exception:
        return float("nan")

def _wrist(arr: np.ndarray) -> list[float]:
    return [g(arr, f"AY{n}") - g(arr, f"BN{n}") for n in range(1,10)]

def _elbow(arr: np.ndarray) -> list[float]:
    return [g(arr, f"AS{n}") - g(arr, f"BH{n}") for n in range(1,10)]

def _pure_rolling(wrist: list[float], elbow: list[float]) -> list[float]:
    pr = [math.nan]
    for i in range(1, len(wrist)):
        pr.append((wrist[i]-wrist[i-1]) - (elbow[i]-elbow[i-1]))
    return pr

def _segment_sums(pr: list[float]) -> tuple[float,float,float]:
    return (np.nansum(pr[1:4]), np.nansum(pr[4:7]), np.nansum(pr[7:10]))

def _cocking_maint(s14: float, s47: float) -> float:
    # 기존 정의 유지
    return (s47 - s14) if s14*s47 >= 0 else (s47 + s14)

def rolling_sim(r_vals: np.ndarray, h_vals: np.ndarray, alpha: float = 2.0) -> float:
    R = np.asarray(r_vals, dtype=float); H = np.asarray(h_vals, dtype=float)
    same = np.sign(R) == np.sign(H)
    diffs = np.abs(R - H)
    weighted = np.where(same, diffs, alpha * diffs)
    if weighted.size == 0 or np.nanmax(weighted) == 0:
        return 100.0
    sim = 100.0 - (np.nansum(weighted) / (weighted.size * np.nanmax(weighted))) * 100.0
    return float(np.round(sim, 2))

def build_rolling_summary_table(pro_arr: np.ndarray, ama_arr: np.ndarray, alpha: float = 2.0) -> pd.DataFrame:
    labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2"]

    # 시퀀스
    w_p, e_p = _wrist(pro_arr), _elbow(pro_arr)
    w_a, e_a = _wrist(ama_arr), _elbow(ama_arr)
    pr_p, pr_a = _pure_rolling(w_p, e_p), _pure_rolling(w_a, e_a)

    rows = []
    for i, name in enumerate(labels):
        rows.append([name, w_p[i], w_a[i], pr_p[i], pr_a[i], np.nan])  # 본문 행(유사도 칸은 공란)

    # 구간 합(순수롤링만)
    s14_p, s47_p, s79_p = _segment_sums(pr_p)
    s14_a, s47_a, s79_a = _segment_sums(pr_a)
    diff17_p, diff17_a  = _cocking_maint(s14_p, s47_p), _cocking_maint(s14_a, s47_a)

    rows += [
        ["1-4", np.nan, np.nan, s14_p,    s14_a,    np.nan],
        ["4-7", np.nan, np.nan, s47_p,    s47_a,    np.nan],
        ["7-9", np.nan, np.nan, s79_p,    s79_a,    np.nan],
        ["1-7", np.nan, np.nan, diff17_p, diff17_a, np.nan],
    ]


    dw_p = np.diff(w_p)        # 길이 8
    dw_a = np.diff(w_a)
    # 표준편차(손목/순수롤링 모두)
    std_wp, std_wa = float(np.nanstd(dw_p)),  float(np.nanstd(dw_a))
    std_pp, std_pa = float(np.nanstd(pr_p)),  float(np.nanstd(pr_a))
    rows.append(["STD", std_wp, std_wa, std_pp, std_pa, np.nan])

    # Total Δ (네 개 다)
    total_wp = float(np.nansum(np.abs(dw_p)))
    total_wa = float(np.nansum(np.abs(dw_a)))
    total_pp = float(np.nansum(np.abs(np.asarray(pr_p)[1:])))  # NaN(ADD) 제외
    total_pa = float(np.nansum(np.abs(np.asarray(pr_a)[1:])))
    rows.append(["Total Δ", total_wp, total_wa, total_pp, total_pa, np.nan])

    # 유사도(순수롤링 기준)
    mask = (~np.isnan(pr_p)) & (~np.isnan(pr_a))
    sim = rolling_sim(np.asarray(pr_p)[mask], np.asarray(pr_a)[mask], alpha=alpha)
    rows.append(["Similarity(%)", np.nan, np.nan, np.nan, np.nan, sim])

    df = pd.DataFrame(rows, columns=["구간","손목(프로)","손목(일반)","순수롤링(프로)","순수롤링(일반)","유사도(%)"])
    # 숫자형으로 캐스팅(스타일링 오류 방지)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
