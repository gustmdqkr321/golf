from __future__ import annotations
import numpy as np
import pandas as pd
import re

# ── A1 → index ──────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

_A1 = re.compile(r'^([A-Za-z]+)(\d+)$')

def g(arr: np.ndarray, code: str) -> float:
    m = _A1.match(code)
    if not m:
        return float("nan")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    try:
        return float(arr[row, col])
    except Exception:
        return float("nan")

# ── 공용: 좌/우 3D 포인트 평균 → 중심좌표 시퀀스(프레임 1..10) ─────────────
_FRAMES = list(range(1, 11))          # 1..10
_SEG_DEF = {"1-4": (0,3), "4-7": (3,6), "7-10": (6,9)}  # 인덱스 영역(0-based)

def _center_series(arr: np.ndarray,
                   Lx: str, Ly: str, Lz: str,
                   Rx: str, Ry: str, Rz: str) -> np.ndarray:
    """프레임별 중심점 (L,R 평균) (10×3)"""
    out = []
    for n in _FRAMES:
        L = np.array([g(arr, f"{Lx}{n}"), g(arr, f"{Ly}{n}"), g(arr, f"{Lz}{n}")], dtype=float)
        R = np.array([g(arr, f"{Rx}{n}"), g(arr, f"{Ry}{n}"), g(arr, f"{Rz}{n}")], dtype=float)
        out.append((L + R) / 2.0)
    return np.vstack(out)  # (10,3)

def _build_center_table_generic(base_pro: np.ndarray, base_ama: np.ndarray,
                                Lx: str, Ly: str, Lz: str,
                                Rx: str, Ry: str, Rz: str) -> pd.DataFrame:
    """
    • 구간별(1-4,4-7,7-10) 평균 중심 - ADD(프레임1) 중심
    • Pro/Ama 값 표시(+.3f), Ama는 Pro와 부호 반대면 '❗'
    • 합계행: Pro/Ama 각 축의 |세 구간 값| 합(소수2), 차이 컬럼은 |세 구간의 (Ama-Pro)| 합(소수2)
    """
    Cp = _center_series(base_pro, Lx, Ly, Lz, Rx, Ry, Rz)  # (10,3)
    Ca = _center_series(base_ama, Lx, Ly, Lz, Rx, Ry, Rz)

    base_p = Cp[0]  # ADD
    base_a = Ca[0]

    rows = []
    pro_vals = {"X": [], "Y": [], "Z": []}
    ama_vals = {"X": [], "Y": [], "Z": []}
    diffs    = {"X": [], "Y": [], "Z": []}

    for label, (s, e) in _SEG_DEF.items():
        avg_p = Cp[s:e].mean(axis=0)
        avg_a = Ca[s:e].mean(axis=0)
        dp = np.round(avg_p - base_p, 3)
        da = np.round(avg_a - base_a, 3)

        for ax, i in zip(("X","Y","Z"), (0,1,2)):
            pro_vals[ax].append(float(dp[i]))
            ama_vals[ax].append(float(da[i]))
            diffs[ax].append(float(da[i] - dp[i]))

        row = {"구간": label}
        for ax, i in zip(("X","Y","Z"), (0,1,2)):
            p = dp[i]; a = da[i]
            row[f"Pro {ax}"] = f"{p:+.3f}"
            row[f"Ama {ax}"] = f"{a:+.3f}" 
        rows.append(row)

    # 세 구간으로만 우선 DataFrame 구성 (3행)
    df = pd.DataFrame(rows, columns=[
        "구간",
        "Pro X","Pro Y","Pro Z",
        "Ama X","Ama Y","Ama Z",
    ])

    # (A) 차이 컬럼은 3개 구간 값만 먼저 채운다 (길이 = 3)
    diff_totals: dict[str, float] = {}
    for ax in ("X","Y","Z"):
        seg_diffs = [round(v, 2) for v in diffs[ax]]  # 길이 3
        df[f"{ax} 차이 (Ama - Pro)"] = seg_diffs
        diff_totals[ax] = round(sum(abs(v) for v in seg_diffs), 2)

    # 합계행 계산(숫자)
    sum_row = {"구간": "합계"}
    for ax in ("X","Y","Z"):
        sum_row[f"Pro {ax}"] = round(sum(abs(v) for v in pro_vals[ax]), 2)
        sum_row[f"Ama {ax}"] = round(sum(abs(v) for v in ama_vals[ax]), 2)

    # (B) 합계행을 붙인 뒤, 차이 컬럼의 Total은 마지막 행에만 값을 넣는다
    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
    last = df.index[-1]
    for ax in ("X","Y","Z"):
        df.at[last, f"{ax} 차이 (Ama - Pro)"] = diff_totals[ax]

    return df


# ── 공개 API: 골반/어깨/무릎 ─────────────────────────────────────────────
def build_pelvis_center_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """골반: (H,I,J) & (K,L,M)"""
    return _build_center_table_generic(base_pro, base_ama, "H","I","J", "K","L","M")

def build_shoulder_center_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """어깨: (AL,AM,AN) & (BA,BB,BC)"""
    return _build_center_table_generic(base_pro, base_ama, "AL","AM","AN", "BA","BB","BC")

def build_knee_center_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """무릎: (BP,BQ,BR) & (CB,CC,CD)"""
    return _build_center_table_generic(base_pro, base_ama, "BP","BQ","BR", "CB","CC","CD")
