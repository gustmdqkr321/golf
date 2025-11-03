# sections/club_path/features/_2shoulder_wrist_elbow_x.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── 기본 유틸 ────────────────────────────────────────────────────────────
_CELL = re.compile(r'([A-Za-z]+)(\d+)')

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    m = _CELL.fullmatch(code.strip())
    if not m:
        return float("nan")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    try:
        return float(arr[row, col])
    except Exception:
        return float("nan")

def _eval_expr(arr: np.ndarray, expr: str) -> float:
    """'AX1-AY1' 같은 간단식 평가 (안전)"""
    def repl(m: re.Match) -> str:
        return str(g(arr, m.group(0)))
    safe = _CELL.sub(repl, expr.replace(" ", ""))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

# ── 표 1: R Wrist–Shoulder(X) 특수 패턴 ──────────────────────────────────
#   프레임별 식
#   1: BO1-BC1, 2~6: BMn-BAn, 7: BO7-BC7, 8~9: BMn-BAn
def build_r_wrist_shoulder_x_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    mapping = {
        1: "BO1 - BC1",
        2: "BM2 - BA2",
        3: "BM3 - BA3",
        4: "BM4 - BA4",
        5: "BM5 - BA5",
        6: "BM6 - BA6",
        7: "BO7 - BC7",
        8: "BM8 - BA8",
        9: "BM9 - BA9",
    }
    rows: list[list] = []
    for fr, expr in mapping.items():
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append([str(fr)+"Frame",p, a, p - a])
    return pd.DataFrame(rows, columns=["Frame", "프로", "일반", "차이(프로-일반)"])

# ── 표 2: Shoulder / Elbow (X) ──────────────────────────────────────────
#   L: ARn-ALn,  R: BGn-BAn  (n=1..9)
def build_shoulder_elbow_x_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    rows: list[list] = []

    # L 블록
    for n in range(1, 10):
        expr = f"AR{n} - AL{n}"
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append(["L", n, p, a, p - a])

    # R 블록
    for n in range(1, 10):
        expr = f"BG{n} - BA{n}"
        p = _eval_expr(pro_arr, expr)
        a = _eval_expr(ama_arr, expr)
        rows.append(["R", n, p, a, p - a])

    return pd.DataFrame(rows, columns=["측", "Frame", "프로", "일반", "차이(프로-일반)"])



def build_shoulder_elbow_x_table_wide(
    pro_arr: np.ndarray, ama_arr: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    가로형 표 (1~9가 컬럼)
    - L: ARn-ALn
    - R: BGn-BAn
    반환: (df_L, df_R)
      * 첫 번째 컬럼 '항목' = ["프로", "일반", "차이(프로-일반)"]
      * 나머지 컬럼: 1..9 (정수, 값은 소수 둘째자리)
    """
    cols = list(range(1, 10))

    # L: ARn-ALn
    pro_L = [_eval_expr(pro_arr, f"AR{n}-AL{n}") for n in cols]
    ama_L = [_eval_expr(ama_arr, f"AR{n}-AL{n}") for n in cols]
    dif_L = [p - a for p, a in zip(pro_L, ama_L)]
    df_L = pd.DataFrame(
        [pro_L, ama_L, dif_L],
        index=["프로", "일반", "차이(프로-일반)"],
        columns=cols,
    )

    # R: BGn-BAn
    pro_R = [_eval_expr(pro_arr, f"BG{n}-BA{n}") for n in cols]
    ama_R = [_eval_expr(ama_arr, f"BG{n}-BA{n}") for n in cols]
    dif_R = [p - a for p, a in zip(pro_R, ama_R)]
    df_R = pd.DataFrame(
        [pro_R, ama_R, dif_R],
        index=["프로", "일반", "차이(프로-일반)"],
        columns=cols,
    )

    # ✅ 인덱스를 컬럼으로 승격 + 숫자형/반올림 유지
    def _to_visible(df: pd.DataFrame) -> pd.DataFrame:
        out = df.reset_index().rename(columns={"index": "항목"})
        for c in out.columns:
            if c == "항목":
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        return out

    return _to_visible(df_L), _to_visible(df_R)
