from __future__ import annotations
import numpy as np, pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters: idx = idx*26 + (ord(ch.upper())-ord('A')+1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])


def _com_x_series(arr: np.ndarray) -> list[float]:
    """
    X_COM = 0.08*AC + 0.35*(AL+BA)/2 + 0.30*(H+K)/2 + 0.15*(BP+CB)/2 + 0.12*(BY+CK)/2
    (각 프레임 1..10, 소수 2자리)
    """
    out = []
    for n in range(1, 11):
        val = (
            0.08 * g(arr, f"AC{n}") +
            0.35 * (g(arr, f"AL{n}") + g(arr, f"BA{n}"))/2 +
            0.30 * (g(arr, f"H{n}")  + g(arr, f"K{n}"))/2 +
            0.15 * (g(arr, f"BP{n}") + g(arr, f"CB{n}"))/2 +
            0.12 * (g(arr, f"BY{n}") + g(arr, f"CK{n}"))/2
        )
        out.append(round(val, 2))
    return out


def _com_y_series(arr: np.ndarray) -> list[float]:
    """프레임 1..10의 Y_COM 시리즈(소수 2자리)"""
    out = []
    for n in range(1, 11):
        val = (
            0.09 * g(arr, f"AD{n}") +
            0.40 * (g(arr, f"AM{n}") + g(arr, f"BB{n}"))/2 +
            0.34 * (g(arr, f"I{n}")  + g(arr, f"L{n}"))/2 +
            0.17 * (g(arr, f"BQ{n}") + g(arr, f"CC{n}"))/2
        )
        out.append(round(val, 2))
    return out

def _com_z_series(arr: np.ndarray) -> list[float]:
    """프레임 1..10의 Z laterality 시리즈(소수 2자리)"""
    out = []
    for n in range(1, 11):
        knee  = (g(arr, f"BR{n}") + g(arr, f"CD{n}"))/2
        hip   = (g(arr, f"J{n}")  + g(arr, f"M{n}"))/2
        chest = (g(arr, f"AN{n}") + g(arr, f"BC{n}"))/2
        head  = g(arr, f"AE{n}")
        foot  = (g(arr, f"CA{n}") + g(arr, f"CM{n}"))/2
        val = ((knee + hip + chest + head)/4) - foot
        out.append(round(val, 2))
    return out

def _sum_abs_step_changes(vals: list[float]) -> float:
    """연속 프레임 간 절대 변화량의 합: |Δ1-2|+|Δ2-3|+..."""
    a = np.asarray(vals, dtype=float)
    m = np.isfinite(a[:-1]) & np.isfinite(a[1:])
    return float(np.abs(a[1:][m] - a[:-1][m]).sum().round(2))

def build_abs_1_10_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    """
    반환: 항목/프로/일반 3열
      - 1-10|abs Y : Σ|ΔY| (1→2..9→10)
      - 1-10|abs Z : Σ|ΔZ| (1→2..9→10)
    """
    x_pro = _com_x_series(pro_arr);  x_ama = _com_x_series(ama_arr)
    y_pro = _com_y_series(pro_arr);  y_ama = _com_y_series(ama_arr)
    z_pro = _com_z_series(pro_arr);  z_ama = _com_z_series(ama_arr)

    
    x_abs_pro  = round(_sum_abs_step_changes(x_pro), 2)
    x_abs_ama  = round(_sum_abs_step_changes(x_ama), 2)
    y_abs_pro  = round(_sum_abs_step_changes(y_pro), 2)
    y_abs_ama  = round(_sum_abs_step_changes(y_ama), 2)
    z_abs_pro  = round(_sum_abs_step_changes(z_pro), 2)
    z_abs_ama  = round(_sum_abs_step_changes(z_ama), 2)


    rows = [
        ["1-10|abs X", x_abs_pro, x_abs_ama],
        ["1-10|abs Y", y_abs_pro, y_abs_ama],
        ["1-10|abs Z", z_abs_pro, z_abs_ama],
    ]
    return pd.DataFrame(rows, columns=["항목", "프로", "일반"])