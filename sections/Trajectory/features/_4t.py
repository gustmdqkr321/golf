# sections/club_path/features/_gs_club.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# 고정 바이어스(파일 형식이 항상 일정할 때 여기 숫자만 바꾸면 됨)
# ──────────────────────────────────────────────────────────────────────
GS_ROW_OFFSET = -3
GS_COL_OFFSET = 0

def set_gs_offset(row_offset: int = 0, col_offset: int = 0) -> None:
    """런타임에서 GS 오프셋을 바꾸고 싶을 때 호출"""
    global GS_ROW_OFFSET, GS_COL_OFFSET
    GS_ROW_OFFSET = int(row_offset)
    GS_COL_OFFSET = int(col_offset)

# ──────────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────────
def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

_CELL = re.compile(r'^([A-Za-z]+)(\d+)$')

def _addr_to_rc(addr: str) -> tuple[int, int]:
    m = _CELL.match(addr.strip())
    if not m:
        raise ValueError(f"잘못된 셀 주소: {addr}")
    col = _col_idx(m.group(1))
    row = int(m.group(2)) - 1
    return row, col

def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    s = str(x).replace(",", "").replace('"', "").replace("'", "").strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

# ──────────────────────────────────────────────────────────────────────
# GS CSV 셀 읽기(고정 바이어스만 적용)
# ──────────────────────────────────────────────────────────────────────
def g_gs(gs_df: pd.DataFrame, addr: str) -> float:
    r, c = _addr_to_rc(addr)        # A1 → (0,0) 기준 좌표
    rr = max(0, r + GS_ROW_OFFSET)  # 음수 방지
    cc = max(0, c + GS_COL_OFFSET)
    try:
        return _to_float(gs_df.iat[rr, cc])
    except Exception:
        return float("nan")

# ──────────────────────────────────────────────────────────────────────
# 무지개(기존 배열) 식 평가
# ──────────────────────────────────────────────────────────────────────
def g_base(arr: np.ndarray, addr: str) -> float:
    r, c = _addr_to_rc(addr)
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

def eval_expr_base(arr: np.ndarray, expr: str) -> float:
    def repl(m):
        return str(g_base(arr, m.group(0)))
    safe = re.sub(r'[A-Za-z]+\d+', repl, expr.replace(" ", ""))
    if not re.fullmatch(r'[-+*/().0-9]+', safe):
        raise ValueError(f"허용되지 않는 식: {expr}")
    return float(eval(safe, {"__builtins__": None}, {}))

# 프레임 매핑
FRAMES = [("ADD", 1), ("IMP", 7), ("FH1", 8)]

def _coords(arr: np.ndarray, n: int):
    """프레임 n에서 필요한 좌표/스칼라를 한 번에 읽기"""
    # 어깨
    AL = g_base(arr, f"AL{n}"); AM = g_base(arr, f"AM{n}"); AN = g_base(arr, f"AN{n}")
    BA = g_base(arr, f"BA{n}"); BB = g_base(arr, f"BB{n}"); BC = g_base(arr, f"BC{n}")
    # 골반
    H  = g_base(arr, f"H{n}");  I  = g_base(arr, f"I{n}");  J  = g_base(arr, f"J{n}")
    K  = g_base(arr, f"K{n}");  L  = g_base(arr, f"L{n}");  M  = g_base(arr, f"M{n}")
    # 손목
    AX = g_base(arr, f"AX{n}"); AY = g_base(arr, f"AY{n}"); AZ = g_base(arr, f"AZ{n}")
    BM = g_base(arr, f"BM{n}"); BN = g_base(arr, f"BN{n}"); BO = g_base(arr, f"BO{n}")

    left_shoulder  = np.array([AL, AM, AN], dtype=float)
    right_shoulder = np.array([BA, BB, BC], dtype=float)
    left_hip       = np.array([H, I, J],   dtype=float)
    right_hip      = np.array([K, L, M],   dtype=float)
    wrist_xy       = np.array([AX, AY],    dtype=float)
    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "wrist_xy": wrist_xy,
        "wrist_z_pair": (AZ, BO),
        "AM": AM, "BB": BB, "AN": AN, "BC": BC
    }

def _calc_metrics(arr: np.ndarray, frame_name: str, n: int, player: str) -> dict:
    c = _coords(arr, n)
    pelvis_center   = (c["left_hip"] + c["right_hip"]) / 2.0     # (x,y,z)
    shoulder_center = (c["left_shoulder"] + c["right_shoulder"]) / 2.0
    wrist_z         = (c["wrist_z_pair"][0] + c["wrist_z_pair"][1]) / 2.0

    lateral_tilt_y  = c["AM"] - c["BB"]                           # (L shoulder Y - R shoulder Y)
    pelvis_z_tilt   = float(pelvis_center[2])                     # (J+M)/2
    shoulder_z_tilt = float(shoulder_center[2])                   # (AN+BC)/2
    shoulder_z_rel  = shoulder_z_tilt - pelvis_z_tilt
    arm_body_dist   = float(np.linalg.norm(c["left_shoulder"][:2] - c["wrist_xy"]))

    return {
        "Frame": frame_name,
        "Player": player,
        "Wrist Z Position": float(wrist_z),
        "Lateral Tilt (Y)": float(lateral_tilt_y),
        "Pelvis Z Tilt": float(pelvis_z_tilt),
        "Shoulder Z Tilt": float(shoulder_z_tilt),
        "Shoulder Z Tilt (Pelvis-based)": float(shoulder_z_rel),
        "Arm-Body Distance (XY)": float(arm_body_dist),
    }

def _diff_row(df: pd.DataFrame, player: str) -> pd.Series:
    imp = df[(df["Frame"]=="IMP") & (df["Player"]==player)].iloc[0]
    add = df[(df["Frame"]=="ADD") & (df["Player"]==player)].iloc[0]
    diff = imp.copy()
    diff["Frame"] = "ADD→IMP"
    for col in df.columns:
        if col in ("Frame", "Player"): continue
        diff[col] = round(float(imp[col]) - float(add[col]), 2)
    return diff

def build_metrics_table(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """
    Trajectory - 4th Feature: ADD/IMP/FH1 메트릭 (프로/일반) + (IMP-ADD) 변동
    반환: ['Frame','Player', ... 지표들 ...]
    """
    rows: list[dict] = []
    for fname, n in FRAMES:
        rows.append(_calc_metrics(base_pro, fname, n, "Pro"))
        rows.append(_calc_metrics(base_ama, fname, n, "Ama"))
    df = pd.DataFrame(rows)

    # 변동(IMP-ADD) 행 추가 (각 Player 별)
    df = pd.concat([df, _diff_row(df, "Pro").to_frame().T, _diff_row(df, "Ama").to_frame().T],
                   ignore_index=True)

    # 숫자형 보장
    metric_cols = [c for c in df.columns if c not in ("Frame","Player")]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
