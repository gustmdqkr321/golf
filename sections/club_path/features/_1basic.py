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

# ────────────────────────────────────────────────────────────────────
# (위) GS 두 값: B9(Club Path), B11(Face to Path)
# ────────────────────────────────────────────────────────────────────
def build_gs_pair_table(gs_pro_df: pd.DataFrame, gs_ama_df: pd.DataFrame) -> pd.DataFrame:
    items = [
        ("Club Path (deg)",    "B9"),
        ("Face to Path (deg)", "B11"),
    ]
    rows: list[list] = []
    for label, addr in items:
        p = g_gs(gs_pro_df,  addr)
        a = g_gs(gs_ama_df, addr)
        d = (p - a) if (np.isfinite(p) and np.isfinite(a)) else float("nan")
        rows.append([label, p, a, d])
    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])

# ────────────────────────────────────────────────────────────────────
# (아래) Alignment(L/R) & Grip: 7~13
# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# (아래) Alignment(L/R) & Grip: 7~13  + 추가(14~18)
# ────────────────────────────────────────────────────────────────────

def _body_hinge_angles_base(arr: np.ndarray, start: int = 1, end: int = 10) -> np.ndarray:
    """_2BH.py와 동일 로직: 어깨-골반 vs 무릎-골반 벡터 각도(°)"""
    # 좌/우 평균 중심
    def _center_xyz(left: tuple[str,str,str], right: tuple[str,str,str]):
        L = np.array([g_base(arr, f"{left[0]}{i}") for i in range(start, end+1)])
        L = np.c_[L,
                  [g_base(arr, f"{left[1]}{i}") for i in range(start, end+1)],
                  [g_base(arr, f"{left[2]}{i}") for i in range(start, end+1)]]
        R = np.array([g_base(arr, f"{right[0]}{i}") for i in range(start, end+1)])
        R = np.c_[R,
                  [g_base(arr, f"{right[1]}{i}") for i in range(start, end+1)],
                  [g_base(arr, f"{right[2]}{i}") for i in range(start, end+1)]]
        return (L + R) * 0.5  # (N,3)

    S = _center_xyz(("AL","AM","AN"), ("BA","BB","BC"))
    P = _center_xyz(("H","I","J"),    ("K","L","M"))
    K = _center_xyz(("BP","BQ","BR"), ("CB","CC","CD"))

    U = S - P
    L = K - P
    dot = (U * L).sum(axis=1)
    nu  = np.linalg.norm(U, axis=1)
    nl  = np.linalg.norm(L, axis=1)
    denom = np.where(nu*nl == 0, np.nan, nu*nl)
    cos = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))  # (N,)

def _shoulder_minus_wrist_x(arr: np.ndarray, n: int) -> float:
    """어깨 대비 손목 X = ((AX+BM)/2) - ((AL+BA)/2) @ frame n"""
    wrist_x   = 0.5 * (g_base(arr, f"AX{n}") + g_base(arr, f"BM{n}"))
    shoulder_x= 0.5 * (g_base(arr, f"AL{n}") + g_base(arr, f"BA{n}"))
    return float(wrist_x - shoulder_x)

def build_alignment_grip_table(base_pro_arr: np.ndarray, base_ama_arr: np.ndarray) -> pd.DataFrame:
    items = [
        ("7. Ankle",     "BY1 - CK1"),
        ("8. Knee",      "BP1 - CB1"),
        ("9. Waist",     "H1 - K1"),
        ("10. Shoulder", "AL1 - BA1"),
        ("11. Elbow",    "AR1 - BG1"),
        ("12. Add",      "AY1 - BN1"),
        ("13. Imp",      "AY7 - BN7"),
    ]
    rows: list[list] = []
    for label, expr in items:
        try:    p = eval_expr_base(base_pro_arr, expr)
        except: p = float("nan")
        try:    a = eval_expr_base(base_ama_arr, expr)
        except: a = float("nan")
        d = (p - a) if (np.isfinite(p) and np.isfinite(a)) else float("nan")
        rows.append([label, p, a, d])

    # ── 14. Body Hinge 1-4 (Backswing) ─────────────────────────────
    ang_p = _body_hinge_angles_base(base_pro_arr)  # 길이 10
    ang_a = _body_hinge_angles_base(base_ama_arr)
    bh14_p = float(np.round(ang_p[6] - ang_p[3], 2))  # 4 - 1
    bh14_a = float(np.round(ang_a[6] - ang_a[3], 2))
    rows.append(["14. Body Hinge 4-7", bh14_p, bh14_a, float(np.round(bh14_p - bh14_a, 2))])

    # ── 15~17. 어깨 대비 손목 X (프레임 4,5,6) ──────────────────────
    for num, fr in zip((15,16,17), (4,5,6)):
        p = _shoulder_minus_wrist_x(base_pro_arr, fr)
        a = _shoulder_minus_wrist_x(base_ama_arr, fr)
        rows.append([f"{num}. OVER THE TOP{fr}", float(np.round(p,2)), float(np.round(a,2)), float(np.round(p-a,2))])

    # ── 18. 6 R WRI/CHD Z  (CP6 - BO6) ─────────────────────────────
    p18 = g_base(base_pro_arr, "CP6") - g_base(base_pro_arr, "BO6")
    a18 = g_base(base_ama_arr, "CP6") - g_base(base_ama_arr, "BO6")
    rows.append(["18. Casting", float(np.round(p18,2)), float(np.round(a18,2)), float(np.round(p18-a18,2))])

    return pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])
