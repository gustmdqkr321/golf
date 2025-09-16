# sections/swing_error/features/_5PS.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num - 1, col_letters_to_index(letters)])

def pelvis_x_center(arr: np.ndarray, row: int) -> float:
    # 골반 좌/우 X 평균: (Hrow + Krow)/2
    return (g(arr, f"H{row}") + g(arr, f"K{row}")) * 0.5

# ── 단일표(스샷 형태) ────────────────────────────────────────────────────────
def build_pelvis_shift_table(arr: np.ndarray) -> pd.DataFrame:
    v1 = pelvis_x_center(arr, 1)  # ① ADD
    v2 = pelvis_x_center(arr, 4)  # ② TOP
    v3 = pelvis_x_center(arr, 7)  # ③ IMP

    # GS(E12) — 없으면 NaN
    try:
        gs = g(arr, "E12")
    except Exception:
        gs = float("nan")

    rows = [
        ["ADD", "① (H1 + K1)/2", v1],
        ["TOP", "② (H4 + K4)/2", v2],
        ["IMP", "③ (H7 + K7)/2", v3],
        ["ADD-TOP", "② − ①", v2 - v1],
        ["TOP-IMP", "③ − ②", v3 - v2],
        ["ADD-IMP", "③ − ①", v3 - v1],
    ]
    df = pd.DataFrame(rows, columns=["항목", "식", "값"])
    return df

# ── 프로/일반 비교표 ─────────────────────────────────────────────────────────
def build_compare_table(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    def compute_triplet(arr):
        v1 = pelvis_x_center(arr, 1)
        v2 = pelvis_x_center(arr, 4)
        v3 = pelvis_x_center(arr, 7)
        try: gs = g(arr, "E12")
        except Exception: gs = float("nan")
        return v1, v2, v3, gs

    p1, p2, p3, pgs = compute_triplet(pro_arr)
    a1, a2, a3, ags = compute_triplet(ama_arr)

    rows = [
        ["① ADD  (H1+K1)/2",  p1,  a1,  p1 - a1],
        ["② TOP  (H4+K4)/2",  p2,  a2,  p2 - a2],
        ["③ IMP  (H7+K7)/2",  p3,  a3,  p3 - a3],
        ["ADD-TOP (②−①)",    p2 - p1,  a2 - a1,  (p2 - p1) - (a2 - a1)],
        ["TOP-IMP (③−②)",    p3 - p2,  a3 - a2,  (p3 - p2) - (a3 - a2)],
        ["ADD-IMP (③−①)",    p3 - p1,  a3 - a1,  (p3 - p1) - (a3 - a1)],
    ]
    df = pd.DataFrame(rows, columns=["항목", "프로", "일반", "차이(프로-일반)"])
    return df

# (= shoulder X) ----------------------------------------------------------------
def shoulder_x_center(arr: np.ndarray, row: int) -> float:
    # 어깨 좌/우 X 평균: (ALrow + BArow)/2
    return (g(arr, f"AL{row}") + g(arr, f"BA{row}")) * 0.5

def build_shoulder_x_table(arr: np.ndarray) -> pd.DataFrame:
    v1 = shoulder_x_center(arr, 1)  # ① ADD
    v2 = shoulder_x_center(arr, 4)  # ② TOP
    v3 = shoulder_x_center(arr, 7)  # ③ IMP
    # GS(E13) — 옵션
    try: gs = g(arr, "E13")
    except Exception: gs = float("nan")

    rows = [
        ["ADD",      "① (AL1 + BA1)/2", v1],
        ["TOP",      "② (AL4 + BA4)/2", v2],
        ["IMP",      "③ (AL7 + BA7)/2", v3],
        ["GS",       "E13(GS)",         gs],
        ["ADD-TOP",  "② − ①",           v2 - v1],
        ["TOP-IMP",  "③ − ②",           v3 - v2],
        ["ADD-IMP",  "③ − ①",           v3 - v1],
    ]
    return pd.DataFrame(rows, columns=["항목", "식", "값"])

def build_shoulder_x_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_shoulder_x_table(pro_arr)
    a = build_shoulder_x_table(ama_arr)
    df = p[["항목","값"]].merge(a[["항목","값"]], on="항목", suffixes=("·프로","·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로":"프로", "값·일반":"일반"})

# (= head X) --------------------------------------------------------------------
def build_head_x_table(arr: np.ndarray) -> pd.DataFrame:
    v1 = g(arr, "AC1")   # ①
    v2 = g(arr, "AC4")   # ②
    v3 = g(arr, "AC7")   # ③
    rows = [
        ["ADD",     "① AC1", v1],
        ["TOP",     "② AC4", v2],
        ["IMP",     "③ AC7", v3],
        ["ADD-TOP", "② − ①", v2 - v1],
        ["TOP-IMP", "③ − ②", v3 - v2],
        ["ADD-IMP", "③ − ①", v3 - v1],
    ]
    return pd.DataFrame(rows, columns=["항목","식","값"])

def build_head_x_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_head_x_table(pro_arr)
    a = build_head_x_table(ama_arr)
    df = p[["항목","값"]].merge(a[["항목","값"]], on="항목", suffixes=("·프로","·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로":"프로", "값·일반":"일반"})

# (= lift/down Y: waist) --------------------------------------------------------
def pelvis_y_center(arr: np.ndarray, row: int) -> float:
    # 골반 좌/우 Y 평균: (Irow + Lrow)/2
    return (g(arr, f"I{row}") + g(arr, f"L{row}")) * 0.5

def build_waist_lifty_table(arr: np.ndarray) -> pd.DataFrame:
    v_add = pelvis_y_center(arr, 1)
    v_top = pelvis_y_center(arr, 4)
    v_imp = pelvis_y_center(arr, 7)
    v_f1  = pelvis_y_center(arr, 8)
    try: gs = g(arr, "E14")   # Pelvis(IMP) GS
    except Exception: gs = float("nan")

    rows = [
        ["ADD - TOP", "(I4 + L4)/2 − (I1 + L1)/2", v_top - v_add],
        ["TOP - IMP", "(I7 + L7)/2 − (I4 + L4)/2", v_imp - v_top],
        ["Pelvis(IMP)","E14(GS)",                   gs],
        ["IMP - F1",  "(I8 + L8)/2 − (I7 + L7)/2", v_f1 - v_imp],
    ]
    return pd.DataFrame(rows, columns=["항목","식","값"])

def build_waist_lifty_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_waist_lifty_table(pro_arr)
    a = build_waist_lifty_table(ama_arr)
    df = p[["항목","값"]].merge(a[["항목","값"]], on="항목", suffixes=("·프로","·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로":"프로", "값·일반":"일반"})

# (= lift/down Y: shoulder) -----------------------------------------------------
def shoulder_y_center(arr: np.ndarray, row: int) -> float:
    # 어깨 좌/우 Y 평균: (AMrow + BBrow)/2
    return (g(arr, f"AM{row}") + g(arr, f"BB{row}")) * 0.5

def build_shoulder_lifty_table(arr: np.ndarray) -> pd.DataFrame:
    v_add = shoulder_y_center(arr, 1)
    v_top = shoulder_y_center(arr, 4)
    v_imp = shoulder_y_center(arr, 7)
    v_f1  = shoulder_y_center(arr, 8)
    try: gs = g(arr, "E15")   # Ribcage(IMP) GS
    except Exception: gs = float("nan")

    rows = [
        ["ADD - TOP", "(AM4 + BB4)/2 − (AM1 + BB1)/2", v_top - v_add],
        ["TOP - IMP", "(AM7 + BB7)/2 − (AM4 + BB4)/2", v_imp - v_top],
        ["Ribcage(IMP)", "E15(GS)",                     gs],
        ["IMP - F1",   "(AM8 + BB8)/2 − (AM7 + BB7)/2", v_f1 - v_imp],
    ]
    return pd.DataFrame(rows, columns=["항목","식","값"])

def build_shoulder_lifty_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_shoulder_lifty_table(pro_arr)
    a = build_shoulder_lifty_table(ama_arr)
    df = p[["항목","값"]].merge(a[["항목","값"]], on="항목", suffixes=("·프로","·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로":"프로", "값·일반":"일반"})

# (= head Y) --------------------------------------------------------------------
def build_head_y_table(arr: np.ndarray) -> pd.DataFrame:
    d_add_top = g(arr, "AD4") - g(arr, "AD1")
    d_top_imp = g(arr, "AD7") - g(arr, "AD4")
    d_imp_f1  = g(arr, "AD8") - g(arr, "AD7")
    rows = [
        ["ADD - TOP", "AD4 − AD1", d_add_top],
        ["TOP - IMP", "AD7 − AD4", d_top_imp],
        ["IMP - F1",  "AD8 − AD7", d_imp_f1],
    ]
    return pd.DataFrame(rows, columns=["항목","식","값"])

def build_head_y_compare(pro_arr: np.ndarray, ama_arr: np.ndarray) -> pd.DataFrame:
    p = build_head_y_table(pro_arr)
    a = build_head_y_table(ama_arr)
    df = p[["항목","값"]].merge(a[["항목","값"]], on="항목", suffixes=("·프로","·일반"))
    df["차이(프로-일반)"] = df["값·프로"] - df["값·일반"]
    return df.rename(columns={"값·프로":"프로", "값·일반":"일반"})