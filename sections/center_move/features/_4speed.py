# sections/center_move/features/_4speed.py

import numpy as np
import pandas as pd

def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

_FRAMES = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","Finish"]
_N = 10

# ─────────────────────────────────────────────────────────
# 프레임별 상세 표 (그대로)
# ─────────────────────────────────────────────────────────
def compute_tilt_report(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    pro_label: str = "Pro",
    ama_label: str = "Ama"
) -> pd.DataFrame:
    """
    • 골반 tilt: θ_t = I_t - L_t
    • 어깨 tilt: θ_t = AM_t - BB_t
    • Δθ_t = |θ_t - θ_{t-1}|
    • v_t = Δθ_t / Δt
    • 각자의 B1~B10을 시간으로 사용
    """
    # 1) 시간 (B1~B10)
    t_pro = np.array([g(pro_arr, f"B{i}") for i in range(1, _N+1)], dtype=float)
    t_ama = np.array([g(ama_arr, f"B{i}") for i in range(1, _N+1)], dtype=float)

    # 2) Tilt
    θp_pel = np.array([g(pro_arr, f"I{i}")  - g(pro_arr, f"L{i}")  for i in range(1, _N+1)], dtype=float)
    θa_pel = np.array([g(ama_arr, f"I{i}")  - g(ama_arr, f"L{i}")  for i in range(1, _N+1)], dtype=float)
    θp_sho = np.array([g(pro_arr, f"AM{i}") - g(pro_arr, f"BB{i}") for i in range(1, _N+1)], dtype=float)
    θa_sho = np.array([g(ama_arr, f"AM{i}") - g(ama_arr, f"BB{i}") for i in range(1, _N+1)], dtype=float)

    df = pd.DataFrame({
        "Frame": _FRAMES,
        f"{pro_label} Time": t_pro.round(3),
        f"{ama_label} Time": t_ama.round(3),
        f"{pro_label} Pelvis θ": θp_pel.round(2),
        f"{ama_label} Pelvis θ": θa_pel.round(2),
        f"{pro_label} Shoulder θ": θp_sho.round(2),
        f"{ama_label} Shoulder θ": θa_sho.round(2),
    })

    # 3) Δθ & speed(프레임별)
    for part in ["Pelvis", "Shoulder"]:
        for label in [pro_label, ama_label]:
            col = f"{label} {part} θ"
            dθ = df[col].diff().abs()
            dt = df[f"{label} Time"].diff()
            df[f"Δ {col}"]      = dθ.round(2)
            df[f"{col} speed"]  = (dθ / dt).round(2)

    # 4) raw 차이 (Ama - Pro)
    df["Pelvis Δ(Ama–Pro)"]   = (df[f"{ama_label} Pelvis θ"]   - df[f"{pro_label} Pelvis θ"]).round(2)
    df["Shoulder Δ(Ama–Pro)"] = (df[f"{ama_label} Shoulder θ"] - df[f"{pro_label} Shoulder θ"]).round(2)
    return df

# ─────────────────────────────────────────────────────────
# 헬퍼: 구간 인덱스, 시간/델타 시리즈 뽑기
# ─────────────────────────────────────────────────────────
_SEGMENTS = {"1-4": (0, 3), "4-7": (3, 6), "7-10": (6, 9)}

def _times(arr: np.ndarray) -> np.ndarray:
    return np.array([g(arr, f"B{i}") for i in range(1, _N+1)], dtype=float)

def _tilt_series(arr: np.ndarray, left: str, right: str) -> np.ndarray:
    return np.array([g(arr, f"{left}{i}") - g(arr, f"{right}{i}") for i in range(1, _N+1)], dtype=float)

def _delta_abs(series: np.ndarray) -> np.ndarray:
    d = np.empty_like(series)
    d[0] = np.nan
    d[1:] = np.abs(np.diff(series))
    return d

# ─────────────────────────────────────────────────────────
# ① Δθ 합계 요약 테이블
# ─────────────────────────────────────────────────────────
def build_tilt_delta_summary_table(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    pro_label: str = "Pro",
    ama_label: str = "Ama"
) -> pd.DataFrame:

    # 각 파트별 Δθ 시퀀스
    pel_pro = _tilt_series(pro_arr, "I", "L")
    pel_ama = _tilt_series(ama_arr, "I", "L")
    sho_pro = _tilt_series(pro_arr, "AM", "BB")
    sho_ama = _tilt_series(ama_arr, "AM", "BB")

    d_pel_pro = _delta_abs(pel_pro)
    d_pel_ama = _delta_abs(pel_ama)
    d_sho_pro = _delta_abs(sho_pro)
    d_sho_ama = _delta_abs(sho_ama)

    rows = []
    for seg, (i0, i1) in _SEGMENTS.items():
        # i0+1..i1 (스텝 합)
        rows.append({
            "구간": seg,
            f"Σ Δθ ({pro_label} Pelvis)":   np.nansum(d_pel_pro[i0+1:i1+1]).round(2),
            f"Σ Δθ ({ama_label} Pelvis)":  np.nansum(d_pel_ama[i0+1:i1+1]).round(2),
            f"Σ Δθ ({pro_label} Shoulder)": np.nansum(d_sho_pro[i0+1:i1+1]).round(2),
            f"Σ Δθ ({ama_label} Shoulder)":np.nansum(d_sho_ama[i0+1:i1+1]).round(2),
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# ② 평균 속도 요약 테이블 (ΣΔθ / 구간시간, 각자 시간축)
# ─────────────────────────────────────────────────────────
def build_tilt_speed_summary_table(
    pro_arr: np.ndarray,
    ama_arr: np.ndarray,
    pro_label: str = "Pro",
    ama_label: str = "Ama"
) -> pd.DataFrame:

    # 시간
    tp = _times(pro_arr)
    ta = _times(ama_arr)

    # Δθ 시퀀스
    pel_pro = _tilt_series(pro_arr, "I", "L")
    pel_ama = _tilt_series(ama_arr, "I", "L")
    sho_pro = _tilt_series(pro_arr, "AM", "BB")
    sho_ama = _tilt_series(ama_arr, "AM", "BB")

    d_pel_pro = _delta_abs(pel_pro)
    d_pel_ama = _delta_abs(pel_ama)
    d_sho_pro = _delta_abs(sho_pro)
    d_sho_ama = _delta_abs(sho_ama)

    rows = []
    for seg, (i0, i1) in _SEGMENTS.items():
        seg_time_p = float(tp[i1] - tp[i0])
        seg_time_a = float(ta[i1] - ta[i0])

        sum_pel_p = float(np.nansum(d_pel_pro[i0+1:i1+1]))
        sum_pel_a = float(np.nansum(d_pel_ama[i0+1:i1+1]))
        sum_sho_p = float(np.nansum(d_sho_pro[i0+1:i1+1]))
        sum_sho_a = float(np.nansum(d_sho_ama[i0+1:i1+1]))

        rows.append({
            "구간": seg,
            f"avg speed ({pro_label} Pelvis)":   round(sum_pel_p / seg_time_p, 2) if seg_time_p else np.nan,
            f"avg speed ({ama_label} Pelvis)":  round(sum_pel_a / seg_time_a, 2) if seg_time_a else np.nan,
            f"avg speed ({pro_label} Shoulder)": round(sum_sho_p / seg_time_p, 2) if seg_time_p else np.nan,
            f"avg speed ({ama_label} Shoulder)":round(sum_sho_a / seg_time_a, 2) if seg_time_a else np.nan,
        })

    return pd.DataFrame(rows)
