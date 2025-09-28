from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── A1 셀 접근 ───────────────────────────────────────────
_CELL = re.compile(r"^([A-Za-z]+)(\d+)$")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g_base(arr: np.ndarray, addr: str) -> float:
    m = _CELL.match(addr.strip())
    if not m: return float("nan")
    c = _col_idx(m.group(1))
    r = int(m.group(2)) - 1
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

# ── 좌표 & 시간 ──────────────────────────────────────────
_FRAMES = 10  # 1..10

def _stack_xyz(arr: np.ndarray, X: str, Y: str, Z: str) -> np.ndarray:
    pts = []
    for i in range(1, _FRAMES+1):
        pts.append([g_base(arr, f"{X}{i}"), g_base(arr, f"{Y}{i}"), g_base(arr, f"{Z}{i}")])
    return np.asarray(pts, dtype=float)

def _time_vec(arr: np.ndarray) -> np.ndarray:
    return np.asarray([g_base(arr, f"B{i}") for i in range(1, _FRAMES+1)], dtype=float)

# ── 계산 유틸 ────────────────────────────────────────────
def _segment_distance(coords_xyz: np.ndarray, steps: list[int]) -> float:
    diffs = np.linalg.norm(np.diff(coords_xyz, axis=0), axis=1)  # 길이 9
    return float(np.sum(diffs[steps]) / 100.0)  # cm→m

def _avg_speed(distance_m: float, dt: float) -> float:
    return float(distance_m / dt) if dt else float("nan")

def _avg_accel(avg_speed: float, dt: float) -> float:
    return float(avg_speed / dt) if dt else float("nan")

# ── 메인 테이블 ──────────────────────────────────────────
def build_club_hand_table(base_pro: np.ndarray,
                          base_ama: np.ndarray,
                          pro_label: str = "Pro",
                          ama_label: str = "Ama",
                          club_mass: float = 0.20,
                          hand_mass: float = 0.50) -> pd.DataFrame:
    # 좌표 & 시간
    pro_chd = _stack_xyz(base_pro, "CN", "CO", "CP")
    ama_chd = _stack_xyz(base_ama, "CN", "CO", "CP")
    pro_wri = _stack_xyz(base_pro, "AX", "AY", "AZ")
    ama_wri = _stack_xyz(base_ama, "AX", "AY", "AZ")

    t_pro = _time_vec(base_pro)
    t_ama = _time_vec(base_ama)

    add_top_steps = [0, 1, 2]   # 1→4
    top_imp_steps = [3, 4, 5]   # 4→7

    pro_dt_add_top = t_pro[3] - t_pro[0]
    pro_dt_top_imp = t_pro[6] - t_pro[3]
    ama_dt_add_top = t_ama[3] - t_ama[0]
    ama_dt_top_imp = t_ama[6] - t_ama[3]

    def pack_row(player: str, part: str, dist_add_top: float, dt_add_top: float,
                 dist_top_imp: float, dt_top_imp: float, mass: float) -> dict:
        v1 = _avg_speed(dist_add_top, dt_add_top)
        v2 = _avg_speed(dist_top_imp, dt_top_imp)
        a2 = _avg_accel(v2, dt_top_imp)
        F  = float(mass * a2) if not np.isnan(a2) else float("nan")
        return {
            "골퍼": player,
            "부위": part,
            "ADD→TOP 이동거리(m)": round(dist_add_top, 2),
            "ADD→TOP 평균속도(m/s)": round(v1, 2) if not np.isnan(v1) else np.nan,
            "TOP→IMP 이동거리(m)": round(dist_top_imp, 2),
            "TOP→IMP 평균속도(m/s)": round(v2, 2) if not np.isnan(v2) else np.nan,
            "TOP→IMP 평균가속도(m/s²)": round(a2, 2) if not np.isnan(a2) else np.nan,
            "임팩트 순간 힘(N)": round(F, 2) if not np.isnan(F) else np.nan,
        }

    rows = []
    # 순서: Pro 클럽 → Ama 클럽 → Pro 손 → Ama 손
    rows.append(pack_row(pro_label, "클럽",
                         _segment_distance(pro_chd, add_top_steps), pro_dt_add_top,
                         _segment_distance(pro_chd, top_imp_steps), pro_dt_top_imp,
                         club_mass))
    rows.append(pack_row(ama_label, "클럽",
                         _segment_distance(ama_chd, add_top_steps), ama_dt_add_top,
                         _segment_distance(ama_chd, top_imp_steps), ama_dt_top_imp,
                         club_mass))
    rows.append(pack_row(pro_label, "손",
                         _segment_distance(pro_wri, add_top_steps), pro_dt_add_top,
                         _segment_distance(pro_wri, top_imp_steps), pro_dt_top_imp,
                         hand_mass))
    rows.append(pack_row(ama_label, "손",
                         _segment_distance(ama_wri, add_top_steps), ama_dt_add_top,
                         _segment_distance(ama_wri, top_imp_steps), ama_dt_top_imp,
                         hand_mass))

    df = pd.DataFrame(rows)

    # 비율 계산 (Pro=100)
    for metric in ["ADD→TOP 평균속도(m/s)", "임팩트 순간 힘(N)"]:
        for part in ["클럽", "손"]:
            try:
                pro_val = float(df[(df["골퍼"]==pro_label) & (df["부위"]==part)][metric].iloc[0])
                if pro_val == 0 or np.isnan(pro_val):
                    df.loc[df["부위"]==part, f"{metric} 비율(Pro=100)"] = np.nan
                else:
                    df.loc[df["부위"]==part, f"{metric} 비율(Pro=100)"] = (
                        df.loc[df["부위"]==part, metric] / pro_val * 100
                    ).round(2)
            except Exception:
                df.loc[df["부위"]==part, f"{metric} 비율(Pro=100)"] = np.nan

    return df
