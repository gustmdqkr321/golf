# sections/club_hand/features/_8_kseq.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import re
import numpy as np
import pandas as pd

HANDEDNESS_DEFAULT = "right"   # "right" or "left"
SMOOTH_WIN = 7
PREIMPACT_MARGIN = 0.03  # 키네틱 Down 상한: 임팩트 30ms 전

# ────────────────────────── 마커 탐색 ──────────────────────────
def _find_triplet(df: pd.DataFrame, base_patterns) -> Optional[Dict[str, str]]:
    cols = df.columns.tolist()
    for base in base_patterns:
        cand = {}
        ok = True
        for ax in ("X", "Y", "Z"):
            pat = re.compile(rf"^{base}\s{ax}$", re.IGNORECASE)
            m = [c for c in cols if pat.match(c)]
            if m: cand[ax] = m[0]
            else: ok = False; break
        if ok: return cand
    for base in base_patterns:
        cand = {}; ok = True
        for ax in ("X", "Y", "Z"):
            pat = re.compile(rf"{base}[\s_]{ax}$", re.IGNORECASE)
            m = [c for c in cols if pat.search(c)]
            if m: cand[ax] = sorted(m, key=len)[0]
            else: ok = False; break
        if ok: return cand
    return None

def _build_markers(df: pd.DataFrame) -> Dict[str, Optional[Dict[str, str]]]:
    return {
        "knee_L": _find_triplet(df, [r"LKneeOut", r"LeftKnee", r"KneeLeft", r"L.*Knee"]),
        "knee_R": _find_triplet(df, [r"RKneeOut", r"RightKnee", r"KneeRight", r"R.*Knee"]),
        "waist_L": _find_triplet(df, [r"WaistLeft", r"L.*Waist", r"LeftWaist"]),
        "waist_R": _find_triplet(df, [r"WaistRight", r"R.*Waist", r"RightWaist"]),
        "shoulder_L": _find_triplet(df, [r"LShoulderTop", r"ShoulderLeft", r"L.*Shoulder"]),
        "shoulder_R": _find_triplet(df, [r"RShoulderTop", r"ShoulderRight", r"R.*Shoulder"]),
        "wrist_L": _find_triplet(df, [r"LWristTop", r"WristLeft", r"L.*Wrist"]),
        "wrist_R": _find_triplet(df, [r"RWristTop", r"WristRight", r"R.*Wrist"]),
        "clubhead": _find_triplet(df, [r"Marker[_ ]?2:2:1", r"Clubhead"]),
        "shaft_dir": _find_triplet(df, [r"Shaft Direction", r"ShaftDirection", r"Shaft_Direction"]),
    }

def _get_xyz(df: pd.DataFrame, trip: Dict[str, str], scale: float = 1.0):
    X = pd.to_numeric(df[trip["X"]], errors="coerce").to_numpy() * scale
    Y = pd.to_numeric(df[trip["Y"]], errors="coerce").to_numpy() * scale
    Z = pd.to_numeric(df[trip["Z"]], errors="coerce").to_numpy() * scale
    return X, Y, Z

# ────────────────────────── 시계열 유틸 ──────────────────────────
def _smooth(a: np.ndarray, w: int = SMOOTH_WIN) -> np.ndarray:
    return pd.Series(a).rolling(window=w, center=True, min_periods=1).mean().to_numpy()

def _yaw_from_dx_dz(dx: np.ndarray, dz: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    th = np.arctan2(dz, dx)
    th = np.unwrap(th)
    return _smooth(th, w=win)  # radians

def _omega_deg_per_s(theta_rad: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(theta_rad, t) * (180.0 / np.pi)

def _tangential_accel(X: np.ndarray, Z: np.ndarray, t: np.ndarray, smooth_win: int = SMOOTH_WIN) -> np.ndarray:
    Xs, Zs = _smooth(X, smooth_win), _smooth(Z, smooth_win)
    dX, dZ = np.gradient(Xs, t), np.gradient(Zs, t)
    aX, aZ = np.gradient(dX, t), np.gradient(dZ, t)
    vmag = np.hypot(dX, dZ)
    ux = np.divide(dX, vmag, out=np.zeros_like(dX), where=vmag > 1e-9)
    uz = np.divide(dZ, vmag, out=np.zeros_like(dZ), where=vmag > 1e-9)
    return aX * ux + aZ * uz  # cm/s^2 (입력이 cm일 때)

# ────────────────────────── 이벤트 검출 ──────────────────────────
def _detect_events(df: pd.DataFrame, markers: Dict[str, Dict[str, str]], t: np.ndarray) -> Tuple[int, int]:
    if "Impact" in df.columns:
        imp = pd.to_numeric(df["Impact"], errors="coerce").fillna(0).to_numpy()
        cand = np.where(imp > 0.5)[0]
        impact_idx = int(cand[0]) if len(cand) else int(np.nanargmax(imp))
    else:
        impact_idx = len(df) - 1

    if markers.get("clubhead"):
        _, Cy, _ = _get_xyz(df, markers["clubhead"], 1.0)
        top_idx = int(np.nanargmax(Cy[:impact_idx])) if impact_idx > 0 else 0
    elif markers.get("shoulder_L") and markers.get("shoulder_R"):
        _, yL, _ = _get_xyz(df, markers["shoulder_L"], 1.0)
        _, yR, _ = _get_xyz(df, markers["shoulder_R"], 1.0)
        y_c = 0.5 * (yL + yR)
        top_idx = int(np.nanargmax(y_c[:impact_idx])) if impact_idx > 0 else 0
    else:
        top_idx = max(0, impact_idx // 2)

    return top_idx, impact_idx

# ────────────────────────── 구간 라벨 (10 키프레임) ──────────────────────────
# 원하는 전역 순서: ADD, BH, BH2, TOP, TR, DH, IMP, FH1, FH2, FIN
# 이번 표는 Back(ADD~TOP), Down(TR~IMP)만 사용.
def _seg_named_label(t_val: float, phase: str, t0: float, t_top: float, t_imp: float) -> str:
    eps = 1e-9
    if phase == "backswing":
        if t_top <= t0 + eps:
            return "ADD"
        pos = (t_val - t0) / max(t_top - t0, eps)  # 0..1
        if pos >= 1 - eps: return "TOP"
        if pos < 1/3:      return "ADD"
        if pos < 2/3:      return "BH"
        return "BH2"
    else:
        if t_imp <= t_top + eps:
            return "TR"
        pos = (t_val - t_top) / max(t_imp - t_top, eps)  # 0..1
        if pos >= 1 - eps: return "IMP"
        if pos < 1/3:      return "TR"
        if pos < 2/3:      return "DH"
        return "DH"

# ────────────────────────── 시계열 생성 ──────────────────────────
def _compute_kinematic_series(df: pd.DataFrame, t: np.ndarray, markers: Dict[str, Dict[str, str]], handedness: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if markers["knee_L"] is not None and markers["knee_R"] is not None:
        Lx, Ly, Lz = _get_xyz(df, markers["knee_L"], 1.0)
        Rx, Ry, Rz = _get_xyz(df, markers["knee_R"], 1.0)
        th = _yaw_from_dx_dz(Rx - Lx, Rz - Lz)
        out["Knee"] = np.abs(_omega_deg_per_s(th, t))
    if markers["waist_L"] is not None and markers["waist_R"] is not None:
        Lx, Ly, Lz = _get_xyz(df, markers["waist_L"], 1.0)
        Rx, Ry, Rz = _get_xyz(df, markers["waist_R"], 1.0)
        th = _yaw_from_dx_dz(Rx - Lx, Rz - Lz)
        out["Pelvis"] = np.abs(_omega_deg_per_s(th, t))
    if markers["shoulder_L"] is not None and markers["shoulder_R"] is not None:
        Lx, Ly, Lz = _get_xyz(df, markers["shoulder_L"], 1.0)
        Rx, Ry, Rz = _get_xyz(df, markers["shoulder_R"], 1.0)
        th = _yaw_from_dx_dz(Rx - Lx, Rz - Lz)
        out["Shoulder"] = np.abs(_omega_deg_per_s(th, t))
    if handedness.lower() == "right":
        parent, child, label = "shoulder_L", "wrist_L", "Hand"
    else:
        parent, child, label = "shoulder_R", "wrist_R", "Hand"
    if markers.get(parent) is not None and markers.get(child) is not None:
        Px, Py, Pz = _get_xyz(df, markers[parent], 1.0)
        Cx, Cy, Cz = _get_xyz(df, markers[child], 1.0)
        th = _yaw_from_dx_dz(Cx - Px, Cz - Pz)
        out[label] = np.abs(_omega_deg_per_s(th, t))
    if markers["shaft_dir"] is not None:
        Sx, Sy, Sz = _get_xyz(df, markers["shaft_dir"], 1.0)
        th = _yaw_from_dx_dz(Sx, Sz)
        out["Club"] = np.abs(_omega_deg_per_s(th, t))
    elif markers["clubhead"] is not None:
        Cx, Cy, Cz = _get_xyz(df, markers["clubhead"], 1.0)
        Cx_s, Cz_s = _smooth(Cx), _smooth(Cz)
        dCx, dCz = np.gradient(Cx_s, t), np.gradient(Cz_s, t)
        th = _yaw_from_dx_dz(dCx, dCz)
        out["Club"] = np.abs(_omega_deg_per_s(th, t))
    return out

def _compute_kinetic_series(df: pd.DataFrame, t: np.ndarray, markers: Dict[str, Dict[str, str]], handedness: str) -> Dict[str, np.ndarray]:
    seg_pos: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if markers["knee_L"] is not None and markers["knee_R"] is not None:
        Lx, Ly, Lz = _get_xyz(df, markers["knee_L"], 1.0)
        Rx, Ry, Rz = _get_xyz(df, markers["knee_R"], 1.0)
        seg_pos["Knee"] = (0.5 * (Lx + Rx), 0.5 * (Ly + Ry), 0.5 * (Lz + Rz))
    if markers["waist_L"] is not None and markers["waist_R"] is not None:
        Lx, Ly, Lz = _get_xyz(df, markers["waist_L"], 1.0)
        Rx, Ry, Rz = _get_xyz(df, markers["waist_R"], 1.0)
        seg_pos["Pelvis"] = (0.5 * (Lx + Rx), 0.5 * (Ly + Ry), 0.5 * (Lz + Rz))
    if markers["shoulder_L"] is not None and markers["shoulder_R"] is not None:
        Lx, Ly, Lz = _get_xyz(df, markers["shoulder_L"], 1.0)
        Rx, Ry, Rz = _get_xyz(df, markers["shoulder_R"], 1.0)
        seg_pos["Shoulder"] = (0.5 * (Lx + Rx), 0.5 * (Ly + Ry), 0.5 * (Lz + Rz))
    if handedness.lower() == "right" and markers["wrist_L"] is not None:
        seg_pos["Hand"] = _get_xyz(df, markers["wrist_L"], 1.0)
    if handedness.lower() == "left" and markers["wrist_R"] is not None:
        seg_pos["Hand"] = _get_xyz(df, markers["wrist_R"], 1.0)
    if markers["clubhead"] is not None:
        seg_pos["Club"] = _get_xyz(df, markers["clubhead"], 1.0)
    return {seg: _tangential_accel(X, Z, t, SMOOTH_WIN) for seg, (X, Y, Z) in seg_pos.items()}

# ────────────────────────── 피크 추출 ──────────────────────────
def _peak_pick(series_dict: Dict[str, np.ndarray],
               t: np.ndarray,
               windows: Dict[str, Tuple[int, int]],
               t0: float, t_top: float, t_imp: float,
               positive_only: bool) -> dict:
    out = {"backswing": {}, "downswing": {}}
    for phase, (i0, i1) in windows.items():
        for seg, arr in series_dict.items():
            s = arr[i0:i1+1]
            if s.size == 0: 
                continue
            if positive_only:
                sp = s.copy(); sp[sp < 0] = -np.inf
                k = int(np.nanargmax(sp)) if np.any(np.isfinite(sp)) else int(np.nanargmax(s))
            else:
                k = int(np.nanargmax(s))
            idx = i0 + k
            tt = float(t[idx])
            val = float(s[k])
            seg_name = _seg_named_label(tt, phase, t0, t_top, t_imp)
            out[phase][seg] = (seg_name, tt, val, idx)
    return out

# ────────────────────────── 단일 표 생성(한 사람·한 구간) ──────────────────────────
def _phase_table_from_peaks(peaks: dict, phase: str, *,
                            title_who: str,
                            round_time: int = 6, round_val: int = 2) -> pd.DataFrame:
    """
    peaks: {"backswing": {"Knee": (seg_label, t, v, idx), ...}, "downswing": {...}}
    phase: "backswing" | "downswing"
    반환: 순서(시간), 분절, 구간, 시각(s), 값, 프레임
    """
    order = ("Knee", "Pelvis", "Shoulder", "Hand", "Club")
    rows = []
    for seg in order:
        tup = peaks.get(phase, {}).get(seg)  # (label, t, v, idx)
        if tup is None:
            continue
        label, tt, val, idx = tup
        rows.append([seg, label, tt, val, idx])

    # 시간 오름차순, 동시엔 값 큰 순, 그 다음 분절명
    rows.sort(key=lambda r: (r[2], -r[3], r[0].lower()))

    # 순서 부여
    out = []
    for i, (seg, lab, tt, val, idx) in enumerate(rows, start=1):
        out.append([i, seg, lab, round(tt, round_time), round(val, round_val), idx])

    return pd.DataFrame(
        out, columns=[f"[{title_who}] 순서", "분절", "구간", "시각(s)", "값", "프레임"]
    )

# ────────────────────────── 메인 빌더 (8개 표 반환) ──────────────────────────
def build_kinematic_and_kinetic_tables_gears(
    gears_pro_df: pd.DataFrame,
    gears_ama_df: pd.DataFrame,
    *,
    pro_name: str = "프로",
    ama_name: str = "아마",
    handedness: str = HANDEDNESS_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """
    반환 dict 키:
      - "키네마틱 - 프로 - Back", "키네마틱 - 프로 - Down",
        "키네마틱 - 아마 - Back", "키네마틱 - 아마 - Down",
      - "키네틱   - 프로 - Back", "키네틱   - 프로 - Down",
        "키네틱   - 아마 - Back", "키네틱   - 아마 - Down"
    각 표 컬럼: [ '[사람] 순서', '분절', '구간', '시각(s)', '값', '프레임' ]
    """
    def _time_vector(df: pd.DataFrame) -> np.ndarray:
        if "Time(sec)" in df.columns:
            t = pd.to_numeric(df["Time(sec)"], errors="coerce").to_numpy()
            if np.isnan(t).any(): t = np.arange(len(df), dtype=float)
        else:
            t = np.arange(len(df), dtype=float)
        return t

    def _compute_for(df: pd.DataFrame):
        t = _time_vector(df)
        mk = _build_markers(df)
        top_idx, impact_idx = _detect_events(df, mk, t)

        # 키네마틱
        kin_series = _compute_kinematic_series(df, t, mk, handedness)
        win_kin = {"backswing": (0, top_idx), "downswing": (top_idx, impact_idx)}
        peaks_kin = _peak_pick(
            kin_series, t, win_kin,
            float(t[0]), float(t[top_idx]), float(t[impact_idx]),
            positive_only=False
        )

        # 키네틱 (임팩트 30ms 전까지)
        impact_time = float(t[impact_idx])
        pre_time = impact_time - PREIMPACT_MARGIN
        i_pre = int(np.searchsorted(t, pre_time, side="right"))
        i_pre = max(top_idx, min(i_pre, impact_idx))

        kin2_series = _compute_kinetic_series(df, t, mk, handedness)
        win_kin2 = {"backswing": (0, top_idx), "downswing": (top_idx, i_pre)}
        peaks_kin2 = _peak_pick(
            kin2_series, t, win_kin2,
            float(t[0]), float(t[top_idx]), float(t[impact_idx]),
            positive_only=True
        )
        return peaks_kin, peaks_kin2

    # 프로/아마 계산
    peaks_kin_p,  peaks_kin2_p  = _compute_for(gears_pro_df)
    peaks_kin_a,  peaks_kin2_a  = _compute_for(gears_ama_df)

    # 4분할 표 생성 (키네마틱/키네틱 각각)
    tables: dict[str, pd.DataFrame] = {}

    tables["키네마틱 - 프로 - Back"] = _phase_table_from_peaks(peaks_kin_p, "backswing", title_who=f"{pro_name}")
    tables["키네마틱 - 프로 - Down"] = _phase_table_from_peaks(peaks_kin_p, "downswing", title_who=f"{pro_name}")
    tables["키네마틱 - 아마 - Back"] = _phase_table_from_peaks(peaks_kin_a, "backswing", title_who=f"{ama_name}")
    tables["키네마틱 - 아마 - Down"] = _phase_table_from_peaks(peaks_kin_a, "downswing", title_who=f"{ama_name}")

    tables["키네틱   - 프로 - Back"] = _phase_table_from_peaks(peaks_kin2_p, "backswing", title_who=f"{pro_name}")
    tables["키네틱   - 프로 - Down"] = _phase_table_from_peaks(peaks_kin2_p, "downswing", title_who=f"{pro_name}")
    tables["키네틱   - 아마 - Back"] = _phase_table_from_peaks(peaks_kin2_a, "backswing", title_who=f"{ama_name}")
    tables["키네틱   - 아마 - Down"] = _phase_table_from_peaks(peaks_kin2_a, "downswing", title_who=f"{ama_name}")

    return tables
