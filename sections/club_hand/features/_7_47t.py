# sections/forces/features/_3forces47.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

# 파일 상단 import 근처
import numpy as np
import pandas as pd

NUM_EPS = 1e-12

# A(프레임 라벨)는 문자열 그대로 보존!
def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    for c in cols:
        tmp[c] = pd.to_numeric(tmp.get(c), errors="coerce")
    # 모든 req 열이 유효한(숫자) 행만 남긴다
    mask = tmp[cols].apply(np.isfinite).all(axis=1)
    return tmp.loc[mask].reset_index(drop=True)

# MAP 그대로 사용한다는 가정 (오른쪽 무릎 CB,CC,CD 포함)
# 예: MAP = {"ankle_L":('BY','BZ','CA'), ...}

def _required_cols_for_part(part: str) -> list[str]:
    if part == "knee":
        need = list(MAP["ankle_L"] + MAP["ankle_R"] + MAP["knee_L"] + MAP["knee_R"])
    elif part == "pelvis":
        need = list(MAP["knee_L"] + MAP["knee_R"] + MAP["pelvis_L"] + MAP["pelvis_R"])
    elif part == "shoulder":
        need = list(MAP["pelvis_L"] + MAP["pelvis_R"] + MAP["shoulder_L"] + MAP["shoulder_R"])
    else:
        raise ValueError("part must be one of ['knee','pelvis','shoulder']")
    # 시간열 B는 숫자여야 하므로 포함 (A는 포함하지 않음!)
    need = ["B"] + need
    # 중복 제거
    out, seen = [], set()
    for c in need:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def get_time_mask(df):
    t = pd.to_numeric(df["B"], errors="coerce").to_numpy()
    frames = df["A"].astype(str).fillna("").to_numpy()
    mask_47 = np.isin(frames, SEG_LABELS)
    return t, frames, mask_47



def to_np(df, cols):  # (N,3) float array, cm->m
    return df[list(cols)].astype(float).to_numpy() * CM2M

# ───────────────────────── 설정(디폴트) ─────────────────────────
CM2M  = 0.01
MASS  = 60.0
SEG_LABELS = ['TOP', 'TR', 'DH', 'IMP']        # 4–7 구간
YHAT = np.array([0.0, 1.0, 0.0])               # 수직축=Y
ZHAT = np.array([0.0, 0.0, 1.0])               # 수직축=Z

# 무지개(기존) 10프레임 라벨 맵 (1..10행) — 없다면 synth 용
BASE10_PHASES = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

# 좌표 매핑 (열 레터 기준)
MAP = {
    "ankle_L":   ('BY','BZ','CA'), "ankle_R":   ('CK','CL','CM'),
    "knee_L":    ('BP','BQ','BR'), "knee_R":    ('CB','CC','CD'),  # 주의: 오른무릎 = CB,CC,CD
    "pelvis_L":  ('H','I','J'),    "pelvis_R":  ('K','L','M'),
    "shoulder_L":('AL','AM','AN'), "shoulder_R":('BA','BB','BC'),
}

# ───────────────────────── 유틸 ─────────────────────────
def _letters(n: int) -> list[str]:
    """0..n-1 -> A,B,...,Z,AA,AB..."""
    out = []
    for i in range(n):
        s = ""
        x = i
        while True:
            s = chr(x % 26 + 65) + s
            x = x // 26 - 1
            if x < 0:
                break
        out.append(s)
    return out

def arr_to_letter_df(arr) -> pd.DataFrame:
    """
    numpy 2D 배열 -> A,B,C,... 컬럼명의 DataFrame
    (app의 다른 피처들과 동일한 컨벤션)
    """
    df = pd.DataFrame(arr)
    df.columns = _letters(df.shape[1])
    return df

def ensure_frames_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 'A' 열: 프레임 라벨(없으면 BASE10_PHASES로 생성)
    - 'B' 열: 시간(s 또는 ms → 자동 인식). 없으면 등간격 0,1,2,...
    반환: 원본을 복사한 DataFrame
    """
    df2 = df.copy()
    # A 열(프레임 라벨)
    if "A" not in df2.columns:
        n = len(df2)
        phases = BASE10_PHASES[:n] if n <= 10 else BASE10_PHASES + [f"F{i}" for i in range(11, n+1)]
        df2["A"] = phases[:n]
    else:
        # 문자열로 강제
        df2["A"] = df2["A"].astype(str)

    # B 열(시간)
    if "B" not in df2.columns:
        df2["B"] = np.arange(len(df2), dtype=float)
    else:
        # 숫자로 캐스팅
        df2["B"] = pd.to_numeric(df2["B"], errors="coerce")
        # 모두 NaN이면 등간격
        if df2["B"].isna().all():
            df2["B"] = np.arange(len(df2), dtype=float)
    return df2

def to_np(df: pd.DataFrame, cols: tuple[str,str,str]) -> np.ndarray:
    """(N,3) float array, cm→m 변환"""
    return df[list(cols)].astype(float).to_numpy() * CM2M

def np_mid(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """좌우 평균 중점"""
    return 0.5 * (L + R)

def to_seconds(t: np.ndarray) -> np.ndarray:
    """시간열(ms 또는 s)을 자동으로 초 단위로 변환"""
    t = np.asarray(t, dtype=float)
    # 간단 휴리스틱: 중앙 dt가 10보다 크면 ms로 보고 /1000
    return t / 1000.0 if np.nanmedian(np.diff(t)) > 10 else t

def grad_kinematics(pos, t):
    # 최소 3포인트 필요 + 시간 단조 증가 체크
    if pos.shape[0] < 3 or t.shape[0] < 3:
        return np.empty_like(pos), np.empty((max(pos.shape[0]-1,0),) + pos.shape[1:])
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        # 시간 값이 비정상이면 안전하게 빈 결과 반환
        return np.empty_like(pos), np.empty((max(pos.shape[0]-1,0),) + pos.shape[1:])
    v = np.gradient(pos, t, axis=0)
    a = np.gradient(v, t, axis=0)
    return v, a


def summarize_abs(x) -> tuple[float,float]:
    """평균±표준편차(절댓값 기준)"""
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    return float(np.mean(np.abs(x))), float(np.std(np.abs(x), ddof=0))

# ───────────────────────── 코어 계산 ─────────────────────────
def part_center_target(df: pd.DataFrame, part: str) -> tuple[np.ndarray, np.ndarray]:
    """
    각 part 별 기준점(C: 아래 세그먼트 중심)과 타깃점(T: 위 세그먼트 중심) 반환
    (모두 meter 좌표)
    """
    if part == "knee":
        C = np_mid(to_np(df, MAP["ankle_L"]),   to_np(df, MAP["ankle_R"]))
        T = np_mid(to_np(df, MAP["knee_L"]),    to_np(df, MAP["knee_R"]))
    elif part == "pelvis":
        C = np_mid(to_np(df, MAP["knee_L"]),    to_np(df, MAP["knee_R"]))
        T = np_mid(to_np(df, MAP["pelvis_L"]),  to_np(df, MAP["pelvis_R"]))
    elif part == "shoulder":
        C = np_mid(to_np(df, MAP["pelvis_L"]),  to_np(df, MAP["pelvis_R"]))
        T = np_mid(to_np(df, MAP["shoulder_L"]),to_np(df, MAP["shoulder_R"]))
    else:
        raise ValueError("part must be one of ['knee','pelvis','shoulder']")
    return C, T

def forces_and_torque_for_part(df, part, mass=MASS):
    req_cols = _required_cols_for_part(part)
    df2 = _coerce_numeric_cols(df, req_cols)  # ← A는 제외, 숫자 강제 변환
    # A는 문자열로 유지해야 하므로 df2에서 보존 (없으면 추가)
    if "A" not in df2.columns and "A" in df.columns:
        df2["A"] = df["A"].values

    # 시간/프레임
    if "B" not in df2.columns or df2.shape[0] < 3:
        # 데이터가 부족하면 빈 아웃 반환
        return {"frames": np.array([], dtype=str),
                "tau_mag": np.array([]), "F_rot": np.array([]),
                "Fy": np.array([]), "Fz": np.array([])}
    t, frames, mask_47 = get_time_mask(df2)

    # 시간 유효성 필터(유한 & 증가)
    good = np.isfinite(t)
    good[1:] &= (np.diff(t) > 0)
    good[:-1] &= (np.diff(t) > 0)
    if not good.any():
        return {"frames": np.array([], dtype=str),
                "tau_mag": np.array([]), "F_rot": np.array([]),
                "Fy": np.array([]), "Fz": np.array([])}
    df2 = df2.loc[good].reset_index(drop=True)
    t, frames, mask_47 = get_time_mask(df2)

    # 좌표
    C, T = part_center_target(df2, part)
    if C.shape[0] < 3 or T.shape[0] < 3:
        return {"frames": np.array([], dtype=str),
                "tau_mag": np.array([]), "F_rot": np.array([]),
                "Fy": np.array([]), "Fz": np.array([])}

    _, aC = grad_kinematics(C, t)
    _, aT = grad_kinematics(T, t)
    # 빈 결과 가드
    if aC.size == 0 or aT.size == 0:
        return {"frames": np.array([], dtype=str),
                "tau_mag": np.array([]), "F_rot": np.array([]),
                "Fy": np.array([]), "Fz": np.array([])}

    a_rel = aT - aC
    r = T - C

    tau = np.cross(r, mass * a_rel)
    tau_mag = np.linalg.norm(tau, axis=1)

    r_mag = np.linalg.norm(r, axis=1)
    F_rot = tau_mag / np.maximum(r_mag, NUM_EPS)

    rxY = np.cross(r, YHAT)
    rxZ = np.cross(r, ZHAT)
    denomY = np.sum(rxY*rxY, axis=1)
    denomZ = np.sum(rxZ*rxZ, axis=1)
    numY = np.sum(tau*rxY, axis=1)
    numZ = np.sum(tau*rxZ, axis=1)
    Fy = np.where(denomY>NUM_EPS, numY/denomY, np.nan)
    Fz = np.where(denomZ>NUM_EPS, numZ/denomZ, np.nan)

    return {
        "frames": frames[mask_47],
        "tau_mag": tau_mag[mask_47],
        "F_rot": F_rot[mask_47],
        "Fy": Fy[mask_47],
        "Fz": Fz[mask_47],
    }



def compute_subject(df, who, mass=MASS):
    rows, perframe = [], []
    for key, label in [("knee","무릎"),("pelvis","골반"),("shoulder","어깨")]:
        res = forces_and_torque_for_part(df, key, mass=mass)

        # 프레임별 (있으면 기록)
        if res["frames"].size > 0:
            for i, fr in enumerate(res["frames"]):
                perframe.append({
                    "선수": who, "부위": label, "Frame": fr,
                    "토크|τ|(N·m)": round(res["tau_mag"][i],2),
                    "회전력 F_rot(N)": round(res["F_rot"][i],2),
                    "Y등가힘 F_y(N)": (round(abs(res["Fy"][i]),2)
                                      if np.isfinite(res["Fy"][i]) else np.nan),
                    "Z등가힘 F_z(N)": (round(abs(res["Fz"][i]),2)
                                      if np.isfinite(res["Fz"][i]) else np.nan),
                })
            tau_m, tau_s = summarize_abs(res["tau_mag"])
            Fr_m,  Fr_s  = summarize_abs(res["F_rot"])
            Fy_m,  Fy_s  = summarize_abs(res["Fy"])
            Fz_m,  Fz_s  = summarize_abs(res["Fz"])
        else:
            tau_m=tau_s=Fr_m=Fr_s=Fy_m=Fy_s=Fz_m=Fz_s=np.nan

        rows.append({
            "선수": who, "부위": label,
            "토크|τ|(N·m) 평균±표준편차": (f"{round(tau_m,2)} ± {round(tau_s,2)}"
                                     if np.isfinite(tau_m) else "NaN ± NaN"),
            "회전력 F_rot(N) 평균±표준편차": (f"{round(Fr_m,2)} ± {round(Fr_s,2)}"
                                       if np.isfinite(Fr_m) else "NaN ± NaN"),
            "Y등가힘 F_y(N) 평균±표준편차": (f"{round(Fy_m,2)} ± {round(Fy_s,2)}"
                                       if np.isfinite(Fy_m) else "NaN ± NaN"),
            "Z등가힘 F_z(N) 평균±표준편차": (f"{round(Fz_m,2)} ± {round(Fz_s,2)}"
                                       if np.isfinite(Fz_m) else "NaN ± NaN"),
            "비율 F_rot/F_y": (round(Fr_m/Fy_m, 2)
                            if np.isfinite(Fr_m) and np.isfinite(Fy_m) and Fy_m!=0 else np.nan),
            "비율 F_rot/F_z": (round(Fr_m/Fz_m, 2)
                            if np.isfinite(Fr_m) and np.isfinite(Fz_m) and Fz_m!=0 else np.nan),
            "비율 F_y/F_z":   (round(Fy_m/Fz_m, 2)
                            if np.isfinite(Fy_m) and np.isfinite(Fz_m) and Fz_m!=0 else np.nan),
        })
    return pd.DataFrame(rows), pd.DataFrame(perframe)

# ───────────────────────── 외부 공개 API ─────────────────────────
@dataclass
class Forces47Result:
    table_summary: pd.DataFrame     # 4–7 구간 요약(평균±표준편차)
    table_perframe: pd.DataFrame    # 4–7 각 프레임 값

def build_47_forces_and_torque(
    pro_base: pd.DataFrame | np.ndarray,
    ama_base: pd.DataFrame | np.ndarray,
    *,
    mass: float = MASS,
    pro_label: str = "Pro",
    ama_label: str = "Ama",
) -> Forces47Result:
    """
    - 입력: 무지개 base (numpy 배열 또는 A,B,C... 레터 컬럼 DF)
    - 출력: (요약표, 프레임별표) 합쳐진 DataFrame(선수 컬럼으로 구분)
    - 화면/엑셀 스타일링은 상위(main.py/app.py)에서 공통 로직을 사용
    """
    # 입력 보정
    pro_df = arr_to_letter_df(pro_base) if isinstance(pro_base, np.ndarray) else pro_base.copy()
    ama_df = arr_to_letter_df(ama_base) if isinstance(ama_base, np.ndarray) else ama_base.copy()

    # 선수별 계산
    summary_pro, per_pro = compute_subject(pro_df, pro_label, mass=mass)
    summary_ama, per_ama = compute_subject(ama_df, ama_label, mass=mass)

    # 합치기
    table_summary  = pd.concat([summary_pro, summary_ama], ignore_index=True)
    table_perframe = pd.concat([per_pro, per_ama], ignore_index=True)

    return Forces47Result(table_summary=table_summary, table_perframe=table_perframe)
