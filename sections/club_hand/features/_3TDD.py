from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ── A1 셀 접근 유틸 ─────────────────────────────────────────────────────
_CELL = re.compile(r"^([A-Za-z]+)(\d+)$")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx*26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1

def g_base(arr: np.ndarray, addr: str) -> float:
    m = _CELL.match(addr.strip())
    if not m: return float("nan")
    c = _col_idx(m.group(1)); r = int(m.group(2)) - 1
    try:    return float(arr[r, c])
    except Exception: return float("nan")

# ── 프레임/좌표 도우미 ──────────────────────────────────────────────────
_FRAMES = 10  # 1..10 (구간: 1-2 ~ 9-10)

def _stack_xyz(arr: np.ndarray, X: str, Y: str, Z: str) -> np.ndarray:
    """주어진 컬럼 레터(X,Y,Z)의 1..10 프레임 값을 (10,3)로 스택"""
    pts = []
    for i in range(1, _FRAMES+1):
        pts.append([g_base(arr, f"{X}{i}"), g_base(arr, f"{Y}{i}"), g_base(arr, f"{Z}{i}")])
    return np.asarray(pts, dtype=float)

def _avg_points(P: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    """두 점 평균(둘 중 하나 None이면 그 하나를 그대로 사용)"""
    if Q is None: return P
    return (P + Q) / 2.0

def _segment_angles_deg(seq: np.ndarray) -> np.ndarray:
    """
    연속 프레임 벡터 간 각도(도). seq.shape=(10,3).
    각도 길이 9: angle_t = arccos( <û_t, û_{t+1}> )
    """
    v0, v1 = seq[:-1], seq[1:]
    n0 = np.linalg.norm(v0, axis=1, keepdims=True)
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n0 = np.where(n0==0, np.nan, n0)
    n1 = np.where(n1==0, np.nan, n1)
    u0, u1 = v0/n0, v1/n1
    dot = np.einsum("ij,ij->i", u0, u1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

# ── 제너릭 TDD 계산 ────────────────────────────────────────────────────
def build_tdd_table_generic(
    base_pro: np.ndarray,
    base_ama: np.ndarray,
    *,
    # 이동(중심) 좌표 정의: disp_A(필수), disp_B(옵션: 있으면 평균)
    disp_A: tuple[str,str,str],                # 예: ("BP","BQ","BR")  (왼)
    disp_B: tuple[str,str,str] | None = None,  # 예: ("CB","CC","CD")  (오른). 없으면 단일 포인트
    # 회전 벡터 정의: rot_from → rot_to. 없으면 disp_A/disp_B를 이용해 (B-A)로 계산
    rot_from: tuple[str,str,str] | None = None,
    rot_to:   tuple[str,str,str] | None = None,
    # 단위 변환
    rot_to_m: float = 0.01,   # 회전환산(m/deg)
    label: str = "Part"       # 표 제목/식별용 라벨(테이블에는 포함 안 함)
) -> pd.DataFrame:
    """
    반환 컬럼:
      ['구간',
       '이동(Pro,m)','이동(Ama,m)',
       '회전환산(Pro,m)','회전환산(Ama,m)',
       '회전량(Pro,deg)','회전량(Ama,deg)',
       'TDD(Pro,m)','TDD(Ama,m)']

    하단 요약행 4줄 추가: '1-4','4-7','7-10'(부호 포함 합), 'Total'(절댓값 합)
    """
    # ── Pro/Ama 좌표 스택
    # 이동(중심) 포인트
    A_p = _stack_xyz(base_pro, *disp_A)
    A_a = _stack_xyz(base_ama, *disp_A)
    B_p = _stack_xyz(base_pro, *disp_B) if disp_B else None
    B_a = _stack_xyz(base_ama, *disp_B) if disp_B else None

    ctr_p = _avg_points(A_p, B_p)        # (10,3)
    ctr_a = _avg_points(A_a, B_a)        # (10,3)

    # 회전 벡터
    if rot_from is not None and rot_to is not None:
        Rf_p = _stack_xyz(base_pro, *rot_from)
        Rt_p = _stack_xyz(base_pro, *rot_to)
        Rf_a = _stack_xyz(base_ama, *rot_from)
        Rt_a = _stack_xyz(base_ama, *rot_to)
        vec_p = Rt_p - Rf_p
        vec_a = Rt_a - Rf_a
    else:
        # disp_B 가 있는 경우: (B - A) 벡터, 없으면 회전량 0 처리
        if disp_B is not None:
            vec_p = _stack_xyz(base_pro, *disp_B) - _stack_xyz(base_pro, *disp_A)
            vec_a = _stack_xyz(base_ama, *disp_B) - _stack_xyz(base_ama, *disp_A)
        else:
            vec_p = np.zeros_like(ctr_p)
            vec_a = np.zeros_like(ctr_a)

    # ── 구간별 기본량
    # 이동거리(m)
    disp_p = np.linalg.norm(ctr_p[1:] - ctr_p[:-1], axis=1) / 100.0
    disp_a = np.linalg.norm(ctr_a[1:] - ctr_a[:-1], axis=1) / 100.0
    # 회전량(도)
    ang_p  = _segment_angles_deg(vec_p)
    ang_a  = _segment_angles_deg(vec_a)
    # 회전환산(m)
    rot_p  = ang_p * float(rot_to_m)
    rot_a  = ang_a * float(rot_to_m)
    # TDD(m)
    tdd_p  = disp_p + rot_p
    tdd_a  = disp_a + rot_a

    labels = [f"{i}-{i+1}" for i in range(1, _FRAMES)]
    df = pd.DataFrame({
        "구간": labels,
        "이동(Pro,m)":       np.round(disp_p, 2),
        "이동(Ama,m)":       np.round(disp_a, 2),
        "회전환산(Pro,m)":   np.round(rot_p,  2),
        "회전환산(Ama,m)":   np.round(rot_a,  2),
        "회전량(Pro,deg)":   np.round(ang_p,  2),
        "회전량(Ama,deg)":   np.round(ang_a,  2),
        "TDD(Pro,m)":        np.round(tdd_p,  2),
        "TDD(Ama,m)":        np.round(tdd_a,  2),
    })

    # ── 요약행: 1-4 / 4-7 / 7-10 (부호 포함 합), Total (절댓값 합)
    segs = {"1-4": (0,2), "4-7": (3,5), "7-10": (6,9)}
    rows = []
    for name,(i0,i1) in segs.items():
        d = {"구간": name}
        for col in df.columns[1:]:
            d[col] = round(df.loc[i0:i1, col].astype(float).sum(), 2)
        rows.append(d)
    dT = {"구간": "Total"}
    for col in df.columns[1:]:
        dT[col] = round(np.abs(df[col].astype(float)).sum(), 2)
    rows.append(dT)

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

# ── 편의 래퍼(예: 무릎) ────────────────────────────────────────────────
def build_knee_tdd_table(base_pro: np.ndarray, base_ama: np.ndarray,
                         rot_to_m: float = 0.01) -> pd.DataFrame:
    """
    무릎 기본 사양:
      이동 중심: (왼무릎 BP,BQ,BR) & (오른무릎 CB,CC,CD) 의 평균
      회전 벡터: 오른무릎 - 왼무릎
    """
    return build_tdd_table_generic(
        base_pro, base_ama,
        disp_A=("BP","BQ","BR"),
        disp_B=("CB","CC","CD"),
        rot_from=None, rot_to=None,   # disp_B-disp_A 사용
        rot_to_m=rot_to_m,
        label="Knee",
    )

def build_hip_tdd_table(base_pro: np.ndarray, base_ama: np.ndarray,
                        rot_to_m: float = 0.01) -> pd.DataFrame:
    """
    골반 기본 사양:
      이동 중심: (왼골반 H,I,J) & (오른골반 K,L,M) 의 평균
      회전 벡터: 오른골반 - 왼골반
    """
    return build_tdd_table_generic(
        base_pro, base_ama,
        disp_A=("H","I","J"),
        disp_B=("K","L","M"),
        rot_from=None, rot_to=None,   # disp_B - disp_A 사용
        rot_to_m=rot_to_m,
        label="Hip",
    )

def build_shoulder_tdd_table(base_pro: np.ndarray, base_ama: np.ndarray,
                             rot_to_m: float = 0.01) -> pd.DataFrame:
    """
    어깨 기본 사양:
      이동 중심: (왼어깨 AL,AM,AN) & (오른어깨 BA,BB,BC) 의 평균
      회전 벡터: 오른어깨 - 왼어깨
    """
    return build_tdd_table_generic(
        base_pro, base_ama,
        disp_A=("AL","AM","AN"),
        disp_B=("BA","BB","BC"),
        rot_from=None, rot_to=None,   # disp_B - disp_A 사용
        rot_to_m=rot_to_m,
        label="Shoulder",
    )
