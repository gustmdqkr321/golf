# sections/club_hand/features/_8_kseq.py
from __future__ import annotations
import numpy as np
import pandas as pd
from . import _3TDD as tdd

_BACK_SEGS = ["1-2", "2-3", "3-4"]
_DOWN_SEGS = ["4-5", "5-6", "6-7"]

def _seg_rank_key(seg: str) -> int:
    if seg in _BACK_SEGS: return _BACK_SEGS.index(seg)
    if seg in _DOWN_SEGS: return _DOWN_SEGS.index(seg)
    return 99

def _pick_peak(seg_series: pd.Series, val_series: pd.Series, seg_pool: list[str]) -> tuple[str, float]:
    df = pd.DataFrame({"seg": seg_series.values,
                       "val": pd.to_numeric(val_series.values, errors="coerce")})
    df = df[df["seg"].isin(seg_pool)].dropna(subset=["val"])
    if df.empty:
        return ("-", float("nan"))
    df = df.sort_values(["val"], ascending=[False])
    top_val = df.iloc[0]["val"]
    tops = df[df["val"] == top_val].copy()
    tops["rk"] = tops["seg"].map(_seg_rank_key)
    tops = tops.sort_values("rk")
    row = tops.iloc[0]
    return (str(row["seg"]), float(row["val"]))

def _peaks_from_tdd(df_part: pd.DataFrame) -> tuple[tuple[str,float], tuple[str,float]]:
    seg = df_part["구간"]
    pro = df_part["TDD(Pro,m)"]
    ama = df_part["TDD(Ama,m)"]
    b_seg_p, b_val_p = _pick_peak(seg, pro, _BACK_SEGS)
    d_seg_p, d_val_p = _pick_peak(seg, pro, _DOWN_SEGS)
    b_seg_a, b_val_a = _pick_peak(seg, ama, _BACK_SEGS)
    d_seg_a, d_val_a = _pick_peak(seg, ama, _DOWN_SEGS)
    return (b_seg_p, b_val_p, d_seg_p, d_val_p), (b_seg_a, b_val_a, d_seg_a, d_val_a)

def _group_seq_text(items: list[tuple[str,str,float]], order: list[str]) -> str:
    """
    items: [(PartName, seg, value), ...]
    같은 구간은 값 내림차순으로 정렬해 쉼표로 묶고,
    구간 사이를 ' ➟ '로 연결.
    """
    by_seg: dict[str, list[tuple[str,float]]] = {s: [] for s in order}
    for name, seg, val in items:
        if seg in by_seg:
            by_seg[seg].append((name, float(val) if np.isfinite(val) else -np.inf))
    chunks = []
    for seg in order:
        if not by_seg[seg]:
            continue
        # 같은 구간 안에서는 값 내림차순
        group_sorted = sorted(by_seg[seg], key=lambda x: (-x[1], x[0]))
        chunks.append(", ".join([g[0].lower() for g in group_sorted]))
    return " ➟ ".join(chunks) if chunks else "-"

def build_kinematic_sequence_table(
    base_pro: np.ndarray,
    base_ama: np.ndarray,
    df_accel: pd.DataFrame,
    *,
    pro_name: str = "프로",
    ama_name: str = "아마",
    rot_to_m: float = 0.01,
) -> pd.DataFrame:
    """
    반환 컬럼(평면 헤더, 엑셀/스타일러 호환):
      ['항목',
       f'({pro_name}) Back 구간', f'({pro_name}) Back 값',
       f'({pro_name}) Down 구간', f'({pro_name}) Down 값',
       f'({ama_name}) Back 구간', f'({ama_name}) Back 값',
       f'({ama_name}) Down 구간', f'({ama_name}) Down 값']
    마지막 행 'Kinematic Sequence'는 Back/Down 칸에 'knee, pelvis ➟ …' 형식 문자열을 채워줍니다.
    """
    # 1) TDD 테이블(무릎/골반/어깨)
    knee = tdd.build_knee_tdd_table(base_pro, base_ama, rot_to_m=rot_to_m)
    pelvis = tdd.build_hip_tdd_table(base_pro, base_ama, rot_to_m=rot_to_m)
    shoulder = tdd.build_shoulder_tdd_table(base_pro, base_ama, rot_to_m=rot_to_m)

    pk_k_p, pk_k_a = _peaks_from_tdd(knee)
    pk_p_p, pk_p_a = _peaks_from_tdd(pelvis)
    pk_s_p, pk_s_a = _peaks_from_tdd(shoulder)

    # 2) 손/클럽 가속도 피크
    segA = df_accel["구간"]
    def _peak_from_accel(col):
        b = _pick_peak(segA, df_accel[col], _BACK_SEGS)
        d = _pick_peak(segA, df_accel[col], _DOWN_SEGS)
        return b + d  # (b_seg, b_val, d_seg, d_val)

    col_hand_pro = next(c for c in df_accel.columns if "손 가속도" in c and ("Pro" in c or "로리" in c))
    col_hand_ama = next(c for c in df_accel.columns if "손 가속도" in c and ("Ama" in c or "홍" in c))
    col_club_pro = next(c for c in df_accel.columns if "클럽 가속도" in c and ("Pro" in c or "로리" in c))
    col_club_ama = next(c for c in df_accel.columns if "클럽 가속도" in c and ("Ama" in c or "홍" in c))

    pk_h_p = _peak_from_accel(col_hand_pro)
    pk_h_a = _peak_from_accel(col_hand_ama)
    pk_c_p = _peak_from_accel(col_club_pro)
    pk_c_a = _peak_from_accel(col_club_ama)

    # 3) 표 본문(부위별 한 줄)
    rows = []
    def _append_row(name, pks_p, pks_a):
        b_seg_p, b_val_p, d_seg_p, d_val_p = pks_p
        b_seg_a, b_val_a, d_seg_a, d_val_a = pks_a
        rows.append([
            name,
            b_seg_p, round(b_val_p, 2), d_seg_p, round(d_val_p, 2),
            b_seg_a, round(b_val_a, 2), d_seg_a, round(d_val_a, 2),
        ])

    _append_row("Knee",     pk_k_p, pk_k_a)
    _append_row("Pelvis",   pk_p_p, pk_p_a)
    _append_row("Shoulder", pk_s_p, pk_s_a)
    _append_row("Hand",     pk_h_p, pk_h_a)
    _append_row("Club",     pk_c_p, pk_c_a)

    # 4) 시퀀스 문자열(화살표, 같은 구간은 콤마로 묶기)
    back_pro_items = [(r[0], r[1], r[2]) for r in rows]  # (name, seg, val)
    down_pro_items = [(r[0], r[3], r[4]) for r in rows]
    back_ama_items = [(r[0], r[5], r[6]) for r in rows]
    down_ama_items = [(r[0], r[7], r[8]) for r in rows]

    seq_back_pro = _group_seq_text(back_pro_items, _BACK_SEGS)
    seq_down_pro = _group_seq_text(down_pro_items, _DOWN_SEGS)
    seq_back_ama = _group_seq_text(back_ama_items, _BACK_SEGS)
    seq_down_ama = _group_seq_text(down_ama_items, _DOWN_SEGS)

    rows.append([
        "Kinematic Sequence",
        seq_back_pro, "",  # 프로 Back 칸에 시퀀스, 값칸 비움
        seq_down_pro, "",
        seq_back_ama, "",
        seq_down_ama, "",
    ])

    cols = [
        "항목",
        f"({pro_name}) Back 구간", f"({pro_name}) Back 값",
        f"({pro_name}) Down 구간", f"({pro_name}) Down 값",
        f"({ama_name}) Back 구간", f"({ama_name}) Back 값",
        f"({ama_name}) Down 구간", f"({ama_name}) Down 값",
    ]
    return pd.DataFrame(rows, columns=cols)
