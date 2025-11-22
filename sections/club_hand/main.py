# sections/club_hand/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import io, re
from datetime import datetime

from .features import _1distance as dis
from .features import _2rot_ang as rot
from .features import _3TDD as tdd
from .features import _4rot_center as rc
from .features import _5summ as misc
from .features import _6_sequance as kseq
from .features import _7_47t as f47
from .features import _add as accel

# ── 세션 저장소 초기화 (마스터 병합용) ────────────────────────────────────────
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}   # {section_id: {"title": str, "tables": dict[str, DataFrame]}}

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

def _arr_to_letter_df(arr) -> pd.DataFrame:
    """numpy 2D 배열 -> A,B,C... 컬럼명의 DataFrame"""
    df = pd.DataFrame(arr)
    df.columns = _letters(df.shape[1])
    return df


def _clean_loc(s: object) -> object:
    if not isinstance(s, str):
        return s
    # 1) Pro/프로/Ama/아마 토큰 제거
    s = re.sub(r'\b(Pro|프로|Ama|아마)\b', '', s, flags=re.IGNORECASE)
    # 2) 남는 구분자/여백 정리 (하이픈/대시 양옆 공백 -> 단일 공백)
    s = re.sub(r'\s*[-–—]\s*', ' ', s)
    # 3) 중복 공백 제거 + 트림
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

# ── 유틸: 시트명 안전화 ─────────────────────────────────────────────────────
def _safe_sheet(name: str, used: set[str]) -> str:
    s = re.sub(r'[\\/\?\*\[\]\:\'"]', '', str(name)).strip()
    s = (s or "Sheet").replace(' ', '_')[:31]
    base, i = s, 1
    while s in used:
        suf = f"_{i}"
        s = (base[:31-len(suf)] if len(base) > 31-len(suf) else base) + suf
        i += 1
    used.add(s)
    return s

# ── 유틸: 섹션 → 단일 시트에 세로로 쌓아 쓰기 ────────────────────────────────
def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 빈 시트 한번 만들어 핸들 확보
    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    cur_row = 0
    for name, df in tables.items():
        # 제목
        ws.write(cur_row, 0, str(name), title_fmt)
        cur_row += 1

        # 본문
        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        # 헤더/숫자 포맷 + 너비
        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)
        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

        # 다음 표 사이 여백 2줄
        cur_row += n_rows + 1 + 2

# ── 유틸: 섹션 표 dict를 마스터에 등록 ───────────────────────────────────────
def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

# ─────────────────────────────────────────────────────────────
# ✅ 화면 하이라이트 유틸 (인덱스로 '라벨 열'만 색칠)
# ─────────────────────────────────────────────────────────────
def _norm_indices(n: int, idxs: list[int]) -> list[int]:
    """음수 인덱스(-1: 마지막 행 등) 허용 → 정규화"""
    out = []
    for i in idxs:
        j = n + i if i < 0 else i
        if 0 <= j < n:
            out.append(j)
    return sorted(set(out))

def _style_highlight_rows_by_index(
    df: pd.DataFrame,
    row_indices: list[int],
    target_cols: list[str] | tuple[str, ...] = (),
    color: str = "#A9D08E",
) -> pd.io.formats.style.Styler:
    """
    row_indices: 0-based 인덱스 리스트.
    target_cols: 색칠할 '라벨 열'만 지정. 비우면 첫 번째 열을 라벨로 간주.
    """
    if not row_indices:
        return df.style
    if not target_cols:
        target_cols = (df.columns[0],)
    elif isinstance(target_cols, str):
        target_cols = (target_cols,)
    target_cols = [c for c in target_cols if c in df.columns]
    if not target_cols:
        target_cols = (df.columns[0],)

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    n = len(df)
    for idx in row_indices:
        if 0 <= idx < n:
            for c in target_cols:
                styles.iat[idx, df.columns.get_loc(c)] = f"background-color: {color}"
    return df.style.apply(lambda _df: styles, axis=None)

def _style_with_key(table_key: str, df: pd.DataFrame, fmt: dict | None = None, color: str = "#A9D08E"):
    label_col, idxs = CH_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    sty = _style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color)
    if fmt:
        sty = sty.format(fmt)
    return sty

# ─────────────────────────────────────────────────────────────
# ✅ Club & Hand 표별 인덱스 / 라벨 열 매핑
# (label_col을 ""로 두면 첫 열을 라벨로 자동 지정)
# ─────────────────────────────────────────────────────────────
IDX_BASIC     = [0,1,2,3]
IDX_LEFT      = []
IDX_CLUB      = []
IDX_KNEE_TDD  = []
IDX_KNEE_ROT  = []
IDX_PELVIS_TDD= []
IDX_HIP_ROT   = []
IDX_SHO_TDD   = []
IDX_SHO_ROT   = []
IDX_PELVIS_C  = [0,1,2,3]
IDX_SHO_C     = [0,1,2,3]
IDX_KNEE_C    = [0,1,2,3]
IDX_SUMMARY   = []

CH_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "클럽헤드/손 운동량과 힘": ("", IDX_BASIC),
    "왼팔 수평/수직 회전각도": ("", IDX_LEFT),
    "클럽 수평/수직 회전각도": ("", IDX_CLUB),
    "손/클럽 프레임별 가속도": ("", []),
    "4–7 구간 힘/토크 요약": ("", []),
    "4–7 구간 힘/토크 (프레임별)": ("", []),
    "키네마틱 시퀀스": ("", []),
    "무릎 TDD": ("", IDX_KNEE_TDD),
    "무릎 수평/수직 회전각도": ("", IDX_KNEE_ROT),
    "골반 TDD": ("", IDX_PELVIS_TDD),
    "골반 수평/수직 회전각도": ("", IDX_HIP_ROT),
    "어깨 TDD": ("", IDX_SHO_TDD),
    "어깨 수평/수직 회전각도": ("", IDX_SHO_ROT),
    "골반 회전 중심": ("", IDX_PELVIS_C),
    "어깨 회전 중심": ("", IDX_SHO_C),
    "무릎 회전 중심": ("", IDX_KNEE_C),
    "통합표": ("", IDX_SUMMARY),
}

META = {"id": "club_hand", "title": "11. Club & Hand", "icon": "🤝", "order": 41}
def get_metadata(): return META

# ─────────────────────────────────────────────────────────────
# ✅ 프로 vs 아마 Top3 (부호 같음 / 부호 다름) by '비율차'
#    - 비율차 = |P-A| / max(|P|, |A|)
#    - 세로형(열쌍) + 가로형(프로/일반 행 × 프레임숫자열) 모두 지원
# ─────────────────────────────────────────────────────────────
_PAIR_RULES = (("프로","일반"), ("Pro","Ama"))

def _to_num(x):
    try: return float(x)
    except Exception: return np.nan

def _ratio_diff(p: float, a: float) -> float:
    denom = max(abs(p), abs(a))
    if denom <= 0:
        return 0.0
    return abs(p - a) / denom

def _collect_pairs_vertical(df: pd.DataFrame, table_name: str) -> list[dict]:
    out: list[dict] = []
    if df is None or df.empty:
        return out

    headers = list(map(str, df.columns))
    label_col = df.columns[0] if len(df.columns) else None

    for a, b in _PAIR_RULES:
        for h in headers:
            if a in h:
                h_ama = h.replace(a, b)
                if h_ama in headers:
                    pvals = pd.to_numeric(df[h], errors="coerce")
                    avals = pd.to_numeric(df[h_ama], errors="coerce")
                    for idx in df.index:
                        p, av = pvals.loc[idx], avals.loc[idx]
                        if not (np.isfinite(p) and np.isfinite(av)): continue
                        ratio = _ratio_diff(p, av)
                        sign_same = (p * av) >= 0
                        row_label = str(df.iloc[idx, 0]) if label_col is not None else str(idx)
                        out.append({
                            "표": table_name,
                            "항목/라벨": row_label,
                            "위치": h,
                            "Pro": float(p),
                            "Ama": float(av),
                            "비율차": float(ratio),
                            "부호": "같음" if sign_same else "다름",
                        })
    return out

def _collect_pairs_horizontal(df: pd.DataFrame, table_name: str) -> list[dict]:
    out: list[dict] = []
    if df is None or df.empty:
        return out

    label_col = next((c for c in ["구분","항목"] if c in df.columns), None)
    if not label_col:
        return out
    frame_cols = [c for c in df.columns if c != label_col and str(c).isdigit()]
    if not frame_cols:
        return out

    def _norm_role(x: object) -> str | None:
        s = "" if x is None else str(x).strip()
        parts = re.split(r"\s*[·\.\|\-:]\s*", s.replace(" ",""))
        cand = (parts[-1] if parts else s).lower()
        if cand.startswith("pro") or cand in ("프로","pro"): return "프로"
        if cand.startswith("ama") or cand in ("일반","ama"): return "일반"
        return None

    r_pro = r_ama = None
    for ridx, v in df[label_col].items():
        role = _norm_role(v)
        if role == "프로" and r_pro is None: r_pro = int(ridx)
        if role == "일반" and r_ama is None: r_ama = int(ridx)
    if r_pro is None or r_ama is None:
        return out

    for c in frame_cols:
        p = _to_num(df.at[r_pro, c])
        a = _to_num(df.at[r_ama, c])
        if not (np.isfinite(p) and np.isfinite(a)): continue
        ratio = _ratio_diff(p, a)
        sign_same = (p * a) >= 0
        out.append({
            "표": table_name,
            "항목/라벨": str(label_col),
            "위치": f"프레임 {c}",
            "Pro": float(p),
            "Ama": float(a),
            "비율차": float(ratio),
            "부호": "같음" if sign_same else "다름",
        })
    return out

def top3_split_by_sign_ratio(df: pd.DataFrame, table_name: str) -> tuple[list[dict], list[dict]]:
    rows = []
    rows += _collect_pairs_vertical(df, table_name)
    rows += _collect_pairs_horizontal(df, table_name)

    same = [r for r in rows if r["부호"] == "같음"]
    opp  = [r for r in rows if r["부호"] == "다름"]

    same.sort(key=lambda r: r["비율차"], reverse=True)
    opp.sort(key=lambda r: r["비율차"], reverse=True)
    return same[:3], opp[:3]

# ─────────────────────────────────────────────────────────────

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    # 🔝 섹션 상단 Top3 박스(좌: 부호 같음 / 우: 부호 다름)
    top_box = st.container()
    col_same, col_opp = top_box.columns(2)

    pro_arr = ctx.get("pro_arr")
    ama_arr = ctx.get("ama_arr")

    # NEW: 원자료(프로/아마) DataFrame (app.py에서 ctx로 전달됨)
    gears_pro_df = ctx.get("gears_pro_df")
    gears_ama_df = ctx.get("gears_ama_df")

    if pro_arr is None or ama_arr is None:
        st.warning("무지개(기존) 엑셀 두 개(프로/일반)가 필요합니다.")
        return

    # ── 표 생성 ─────────────────────────────────────────────────────────────
    df_basic = dis.build_club_hand_table(pro_arr, ama_arr, pro_label="Pro", ama_label="Ama")
    st.dataframe(
        _style_with_key(
            "클럽헤드/손 운동량과 힘",
            df_basic,
            fmt={
                "ADD→TOP 이동거리(m)": "{:.2f}",
                "ADD→TOP 평균속도(m/s)": "{:.2f}",
                "TOP→IMP 이동거리(m)": "{:.2f}",
                "TOP→IMP 평균속도(m/s)": "{:.2f}",
                "TOP→IMP 평균가속도(m/s²)": "{:.2f}",
                "임팩트 순간 힘(N)": "{:.2f}",
                "ADD→TOP 평균속도(m/s) 비율(로리=100)": "{:.2f}",
                "임팩트 순간 힘(N) 비율(로리=100)": "{:.2f}",
            },
        ),
        use_container_width=True
    )

    # ✅ 손/클럽 프레임별 가속도
    st.divider()
    st.subheader("손/클럽 프레임별 가속도")
    df_pro_base = _arr_to_letter_df(pro_arr)
    df_ama_base = _arr_to_letter_df(ama_arr)

    df_accel = accel.build_hand_club_accel_table(
        df_pro_base, df_ama_base,
        time_col="B",      # 시간열(B): ms 또는 s 자동 처리
        pro_label="Pro",
        ama_label="Ama",
    )

    st.dataframe(
        _style_with_key(
            "손/클럽 프레임별 가속도",
            df_accel,
            fmt={
                "손 가속도(m/s²) - Pro":   "{:.2f}",
                "손 가속도(m/s²) - Ama":   "{:.2f}",
                "클럽 가속도(m/s²) - Pro": "{:.2f}",
                "클럽 가속도(m/s²) - Ama": "{:.2f}",
            },
        ),
        use_container_width=True
    )

    # 왼팔/클럽 회전각
    st.divider()
    st.subheader("왼팔 회전각 (Left Arm)")
    df_left = rot.build_left_arm_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "왼팔 수평/수직 회전각도",
            df_left,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.subheader("클럽 회전각")
    df_club = rot.build_club_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "클럽 수평/수직 회전각도",
            df_club,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    # TDD, 회전각(무릎/골반/어깨)
    st.divider()
    st.subheader("무릎 TDD")
    df_knee = tdd.build_knee_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(_style_with_key("무릎 TDD", df_knee), use_container_width=True)

    st.divider()
    st.markdown("무릎 수평 수직")
    df_knee_rot = rot.build_knee_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "무릎 수평/수직 회전각도",
            df_knee_rot,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )
    
    st.divider()
    st.markdown("골반 TDD")
    df_pelvis = tdd.build_hip_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(_style_with_key("골반 TDD", df_pelvis), use_container_width=True)

    st.divider()
    st.markdown("골반 수평 수직")
    df_hip_rot = rot.build_hip_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "골반 수평/수직 회전각도",
            df_hip_rot,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.markdown("어깨 TDD")
    df_shoulder = tdd.build_shoulder_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(_style_with_key("어깨 TDD", df_shoulder), use_container_width=True)

    st.divider()
    st.markdown("어깨 수평 수직")
    df_sho_rot = rot.build_shoulder_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "어깨 수평/수직 회전각도",
            df_sho_rot,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    # 회전 중심
    st.divider()
    st.markdown("회전 중심")

    st.subheader("골반")
    df_p = rc.build_pelvis_center_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("골반 회전 중심", df_p), use_container_width=True)

    st.subheader("어깨")
    df_s = rc.build_shoulder_center_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("어깨 회전 중심", df_s), use_container_width=True)

    st.subheader("무릎")
    df_k = rc.build_knee_center_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("무릎 회전 중심", df_k), use_container_width=True)

    # 요약 표
    st.divider()
    st.subheader("회전 중심 구간차")
    df_center = misc.build_rotation_center_diff_all(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "통합표",
            df_center,
            fmt={"X 차이 (Ama - Pro)":"{:+.2f}","Y 차이 (Ama - Pro)":"{:+.2f}","Z 차이 (Ama - Pro)":"{:+.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.subheader("회전각 요약 (구간별: 1-4 / 4-7 / 7-10 / 합계)")
    df_rot_summary = rot.build_rotation_summary_all(pro_arr, ama_arr, pro_label="Pro", ama_label="Ama")
    st.dataframe(
        df_rot_summary.style.format({
            "Pro 수평회전각": "{:.2f}", "Ama 수평회전각": "{:.2f}",
            "Pro 수직회전각": "{:.2f}", "Ama 수직회전각": "{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("TDD 요약 (Knee / Pelvis / Shoulder, 구간별)")
    df_tdd_summary = tdd.build_tdd_summary_all(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(
        df_tdd_summary.style.format({
            "이동(Pro,m)": "{:.2f}", "이동(Ama,m)": "{:.2f}",
            "회전량(Pro,deg)": "{:.2f}", "회전량(Ama,deg)": "{:.2f}",
            "TDD(Pro,m)": "{:.2f}", "TDD(Ama,m)": "{:.2f}",
        }),
        use_container_width=True
    )

    # ─────────────────────────────────────────────────────────
    # ✅  키네마틱 / 키네틱 시퀀스 (원자료 gears_* 사용, 4×2 표)
    #     - 백/다운은 한 줄에 2개 컬럼으로 배치
    # ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("키네마틱 / 키네틱 시퀀스 (원자료 기반)")

    kin_kinetic_tables = {}  # 엑셀 내보내기용으로 모음

    if gears_pro_df is None or gears_ama_df is None:
        st.info("원자료(gears_raw_preprocessed.csv)를 프로/아마 각각 업로드하면 4×2 표가 표시됩니다.")
    else:
        tables_pairwise = kseq.build_kinematic_and_kinetic_tables_gears(
            gears_pro_df, gears_ama_df,
            pro_name="프로", ama_name="아마", handedness="right"
        )

        # 공통 포맷
        fmt = {"시각(s)": "{:.6f}", "값": "{:.2f}"}

        # ── (교체) 4×2 표 한 줄에 두 개 붙여서 표시 ─────────────────────────
        def _row(title_left: str, title_right: str):
            # 간격 좁게
            c1, c2 = st.columns([1, 1], gap="small")
            fmt = {"시각(s)": "{:.6f}", "값": "{:.2f}"}

            with c1:
                st.markdown(f"**{title_left}**")
                dfL = tables_pairwise[title_left]
                # 표 자체는 좌측 정렬된 느낌을 주기 위해 여백 최소화 (container 폭은 표 폭에 맞춤)
                st.dataframe(dfL.style.format(fmt), use_container_width=True)
                kin_kinetic_tables[title_left] = dfL

            with c2:
                st.markdown(f"**{title_right}**")
                dfR = tables_pairwise[title_right]
                st.dataframe(dfR.style.format(fmt), use_container_width=True)
                kin_kinetic_tables[title_right] = dfR


        # 1행: 키네마틱 - 프로 (Back | Down)
        _row("키네마틱 - 프로 - Back", "키네마틱 - 프로 - Down")
        # 2행: 키네마틱 - 아마 (Back | Down)
        _row("키네마틱 - 아마 - Back", "키네마틱 - 아마 - Down")
        # 3행: 키네틱 - 프로 (Back | Down)
        _row("키네틱   - 프로 - Back", "키네틱   - 프로 - Down")
        # 4행: 키네틱 - 아마 (Back | Down)
        _row("키네틱   - 아마 - Back", "키네틱   - 아마 - Down")

    # ─────────────────────────────────────────────────────────
    # 🔝 섹션 상단: “부호 같음 Top3 / 부호 다름 Top3” (비율차 기준) 표시
    #   - 섹션 전체 표를 대상으로 선별
    #   - 원하면 포함/제외 조정 가능
    # ─────────────────────────────────────────────────────────
    candidate_for_top = {
        "클럽헤드/손 운동량과 힘": df_basic,
        "Hand & Club Average Acceleration(구간별 평균가속도)": df_accel,
        "왼팔 수평/수직 회전각도": df_left,
        "클럽 수평/수직 회전각도": df_club,
        "무릎 TDD": df_knee,
        "무릎 수평/수직 회전각도": df_knee_rot,
        "골반 TDD": df_pelvis,
        "골반 수평/수직 회전각도": df_hip_rot,
        "어깨 TDD": df_shoulder,
        "어깨 수평/수직 회전각도": df_sho_rot,
    }
    # 원자료 기반 표도 포함(있으면)
    if "키네마틱 - 프로 - Back" in (kin_kinetic_tables or {}):
        for k, v in kin_kinetic_tables.items():
            candidate_for_top[k] = v

    same_all: list[dict] = []
    opp_all:  list[dict] = []
    for name, df in candidate_for_top.items():
        try:
            same3, opp3 = top3_split_by_sign_ratio(df, name)
            same_all.extend(same3)
            opp_all.extend(opp3)
        except Exception:
            pass

    same_all.sort(key=lambda r: r["비율차"], reverse=True)
    opp_all.sort(key=lambda r: r["비율차"], reverse=True)
    same_top3 = same_all[:3]
    opp_top3  = opp_all[:3]

    # ── 부호 같음 Top3 표시 (비율차/부호 컬럼은 표시 제거)
    with col_same:
        st.markdown("### ⚖️ 부호 **같음** – 비율차 Top 3")
        if not same_top3:
            st.info("해당 없음")
        else:
            df_same = pd.DataFrame(same_top3)[["표","항목/라벨","위치","Pro","Ama"]].copy()
            df_same["위치"] = df_same["위치"].map(_clean_loc)
            st.dataframe(
                df_same.style.format({"Pro":"{:.2f}", "Ama":"{:.2f}"}),
                use_container_width=True
            )

    # ── 부호 다름 Top3 표시 (비율차/부호 컬럼은 표시 제거)
    with col_opp:
        st.markdown("### 🧲 부호 **다름** – 비율차 Top 3")
        if not opp_top3:
            st.info("해당 없음")
        else:
            df_opp = pd.DataFrame(opp_top3)[["표","항목/라벨","위치","Pro","Ama"]].copy()
            df_opp["위치"] = df_opp["위치"].map(_clean_loc)
            st.dataframe(
                df_opp.style.format({"Pro":"{:.2f}", "Ama":"{:.2f}"}),
                use_container_width=True
            )



    # ─────────────────────────────────────────────────────────
    # ✅ 4–7 구간 힘/토크 (요약 & 프레임별)
    # ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("회전, 수직, 직선력")

    df_pro_base = _arr_to_letter_df(pro_arr)
    df_ama_base = _arr_to_letter_df(ama_arr)

    res47 = f47.build_47_forces_and_torque(
        df_pro_base, df_ama_base,
        mass=float(ctx.get("mass", 60.0)),
        pro_label="Pro", ama_label="Ama",
    )

    st.markdown("**요약 (평균±표준편차 / 비율)**")
    st.dataframe(
        _style_with_key("4–7 구간 힘/토크 요약", res47.table_summary),
        use_container_width=True
    )

    st.markdown("**프레임별 값**")
    st.dataframe(
        _style_with_key(
            "4–7 구간 힘/토크 (프레임별)",
            res47.table_perframe,
            fmt={
                "토크|τ|(N·m)": "{:.2f}",
                "회전력 F_rot(N)": "{:.2f}",
                "Y등가힘 F_y(N)": "{:.2f}",
                "Z등가힘 F_z(N)": "{:.2f}",
            },
        ),
        use_container_width=True
    )

    # ── 단일 시트 엑셀 다운로드 + 마스터 등록 ────────────────────────────────
    tables = {
        "클럽헤드/손 운동량과 힘": df_basic,
        "Hand & Club Average Acceleration(구간별 평균가속도)": df_accel,
        "왼팔 수평/수직 회전각도":            df_left,
        "클럽 수평/수직 회전각도":                df_club,
        "무릎 TDD":                     df_knee,
        "무릎 수평/수직 회전각도":          df_knee_rot,
        "골반 TDD":                   df_pelvis,
        "골반 수평/수직 회전각도":        df_hip_rot,
        "어깨 TDD":                 df_shoulder,
        "어깨 수평/수직 회전각도":      df_sho_rot,
        "골반 회전 중심":                df_p,
        "어깨 회전 중심":              df_s,
        "무릎 회전 중심":                  df_k,
        "통합표":      df_center,
        "회전각 요약(구간별)": df_rot_summary,
        "TDD 요약(구간별)": df_tdd_summary,
        **kin_kinetic_tables,  # 원자료 기반 4×2 표도 포함
        "회전, 수직, 직선력 요약": res47.table_summary,
        "회전, 수직, 직선력 (프레임별)": res47.table_perframe,
    }

    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        _write_section_sheet(writer, sheet_name="All", tables=tables)
    xbuf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Club & Hand (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"club_hand_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_club_hand_all"
    )

    if st.button("➕ 이 섹션을 마스터 엑셀에 추가", use_container_width=True, key="reg_club_hand_master"):
        register_section(META["id"], META["title"], tables)
        st.success("Club & Hand 섹션을 마스터 엑셀에 등록했습니다. (사이드바/메인에서 '모든 섹션 합쳐서 다운로드' 버튼으로 병합 파일을 받을 수 있어요.)")
