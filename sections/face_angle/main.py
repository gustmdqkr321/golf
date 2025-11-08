# sections/face_angle/main.py
from __future__ import annotations
import streamlit as st
from .features import _1basic as feat
from .features import _2rolling as roll
from .features import _3dcocking as ck3
from .features import _4cocking2d as ck2
from .features import _5hinge as hinge
from .features import _6boncu as bc
from .features import _7tilt as tilt
from .features import _8foream as aux

import io, re
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# 마스터 합본용 세션 저장소
# {section_id: {"title": str, "tables": dict[str, pd.DataFrame]}}
# ─────────────────────────────────────────────────────────────────────
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}

def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

# 시트명 안전화 (엑셀 금지문자 제거, 31자 제한, 중복 처리)
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

# 섹션 → 엑셀 단일 시트로 쓰기(섹션 내부 표 dict를 세로로 쌓음)
def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 빈 시트를 먼저 한 번 써서 핸들 확보
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

        # 다음 표 시작 위치로 이동: 본문 n_rows + 헤더 1 + 여백 2
        cur_row += n_rows + 1 + 2

# ─────────────────────────────────────────────────────────────
# ✅ 화면 하이라이트 유틸 (인덱스로 라벨 열만 색칠)
# ─────────────────────────────────────────────────────────────
def _norm_indices(n: int, idxs: list[int]) -> list[int]:
    """음수 인덱스 허용(-1: 마지막 행 등) → 정규화"""
    out = []
    for i in idxs:
        j = n + i if i < 0 else i
        if 0 <= j < n:
            out.append(j)
    return sorted(set(out))

def _style_highlight_rows_by_index(df: pd.DataFrame,
                                   row_indices: list[int],
                                   target_cols: list[str] | tuple[str, ...] = (),
                                   color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    """
    row_indices: 0-based 인덱스 리스트. 빈 리스트면 원본 스타일.
    target_cols: 색칠할 '라벨 열'만 지정. 비우면 첫 번째 열을 자동 선택.
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

def _apply_2f(styler: pd.io.formats.style.Styler, df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """숫자열만 소수 둘째 자리 포맷"""
    fmt = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return styler.format(fmt)

# ─────────────────────────────────────────────────────────────
# ✅ Face Angle 표별 인덱스 / 라벨 열 매핑
# (라벨 열을 모르면 빈 문자열 ""로 두면 첫 열을 자동 라벨로 칠함)
# ─────────────────────────────────────────────────────────────
IDX_FACE_BASIC  = []
IDX_ROLL        = [9,10,12,13,15]
IDX_COCK3D      = [0,3,4,5,10,11,12,13]
IDX_COCK2D      = []
IDX_HINGE       = [11,13]
IDX_TILT        = [3,4,5,6,9]
IDX_BOWCUP     = [10,11,13]  
IDX_OPEN_CLOSE  = [3,4,5,6]
IDX_SUP1        = []
IDX_SUP2        = [3,4,5,6,7]

FA_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "1_Basic Data": ("", IDX_FACE_BASIC),                 # 예: "항목" 또는 "검사명"이면 그 이름으로 바꿔도 됨
    "2.Wrist Rolling Angle": ("", IDX_ROLL),
    "3_3D_Cocking": ("", IDX_COCK3D),
    "4_2D_Cocking": ("", IDX_COCK2D),
    "5_Hinging": ("", IDX_HINGE),
    "6_Bowing_Cupping": ("", IDX_BOWCUP), 
    "7_Clubface : open/close(Heel/Toe Tilt) ": ("", IDX_TILT),
    "8_Club_OpenClose": ("", IDX_OPEN_CLOSE),
    "9_Forearm_Supination_1": ("", IDX_SUP1),
    "10_Forearm_Supination_2": ("", IDX_SUP2),
}

def _style_with_key(table_key: str, df: pd.DataFrame, color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    label_col, idxs = FA_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    return _apply_2f(_style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color), df)

# ─────────────────────────────────────────────────────────────
META = {"id": "face_angle", "title": "7. Face Angle", "icon": "🎯", "order": 19}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    base_pro = ctx.get("pro_arr")
    base_ama = ctx.get("ama_arr")
    gs_pro   = ctx.get("gs_pro_arr")   # DataFrame
    gs_ama   = ctx.get("gs_ama_arr")   # DataFrame

    if gs_pro is None or gs_ama is None:
        st.warning("GS CSV(프로/일반)를 업로드하거나 app.py에 디폴트 경로를 설정하세요.")
        return
    if base_pro is None or base_ama is None:
        st.warning("무지개(기존) 엑셀 파일도 필요합니다.")
        return

    # ─────────────────────────────────────────────────────────────
    # 1) 표 생성
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}

    # 1. Face Angle 기본 표 (셀/식 숨김 버전)
    df_face = feat.build_face_angle_table(gs_pro, gs_ama, base_pro, base_ama)
    st.subheader("Face Angle (Summary)")
    st.dataframe(_style_with_key("1_Basic Data", df_face), use_container_width=True)
    tables["1_Basic Data"] = df_face

    # 2. Rolling
    st.divider(); st.subheader("Rolling")
    df_roll = roll.build_rolling_summary_table(base_pro, base_ama, alpha=2.0)
    st.dataframe(_style_with_key("2.Wrist Rolling Angle", df_roll), use_container_width=True)
    tables["2.Wrist Rolling Angle"] = df_roll

    # 3. 3D Cocking
    st.divider(); st.subheader("3D Cocking")
    df_ck3 = ck3.compute_cocking_table_from_arrays(base_pro, base_ama)
    st.dataframe(_style_with_key("3_3D_Cocking", df_ck3), use_container_width=True)
    tables["3_3D_Cocking"] = df_ck3

    # 4. 2D Cocking
    st.divider(); st.subheader("2D Cocking")
    df_ck2 = ck2.build_yz_plane_compare_table(base_pro, base_ama)
    st.dataframe(_style_with_key("4_2D_Cocking", df_ck2), use_container_width=True)
    tables["4_2D_Cocking"] = df_ck2

    # 5. Hinging
    st.divider(); st.subheader("Hinging")
    df_hinge = hinge.build_hinging_compare_table(base_pro, base_ama, alpha=2.0)
    st.dataframe(_style_with_key("5_Hinging", df_hinge), use_container_width=True)
    tables["5_Hinging"] = df_hinge

    # 6. Bowing/Cupping
    st.divider(); st.subheader("Bowing / Cupping")
    df_bc = bc.build_bowing_table_from_arrays(base_pro, base_ama)
    st.dataframe(_style_with_key("6_Bowing_Cupping", df_bc), use_container_width=True)
    tables["6_Bowing_Cupping"] = df_bc

    # 7. Tilt
    st.divider(); st.subheader("Tilt")
    df_tilt = tilt.build_tilt_compare_table(base_pro, base_ama)
    st.dataframe(_style_with_key("7_Clubface : open/close(Heel/Toe Tilt) ", df_tilt), use_container_width=True)
    tables["7_Clubface : open/close(Heel/Toe Tilt) "] = df_tilt

    # 8. CLUB: (-) CLOSE, (+) OPEN
    st.divider(); st.subheader("CLUB  : (-): CLOSE, (+) : OPEN")
    df_club = aux.build_tilt_numerators_table(base_pro, base_ama)
    st.dataframe(_style_with_key("8_Club_OpenClose", df_club), use_container_width=True)
    tables["8_Club_OpenClose"] = df_club

    # 9. Forearm Supination 1
    st.divider(); st.subheader("Forearm Supination 1")
    df_sup1 = aux.build_ay_bn_diffs_table(base_pro, base_ama)
    st.dataframe(_style_with_key("9_Forearm_Supination_1", df_sup1), use_container_width=True)
    tables["9_Forearm_Supination_1"] = df_sup1

    # 10. Forearm Supination 2
    st.divider(); st.subheader("Forearm Supination 2")
    df_sup2 = aux.build_abc_angles_table(base_pro, base_ama)
    st.dataframe(_style_with_key("10_Forearm_Supination_2", df_sup2), use_container_width=True)
    tables["10_Forearm_Supination_2"] = df_sup2

    # ─────────────────────────────────────────────────────────────
    # 2) 섹션 단일 시트 엑셀 다운로드 + 마스터 합본 등록
    # ─────────────────────────────────────────────────────────────
    # 단일 시트 엑셀
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        sheet_name = "FaceAngle"
        _write_section_sheet(writer, sheet_name, tables)
    xbuf.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Face Angle (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"face_angle_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # 마스터 합본에 등록 (app.py의 “마스터 합본 다운로드”에서 한 번에 합쳐짐)
    register_section(META["id"], META["title"], tables)
