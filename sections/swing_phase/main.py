# sections/swing_phase/main.py
from __future__ import annotations
import streamlit as st
from .features import _1take_back as feat
from .features import _2half as half
from .features import _3t214 as t214
from .features import _4transition as trans 
from .features import _5downswing as down
from .features import _6impact as imp
from .features import _7impact2 as imp2
from .features import _8t218 as t218
from .features import _9follow1 as fol1
from .features import _10follow2 as fol2

import io, re
import pandas as pd
from datetime import datetime  # ✅ Datetime 오타 수정

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
    row_indices: 0-based 인덱스 리스트. 빈 리스트면 원본 스타일 유지.
    target_cols: 색칠할 '라벨 열'만 지정. 비우면 첫 번째 열을 자동 라벨로 칠함.
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
# ✅ Swing Phase 표별 인덱스 / 라벨 열 매핑
# (라벨 열을 모르면 ""로 두면 첫 열을 자동 라벨로 칠함)
# 필요 시 인덱스 리스트를 너 기준으로 바꿔 써!
# ─────────────────────────────────────────────────────────────
IDX_TAKE_BACK   = [0,1,4,5,11,12]          # 예시
IDX_HALF        = []          # 예시
IDX_TOP         = [0,3,4,5,6,7,9,11,12,14,16,19]      # 예시(마지막 행 포함)
IDX_TRANS       = [1,5,9,11,12,14,18,22,23]          # 예시
IDX_DOWN        = [0,1,5,6,7,10,11,12,13,20,21]       # 예시
IDX_IMPACT      = [0,1,2,3,7,11,12,13]          # 예시
IDX_IMP_TBS     = []          # 예시 (5.10 표)
IDX_IMP_ADDIMP  = [9,12,14,15,16,17,18,22,23,24,25,30,32,33]       # 예시 (2.1.8 표)
IDX_FOLLOW1     = [1]          # 예시
IDX_FOLLOW2     = [7,11]       # 예시

SP_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "2_1_2_Take_Back":              ("", IDX_TAKE_BACK),
    "2_1_3_Half_Swing":             ("", IDX_HALF),
    "2_1_4_Top":                    ("", IDX_TOP),
    "2_1_5_Transition":             ("", IDX_TRANS),
    "2_1_6_Downswing":              ("", IDX_DOWN),
    "2_1_7_Impact":                 ("", IDX_IMPACT),
    "5_10_Impact_Turn_Bend_SideBend": ("", IDX_IMP_TBS),
    "2_1_8_Imp_AddImp":             ("", IDX_IMP_ADDIMP),
    "2_1_9_Follow1":                ("", IDX_FOLLOW1),
    "2_1_10_Follow2":               ("", IDX_FOLLOW2),
}

def _style_with_key(table_key: str, df: pd.DataFrame, color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    label_col, idxs = SP_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    return _apply_2f(_style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color), df)

# ─────────────────────────────────────────────────────────────
META = {"id": "swing_phase", "title": "8. Swing Phase", "icon": "🏌️‍♂️", "order": 28}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    pro_arr = ctx.get("pro_arr")
    ama_arr = ctx.get("ama_arr")
    gs_pro  = ctx.get("gs_pro_arr")  # DataFrame
    gs_ama  = ctx.get("gs_ama_arr")  # DataFrame

    if pro_arr is None or ama_arr is None:
        st.warning("무지개(프로/일반) 엑셀이 필요합니다.")
        return

    # imp2, t218 은 GS 사용
    if gs_pro is None or gs_ama is None:
        st.warning("일부 표(Impact Turn/Bend/Side Bend, Imp & Add/Imp)는 GS CSV가 필요합니다.")
        # 계속 진행은 함 (GS 필요한 표는 건너뜀)

    # ─────────────────────────────────────────────────────────────
    # 1) 표 생성 및 화면 표시 (✅ 하이라이트 적용)
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}

    # 2.1.2 Take Back
    st.subheader("2.1.2 Take Back")
    df_tb = feat.build_swing_phase_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_2_Take_Back", df_tb), use_container_width=True)
    tables["2_1_2_Take_Back"] = df_tb

    # 2.1.3 Half Swing
    st.divider(); st.subheader("2.1.3 Half Swing")
    df_half = half.build_swing_phase_table_v2(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_3_Half_Swing", df_half), use_container_width=True)
    tables["2_1_3_Half_Swing"] = df_half

    # 2.1.4 Top
    st.divider(); st.subheader("2.1.4 Top")
    df_top = t214.build_quarter_phase_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_4_Top", df_top), use_container_width=True)
    tables["2_1_4_Top"] = df_top

    # 2.1.5 Transition
    st.divider(); st.subheader("2.1.5 Transition")
    df_q5 = trans.build_quarter5_phase_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_5_Transition", df_q5), use_container_width=True)
    tables["2_1_5_Transition"] = df_q5

    # 2.1.6 Downswing
    st.divider(); st.subheader("2.1.6 Downswing")
    df_q6 = down.build_quarter6_phase_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_6_Downswing", df_q6), use_container_width=True)
    tables["2_1_6_Downswing"] = df_q6

    # 2.1.7 Impact
    st.divider(); st.subheader("2.1.7 Impact")
    df_q7 = imp.build_quarter7_impact_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_7_Impact", df_q7), use_container_width=True)
    tables["2_1_7_Impact"] = df_q7

    # 5.10 Impact : Turn, Bend, Side Bend (GS 필요)
    if gs_pro is not None and gs_ama is not None:
        st.divider(); st.subheader("5.10 Impact : Turn, Bend, Side Bend")
        df_tb_s = imp2.build_turn_bend_table(gs_pro, gs_ama)
        st.dataframe(_style_with_key("5_10_Impact_Turn_Bend_SideBend", df_tb_s), use_container_width=True)
        tables["5_10_Impact_Turn_Bend_SideBend"] = df_tb_s

    # 2.1.8 Imp & Add/Imp (GS + Base)
    if gs_pro is not None and gs_ama is not None:
        st.divider(); st.subheader("2.1.8 Imp & Add/Imp")
        df_sum = t218.build_summary_phase_table(gs_pro, gs_ama, pro_arr, ama_arr)
        st.dataframe(_style_with_key("2_1_8_Imp_AddImp", df_sum), use_container_width=True)
        tables["2_1_8_Imp_AddImp"] = df_sum

    # 2.1.9 Follow1
    st.divider(); st.subheader("2.1.9 Follow1")
    df_q8 = fol1.build_quarter8_phase_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_9_Follow1", df_q8), use_container_width=True)
    tables["2_1_9_Follow1"] = df_q8

    # 2.1.10 Follow2
    st.divider(); st.subheader("2.1.10 Follow2")
    df_q9q10 = fol2.build_quarter9_10_phase_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("2_1_10_Follow2", df_q9q10), use_container_width=True)
    tables["2_1_10_Follow2"] = df_q9q10

    # ─────────────────────────────────────────────────────────────
    # 2) 섹션 단일 시트 엑셀 다운로드 + 마스터 합본 등록
    # ─────────────────────────────────────────────────────────────
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        sheet_name = "SwingPhase"
        _write_section_sheet(writer, sheet_name, tables)
    xbuf.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Swing Phase (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"swing_phase_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # 마스터 합본에 등록 (app.py의 “마스터 합본 다운로드”에서 한 파일로 합치기)
    register_section(META["id"], META["title"], tables)
