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
# {section_id: {"title": str, "tables": dict[str, DataFrame]}}
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
    # 1) 표 생성 및 화면 표시
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}

    # 2.1.2 Take Back
    st.subheader("2.1.2 Take Back")
    df_tb = feat.build_swing_phase_table(pro_arr, ama_arr)
    st.dataframe(df_tb.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_2_Take_Back"] = df_tb

    # 2.1.3 Half Swing
    st.divider(); st.subheader("2.1.3 Half Swing")
    df_half = half.build_swing_phase_table_v2(pro_arr, ama_arr)
    st.dataframe(df_half.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_3_Half_Swing"] = df_half

    # 2.1.4 Top
    st.divider(); st.subheader("2.1.4 Top")
    df_top = t214.build_quarter_phase_table(pro_arr, ama_arr)
    st.dataframe(df_top.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_4_Top"] = df_top

    # 2.1.5 Transition
    st.divider(); st.subheader("2.1.5 Transition")
    df_q5 = trans.build_quarter5_phase_table(pro_arr, ama_arr)
    st.dataframe(df_q5.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_5_Transition"] = df_q5

    # 2.1.6 Downswing
    st.divider(); st.subheader("2.1.6 Downswing")
    df_q6 = down.build_quarter6_phase_table(pro_arr, ama_arr)
    st.dataframe(df_q6.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_6_Downswing"] = df_q6

    # 2.1.7 Impact
    st.divider(); st.subheader("2.1.7 Impact")
    df_q7 = imp.build_quarter7_impact_table(pro_arr, ama_arr)
    st.dataframe(df_q7.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_7_Impact"] = df_q7

    # 5.10 Impact : Turn, Bend, Side Bend (GS 필요)
    if gs_pro is not None and gs_ama is not None:
        st.divider(); st.subheader("5.10 Impact : Turn, Bend, Side Bend")
        df_tb_s = imp2.build_turn_bend_table(gs_pro, gs_ama)
        st.dataframe(df_tb_s.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                     use_container_width=True)
        tables["5_10_Impact_Turn_Bend_SideBend"] = df_tb_s

    # 2.1.8 Imp & Add/Imp (GS + Base)
    if gs_pro is not None and gs_ama is not None:
        st.divider(); st.subheader("2.1.8 Imp & Add/Imp")
        df_sum = t218.build_summary_phase_table(gs_pro, gs_ama, pro_arr, ama_arr)
        st.dataframe(df_sum.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                     use_container_width=True)
        tables["2_1_8_Imp_AddImp"] = df_sum

    # 2.1.9 Follow1
    st.divider(); st.subheader("2.1.9 Follow1")
    df_q8 = fol1.build_quarter8_phase_table(pro_arr, ama_arr)
    st.dataframe(df_q8.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["2_1_9_Follow1"] = df_q8

    # 2.1.10 Follow2
    st.divider(); st.subheader("2.1.10 Follow2")
    df_q9q10 = fol2.build_quarter9_10_phase_table(pro_arr, ama_arr)
    st.dataframe(df_q9q10.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
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
