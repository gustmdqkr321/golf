# sections/swing/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd

META = {"id": "swing", "title": "1. 스타일", "icon": "🏌️", "order": 10}
def get_metadata(): 
    return META

# 필요 기능만 임포트
from .features import _1hand_hight as hand         # 01 손높이
from .features import _2swing_tempo as swing       # 02 템포/리듬
from .features import _7takeback as tb             # 07 테이크백(손목·클럽헤드)
from .features import _8top as top                 # 08 프레임4 벡터 차
from .features import _16ankle as ank              # 16 ANKLE: CL7-CL1
from .features import _25to26 as sp                # 26 SWING PATH (2/6)
from .features import setup as setup               # Setup / Address 요약표

# 마스터 엑셀 등록 폴백
try:
    from app import register_section as _register_section
except Exception:
    def _register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
        st.session_state.setdefault("section_tables", {})
        st.session_state["section_tables"][section_id] = {"title": section_title, "tables": tables}

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("상단 메인앱에서 프로/일반 엑셀을 업로드하면 여기서 자동으로 비교가 실행됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]

    tab_setup, tab_style = st.tabs(["Setup / Address", "스윙 스타일"])

    # ── Tab 1: Setup / Address ───────────────────────────────────────────
    tables_to_register: dict[str, pd.DataFrame] = {}

    with tab_setup:
        st.subheader("Setup 스타일")
        setup_df = setup.build_setup_summary_table(pro_arr, ama_arr)
        st.dataframe(
            setup_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}"}, na_rep=""),
            use_container_width=True
        )

        
        tables_to_register["셋업 스타일"] = setup_df

    # ── Tab 2: 스윙 스타일(각 항목 개별 표 + 이름) ─────────────────────────
    with tab_style:
        # 01 손높이 (row=4 관례)
        st.subheader("하이핸드 & 로우핸드")
        row_for_hand = 4
        p_m = hand.compute_metrics(pro_arr, row=row_for_hand)
        a_m = hand.compute_metrics(ama_arr, row=row_for_hand)
        df01 = hand.build_compare_df(p_m, a_m)
        st.dataframe(df01.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}), use_container_width=True)
        tables_to_register["하이핸드 & 로우핸드"] = df01

        st.divider()

        # 02 스윙 템포/리듬
        st.subheader("스윙 템포/리듬")
        pm = swing.compute_tempo_rhythm(pro_arr)
        am = swing.compute_tempo_rhythm(ama_arr)
        df02 = swing.build_tempo_rhythm_compare(pm, am)
        st.dataframe(df02.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}), use_container_width=True)
        tables_to_register["스윙 템포·리듬"] = df02

        st.divider()

        # 07 테이크백: 손목–클럽헤드 표
        st.subheader("테이크백 X,Y,Z")
        df07 = tb.build_wri_chd_table_compare(pro_arr, ama_arr)
        st.dataframe(df07.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}), use_container_width=True)

        tables_to_register["테이크백 X,Y,Z"] = df07

        st.divider()

        # 08 프레임4: CN4-AX4 / CO4-AY4 / CP4-AZ4
        st.subheader("TOP X,Y,Z")
        df08 = top.build_frame4_cnax_table(pro_arr, ama_arr)
        st.dataframe(df08.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}), use_container_width=True)

        tables_to_register["TOP X,Y,Z"] = df08

        st.divider()

        # 16 ANKLE: CL7 - CL1
        st.subheader("7 R Ankle Y")
        df16 = ank.build_cl7_minus_cl1_table(pro_arr, ama_arr)
        st.dataframe(df16.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}), use_container_width=True)
        
        tables_to_register["7 R Ankle Y"] = df16

        st.divider()

        # 26 SWING PATH (2/6)
        st.subheader("2/6 swing path")
        df26 = sp.build_26_swing_path(pro_arr, ama_arr)
        st.dataframe(df26.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}), use_container_width=True)

        tables_to_register["2/6 swing path"] = df26

    # 마스터 엑셀 내보내기용 등록 (섹션별 표 이름을 그대로 시트에 쓴다)
    _register_section(META["id"], META["title"], tables=tables_to_register)
