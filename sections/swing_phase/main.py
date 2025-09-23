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

META = {"id": "swing_phase", "title": "Swing Phase", "icon": "🏌️‍♂️", "order": 28}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    pro_arr = ctx.get("pro_arr")
    ama_arr = ctx.get("ama_arr")
    gs_pro   = ctx.get("gs_pro_arr")   # DataFrame
    gs_ama   = ctx.get("gs_ama_arr")   # DataFrame

    if pro_arr is None or ama_arr is None:
        st.warning("무지개(기존) 엑셀 두 개(프로/일반)가 필요합니다.")
        return

    df = feat.build_swing_phase_table(pro_arr, ama_arr)

    st.dataframe(
        df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
        use_container_width=True,
    )

    st.download_button(
        "CSV 내려받기 (Swing Phase)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="swing_phase_compare.csv",
        mime="text/csv",
        key="dl_swing_phase",
    )

    st.divider()
    st.subheader("half swing")
    df = half.build_swing_phase_table_v2(pro_arr, ama_arr)
    st.dataframe(df.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}), use_container_width=True)

    st.divider()
    st.subheader("2.1.4")
    df = t214.build_quarter_phase_table(pro_arr, ama_arr)
    st.dataframe(
        df.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("2.1.5 Transition")
    df_q5 = trans.build_quarter5_phase_table(pro_arr, ama_arr)
    st.dataframe(
        df_q5.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("2.1.6 Downswing")
    df_q6 = down.build_quarter6_phase_table(pro_arr, ama_arr)
    st.dataframe(
        df_q6.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("2.1.7 Impact")
    df_q7 = imp.build_quarter7_impact_table(pro_arr, ama_arr)
    st.dataframe(
        df_q7.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("5.10 Impact : Turn, Bend. Side Bend")
    df_tb = imp2.build_turn_bend_table(gs_pro, gs_ama)
    st.dataframe(
        df_tb.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("2.1.8. Imp & Add/Imp")
    df_sum = t218.build_summary_phase_table(gs_pro, gs_ama, pro_arr, ama_arr)
    st.dataframe(
        df_sum.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("2.1.9 Follow1")
    df_q8 = fol1.build_quarter8_phase_table(pro_arr, ama_arr)
    st.dataframe(
        df_q8.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("2.1.10 Follow2")
    df_q9q10 = fol2.build_quarter9_10_phase_table(pro_arr, ama_arr)
    st.dataframe(
        df_q9q10.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )