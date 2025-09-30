# sections/setup_address/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from .features import _1t as pos  # build_* 함수들 사용

META = {"id": "setup", "title": "Setup Address", "icon": "🧭", "order": 6}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]

    (tab_only,) = st.tabs(["Grip & L WRI/CLU"])

    with tab_only:
        # 1) Grip face angle (BN1 − AY1)
        st.subheader("2.1.1.1 Grip face angle — BN1 − AY1")
        grip_df = pos.build_grip_compare(pro_arr, ama_arr)  # 항목 / 프로 / 일반 / 차이
        st.dataframe(
            grip_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        st.divider()

        # 2) L WRI/CLU (CP1 − AZ1) — 기존 posture 비교표 활용
        st.subheader("2.1.1.2-23 L WRI/CLU — CP1 − AZ1")
        df = pos.build_posture_compare(pro_arr, ama_arr)
        st.dataframe(
            df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # 3) 2.1.1.3 Alignment (L/R)
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("2.1.1.3 Alignment (L/R)")
        align_df = pos.build_alignment_compare(pro_arr, ama_arr)
        st.dataframe(
            align_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # 4) 2.1.1.4 Stance & Ball Position (ALL)
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("2.1.1.4 Stance & Ball Position — ALL")
        sb_df = pos.build_stance_ball_compare(pro_arr, ama_arr)
        st.dataframe(
            sb_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # 5) 2.1.1.5 Basic Body Data (Length, cm) — ALL
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("2.1.1.5 Basic Body Data (Length, cm) — ALL")
        body_df = pos.build_basic_body_compare(pro_arr, ama_arr)
        st.dataframe(
            body_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )
