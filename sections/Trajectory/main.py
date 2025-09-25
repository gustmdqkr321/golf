from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from .features import _1basic as feat
from .features import _2loft as feat2
from .features import _3ang as ang
from .features import _4t as sum1
from .features import _5t as sum2
from .features import _6t as case

META = {"id": "trajectory", "title": "Trajectory", "icon": "🧭", "order": 40}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    pro_arr: np.ndarray = ctx.get("pro_arr")
    ama_arr: np.ndarray = ctx.get("ama_arr")
    gs_pro: pd.DataFrame = ctx.get("gs_pro_arr")
    gs_ama: pd.DataFrame = ctx.get("gs_ama_arr")

    if pro_arr is None or ama_arr is None or gs_pro is None or gs_ama is None:
        st.warning("무지개(베이직) 프로/일반 + GS(기어스) 프로/일반이 모두 필요합니다.")
        return

    st.subheader("4.4.1")
    df = feat.build_trajectory_table(gs_pro, gs_ama, pro_arr, ama_arr)
    st.dataframe(
        df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
        use_container_width=True,
    )

    st.download_button(
        "CSV 내려받기 (Trajectory - Basic)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="trajectory_basic.csv",
        mime="text/csv",
        key="dl_trajectory_basic",
    )

    st.divider()
    st.subheader("4.4.2")

    df_dm = feat2.build_dm_series_table(pro_arr, ama_arr)
    st.dataframe(df_dm.style.format("{:.2f}"), use_container_width=True)


    st.divider()
    st.subheader("L WRI/CHD Y & 6/7/8 Angle")

    df_ang = ang.build_wri_chd_angle_table(pro_arr, ama_arr)
    st.dataframe(
        df_ang.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )

    st.divider()
    st.subheader("손목, 양 어깨 Y차이각, 골반/어깨 틸트각, 어깨/팔 거리(XY거리)")

    df_metrics = sum1.build_metrics_table(pro_arr, ama_arr)
    st.dataframe(
        df_metrics.style.format({
            "Wrist Z Position":"{:.2f}",
            "Lateral Tilt (Y)":"{:.2f}",
            "Pelvis Z Tilt":"{:.2f}",
            "Shoulder Z Tilt":"{:.2f}",
            "Shoulder Z Tilt (Pelvis-based)":"{:.2f}",
            "Arm-Body Distance (XY)":"{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("Arm / Shoulder Angles")

    df_ang = sum2.build_arm_shoulder_angle_table(pro_arr, ama_arr)
    st.dataframe(
        df_ang.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )

    st.divider()
    st.subheader("club plane")
    df_case = case.build_bac_cases_table(pro_arr, ama_arr)
    st.dataframe(
        df_case.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )