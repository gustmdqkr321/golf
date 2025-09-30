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

META = {"id": "face_angle", "title": "Face Angle", "icon": "🎯", "order": 19}
def get_metadata(): return META

def run(ctx=None):
    st.subheader("1 Face Angle")

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

    df = feat.build_face_angle_table(gs_pro, gs_ama, base_pro, base_ama)

    st.dataframe(
        df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
        use_container_width=True
    )


    st.divider()
    st.subheader("rolling")
    if ctx is None:
        st.info("컨텍스트가 없습니다."); return
    
    df = roll.build_rolling_summary_table(base_pro, base_ama, alpha=2.0)
    st.dataframe(
        df.style.format({
            "손목(프로)":"{:.2f}", "손목(일반)":"{:.2f}",
            "순수롤링(프로)":"{:.2f}", "순수롤링(일반)":"{:.2f}",
            "유사도(%)":"{:.2f}",
        }, na_rep=""),
        use_container_width=True
    )

    st.divider()
    st.subheader("3D Cocking")
    df = ck3.compute_cocking_table_from_arrays(base_pro, base_ama)

    num_cols = ["Pro ∠ABC","Ama ∠ABC","Pro Δ(°)","Ama Δ(°)","Similarity(0–100)"]
    st.dataframe(
        df.style.format({c: "{:.2f}" for c in num_cols}),
        use_container_width=True
    )

    
    st.divider()
    st.subheader("2D Cocking")
    df_yz = ck2.build_yz_plane_compare_table(base_pro, base_ama)
    st.dataframe(
        df_yz.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    

    st.divider()
    st.subheader("Hinging")
    df_hinge = hinge.build_hinging_compare_table(base_pro, base_ama, alpha=2.0)
    st.dataframe(
        df_hinge.style.format({
            "pro Hinging(°)":"{:.2f}", "Δpro(°)":"{:+.2f}",
            "ama Hinging(°)":"{:.2f}", "Δama(°)":"{:+.2f}",
            "Similarity(0-100)":"{:.2f}",
        }, na_rep=""),
        use_container_width=True
    )


    st.divider()
    st.subheader("Bowing / Cupping")
    df = bc.build_bowing_table_from_arrays(base_pro, base_ama)
    st.dataframe(
        df.style.format({
            "Pro Rel. Bowing (°)" : "{:.2f}",
            "Ama Rel. Bowing (°)" : "{:.2f}",
            "Pro ΔRel. Bowing"    : "{:.2f}",
            "Ama ΔRel. Bowing"    : "{:.2f}",
            "Similarity"          : "{:.2f}",
        }, na_rep=""),
        use_container_width=True
    )

    st.divider()
    st.subheader("Tilt")
    df_tilt = tilt.build_tilt_compare_table(base_pro, base_ama)
    st.dataframe(
        df_tilt.style.format({
            "Pro Tilt (°)"   : "{:.2f}",
            "Ama Tilt (°)"   : "{:.2f}",
            "similarity" : "{:.2f}",
        }),
        use_container_width=True,
    )

    st.divider()
    st.subheader("CLUB  : (-): CLOSE, (+) : OPEN")
    df1 = aux.build_tilt_numerators_table(base_pro, base_ama)
    st.dataframe(df1.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)
   
    st.divider()
    st.subheader("Forearm Supination 1")
    df2 = aux.build_ay_bn_diffs_table(base_pro, base_ama)
    st.dataframe(df2.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)
  
    st.divider()
    st.subheader("Forearm Supination 2")
    df3 = aux.build_abc_angles_table(base_pro, base_ama)
    st.dataframe(df3.style.format({"프로 ∠ABC(°)":"{:.2f}","일반 ∠ABC(°)":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)
   