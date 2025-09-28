# sections/club_hand/main.py
from __future__ import annotations
import streamlit as st
from .features import _1distance as dis
from .features import _2rot_ang as rot
from .features import _3TDD as tdd
from .features import _4rot_center as rc
from .features import _5summ as misc

META = {"id": "club_hand", "title": "Club & Hand", "icon": "🤝", "order": 41}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    pro_arr = ctx.get("pro_arr")
    ama_arr = ctx.get("ama_arr")
    if pro_arr is None or ama_arr is None:
        st.warning("무지개(기존) 엑셀 두 개(프로/일반)가 필요합니다.")
        return

    df = dis.build_club_hand_table(pro_arr, ama_arr, pro_label="Pro", ama_label="Ama")
    st.dataframe(
        df.style.format({
            "ADD→TOP 이동거리(m)": "{:.2f}",
            "ADD→TOP 평균속도(m/s)": "{:.2f}",
            "TOP→IMP 이동거리(m)": "{:.2f}",
            "TOP→IMP 평균속도(m/s)": "{:.2f}",
            "TOP→IMP 평균가속도(m/s²)": "{:.2f}",
            "임팩트 순간 힘(N)": "{:.2f}",
            "ADD→TOP 평균속도(m/s) 비율(로리=100)": "{:.2f}",
            "임팩트 순간 힘(N) 비율(로리=100)": "{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("왼팔 회전각 (Left Arm)")
    df_left = rot.build_left_arm_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        df_left.style.format({
            "수평(Pro)":"{:.2f}", "수평(Ama)":"{:.2f}",
            "수직(Pro)":"{:.2f}", "수직(Ama)":"{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("클럽 회전각 (Wrist → Clubhead)")
    df_club = rot.build_club_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        df_club.style.format({
            "수평(Pro)":"{:.2f}", "수평(Ama)":"{:.2f}",
            "수직(Pro)":"{:.2f}", "수직(Ama)":"{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("무릎 TDD")
    df_knee = tdd.build_knee_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(df_knee, use_container_width=True)

    st.divider()
    st.markdown("무릎 수평 수직")
    df_knee_rot = rot.build_knee_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        df_knee_rot.style.format({
            "수평(Pro)":"{:.2f}", "수평(Ama)":"{:.2f}",
            "수직(Pro)":"{:.2f}", "수직(Ama)":"{:.2f}",
        }),
        use_container_width=True
    )
    
    st.divider()
    st.markdown("골반 TDD")
    df_pelvis = tdd.build_hip_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(df_pelvis, use_container_width=True)

    st.divider()
    st.markdown("골반 수평 수직")
    df_hip_rot = rot.build_hip_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        df_hip_rot.style.format({
            "수평(Pro)":"{:.2f}", "수평(Ama)":"{:.2f}",
            "수직(Pro)":"{:.2f}", "수직(Ama)":"{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.markdown("어깨 TDD")
    df_shoulder = tdd.build_shoulder_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(df_shoulder, use_container_width=True)

    st.divider()
    st.markdown("어깨 수평 수직")
    df_sho_rot = rot.build_shoulder_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        df_sho_rot.style.format({
            "수평(Pro)":"{:.2f}", "수평(Ama)":"{:.2f}",
            "수직(Pro)":"{:.2f}", "수직(Ama)":"{:.2f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.markdown("회전 중심")

    st.subheader("골반")
    df_p = rc.build_pelvis_center_table(pro_arr, ama_arr)
    st.dataframe(df_p, use_container_width=True)

    st.subheader("어깨")
    df_s = rc.build_shoulder_center_table(pro_arr, ama_arr)
    st.dataframe(df_s, use_container_width=True)

    st.subheader("무릎")
    df_k = rc.build_knee_center_table(pro_arr, ama_arr)
    st.dataframe(df_k, use_container_width=True)

    st.divider()
    st.subheader("회전 중심 구간차 (Ama − Pro)")
    df_center = misc.build_rotation_center_diff_all(pro_arr, ama_arr)
    st.dataframe(
        df_center.style.format({
            "X 차이 (Ama - Pro)": "{:+.2f}",
            "Y 차이 (Ama - Pro)": "{:+.2f}",
            "Z 차이 (Ama - Pro)": "{:+.2f}",
        }),
        use_container_width=True
    )