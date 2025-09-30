# sections/forces/main.py
import streamlit as st
import pandas as pd
from .features import _1power as fc
from .features import _2torque as fc2

META = {"id": "forces", "title": "힘/토크 비교", "icon": "🧲", "order": 15}
def get_metadata(): return META

# 표기 라벨
_FORCE_PARTS = [
    ("knee", "무릎"),
    ("pelvis", "골반"),
    ("shoulder", "어깨"),
    ("wrist", "손목"),
    ("clubhead", "클럽헤드"),
]
_TORQUE_PARTS = [
    ("knee", "무릎"),
    ("pelvis", "골반"),
    ("shoulder", "어깨"),
]

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if not ctx or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]; ama_arr = ctx["ama_arr"]

    # 전역 설정만 상단에서 입력
    col1, col2 = st.columns([1,1])
    with col1:
        mass = st.number_input("질량(kg)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)
    with col2:
        summary_mode = st.selectbox(
            "요약 방식(토크)",
            options=["mean", "abs_sum"],
            index=0,
            help="요약 1-4/4-7/7-10에서 평균(mean) 또는 절대합(abs_sum)을 사용"
        )

    # ───────────────────────── 힘(Force) 전 부위 ─────────────────────────
    st.markdown("## 🧠 힘(Force) 비교 — 모든 부위")
    for part, label in _FORCE_PARTS:
        with st.expander(f"🔹 {label} — Force", expanded=False):
            try:
                res = fc.build_all_tables(pro_arr, ama_arr, part=part, mass=mass)

                st.markdown("**표 1. 전체 힘 비교표 (요약·지표 포함)**")
                st.dataframe(res.table_main, use_container_width=True)

                st.markdown("**표 2. 부호 반대 항목만 (차이 큰 순, 요약 제외)**")
                st.dataframe(res.table_opposite, use_container_width=True)

                st.markdown("**표 3. 부호 같고 차이 큰 상위 3 (xyz 무구분, 요약 제외)**")
                st.dataframe(res.table_same_top3, use_container_width=True)
            except Exception as e:
                st.warning(f"{label} Force 계산 중 오류: {e}")

    # ───────────────────────── 토크(Torque) 전 부위 ────────────────────────
    st.divider()
    st.markdown("## 🔧 토크(Torque) 비교 — 무릎/골반/어깨")

    for part, label in _TORQUE_PARTS:
        with st.expander(f"🔹 {label} — Torque", expanded=False):
            try:
                tres = fc2.build_torque_tables(
                    pro_arr, ama_arr,
                    part=part,
                    mass=mass,
                    summary_mode=summary_mode
                )

                st.markdown("**표 1. 전체 토크 비교표 (요약·지표 포함)**")
                st.dataframe(tres.table_main, use_container_width=True)

                st.markdown("**표 2. 부호 반대 항목만 (차이 큰 순, 요약 제외)**")
                st.dataframe(tres.table_opposite, use_container_width=True)

                st.markdown("**표 3. 부호 같고 차이 큰 상위 3 (xyz 무구분, 요약 제외)**")
                st.dataframe(tres.table_same_top3, use_container_width=True)
            except Exception as e:
                st.warning(f"{label} Torque 계산 중 오류: {e}")
