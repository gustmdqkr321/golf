# sections/forces/main.py
import streamlit as st
import pandas as pd
from .features import power as fc

META = {"id": "forces", "title": "힘/토크 비교", "icon": "🧲", "order": 15}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if not ctx or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]; ama_arr = ctx["ama_arr"]

    col1, col2 = st.columns([1,1])
    with col1:
        part = st.selectbox("신체 부위",
    ("knee", "pelvis", "shoulder", "wrist"),  # ← wrist 추가
    format_func=lambda x: {"knee":"무릎","pelvis":"골반","shoulder":"어깨","wrist":"손목"}[x])
    with col2:
        mass = st.number_input("질량(kg)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)

    res = fc.build_all_tables(pro_arr, ama_arr, part=part, mass=mass)

    st.markdown("### 표 1. 전체 힘 비교표 (요약·지표 포함)")
    st.dataframe(res.table_main, use_container_width=True)

    st.markdown("### 표 2. 부호 반대 항목만 (차이 큰 순, 요약 제외)")
    st.dataframe(res.table_opposite, use_container_width=True)

    st.markdown("### 표 3. 부호 같고 차이 큰 상위 3 (xyz 무구분, 요약 제외)")
    st.dataframe(res.table_same_top3, use_container_width=True)

    # 다운로드
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("CSV 다운로드 - 표1", res.table_main.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{part}_force_main.csv", mime="text/csv")
    with c2:
        st.download_button("CSV 다운로드 - 표2", res.table_opposite.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{part}_force_opposites.csv", mime="text/csv")
    with c3:
        st.download_button("CSV 다운로드 - 표3", res.table_same_top3.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{part}_force_same_top3.csv", mime="text/csv")
