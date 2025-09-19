from __future__ import annotations
import streamlit as st

from .features import _1distance as gs
from .features import _2direction as dir
from .features import _3etc as etc

META = {"id": "gs", "title": "GS 표(프로·일반)", "icon": "📑", "order": 17}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if not ctx:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    base_pro = ctx.get("pro_arr")
    base_ama = ctx.get("ama_arr")
    gs_pro   = ctx.get("gs_pro_arr")   # DataFrame
    gs_ama   = ctx.get("gs_ama_arr")   # DataFrame

    if gs_pro is None or gs_ama is None:
        st.warning("GS CSV(프로/일반)를 업로드하거나 app.py 디폴트 경로를 설정하세요.")
        return
    if base_pro is None or base_ama is None:
        st.warning("무지개(기존) 엑셀도 필요합니다. 사이드바에서 업로드하세요.")
        return


    df = gs.build_gs_mixed_compare(gs_pro, gs_ama, base_pro, base_ama)

    st.dataframe(
        df.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )
    st.download_button(
        "CSV (GS+무지개 비교표)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="gs_mixed_compare.csv",
        mime="text/csv",
    )
    st.divider()
    df = dir.build_gs_club_table(gs_pro, gs_ama, base_pro, base_ama)
    st.dataframe(
        df.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"},na_rep=""),
        use_container_width=True
    )

    st.divider()
    df = etc.build_gs_b48_b55_table(gs_pro, gs_ama)
    st.dataframe(
        df.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )