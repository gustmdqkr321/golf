from pathlib import Path
import streamlit as st
import pandas as pd


META = {"id": "setup", "title": "Setup Address", "icon": "🧭", "order": 6}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]