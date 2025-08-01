# app.py (project_root)
import streamlit as st
from pathlib import Path
import importlib

# 1) 앱 목록 자동 수집
base = Path(__file__).parent / "apps"
apps = [p.name for p in base.iterdir() if p.is_dir()]

st.set_page_config(page_title="Multi‑Golf Swing 리포트", layout="wide")
st.sidebar.header("앱 선택")
app_choice = st.sidebar.selectbox("▶ 어떤 리포트를 보시겠습니까?", apps)

# 2) 업로드 UI
pro_upl    = st.sidebar.file_uploader("Pro 기준 엑셀",    type="xlsx")
golfer_upl = st.sidebar.file_uploader("Golfer 엑셀",     type="xlsx")
times_str  = st.sidebar.text_input("Pelvis frame 리스트", "0,1,2,3,4,5,6,7,8,9")

if pro_upl and golfer_upl:
    # 3) 임시 파일로 저장
    pro_path    = Path(pro_upl.name)
    golfer_path = Path(golfer_upl.name)
    pro_path.write_bytes(pro_upl.getbuffer())
    golfer_path.write_bytes(golfer_upl.getbuffer())
    times = [int(x) for x in times_str.split(",") if x.strip().isdigit()]

    st.sidebar.markdown("**⏳ 계산 중…**")
    # 4) 동적으로 해당 앱 모듈 import 후 main() 호출
    module = importlib.import_module(f"apps.{app_choice}.main")
    dfs = module.main(pro_path, golfer_path, times)

    # 5) 탭으로 출력
    tabs = st.tabs(list(dfs.keys()))
    for tab, (title, df) in zip(tabs, dfs.items()):
        with tab:
            st.subheader(f"{app_choice} – {title}")
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("CSV 다운로드", csv_bytes, file_name=f"{app_choice}_{title}.csv")

    st.success("✅ 완료!")

else:
    st.info("왼쪽 사이드바에서 Pro/​Golfer 엑셀을 업로드해주세요.")
