# sections/sim_pro/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from .features import _1body as body
from .features import _2t as bnay          # ← BN1-AY1
from .features import _3t as abs_cam   # ← |CA1-CM1|
from .features import _4t as ab4          # ← AB 거리·행4
from .features import _5t as ab14
from .features import _6t as ab4mid
from .features import _7t as arbg
from .features import _8t as cndiff
from .features import _9t as bmet
from .features import vectorize as vec

PRO_DB_ROOT = "/Users/park_sh/Desktop/sim_pro/레퍼/driver"                         # ← 너의 프로 스윙 폴더 루트
PRO_FILE_PATTERN = "**/first_data_transi*.xlsx"  # ← 파일명 패턴(필요시 수정)
SAVE_DB_AS = "pros_db.npz"                     # ← 저장 파일명(선택)

META = {"id": "sim", "title": "2. 유사 프로 찾기", "icon": "🔎", "order": 15}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if not ctx or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]

    tab_body, tab_sim = st.tabs(["신체", "유사 프로 찾기"])

    # ── 탭 1: 신체(첫 표) ─────────────────────────────────────────────────────
    with tab_body:
        st.caption("계산 행: 1 (피처 내부 고정)")
        df = body.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        st.divider()
        st.caption("계산 행: 1 (피처 내부 고정)")
        df1 = bnay.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            df1.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        st.divider()
        st.caption("계산 행: 1 (피처 내부 고정)")
        df2 = abs_cam.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            df2.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        st.divider()
        st.caption("행 4 고정: A=(AX4,AY4,AZ4), B=(AC4,AD4,AE4)")
        df3 = ab4.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            df3.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        st.divider()
        st.caption("A=(AX1,AY1,AZ1), B=(AX4,AY4,AZ4) · 직각 ∠ACB는 Y<0→음수")
        df_ab = ab14.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            df_ab.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
            use_container_width=True
        )

        st.divider()
        st.caption("A=(AX4,AY4,AZ4),  B= ((AL4+BA4)/2, (AM4+BB4)/2, (AN4+BC4)/2)")
        df = ab4mid.build_compare_table(pro_arr, ama_arr)
        st.dataframe(df.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
   

        st.divider()
        st.caption("A=(AR1/4,AS1/4,AT1/4) · B=(BG1/4,BH1/4,BI1/4) — 3D 거리와 Δ(4−1)")
        df = arbg.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            df.style.format({"프로":"{:.3f}", "일반":"{:.3f}", "차이(프로-일반)":"{:+.3f}"}),
            use_container_width=True
        )

        st.divider()
        df1 = cndiff.build_compare_table(pro_arr, ama_arr)
        st.dataframe(df1.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
       

        st.divider()
        df2 = bmet.build_compare_table(pro_arr, ama_arr)
        st.dataframe(df2.style.format({"프로":"{:.3f}","일반":"{:.3f}","차이(프로-일반)":"{:+.3f}"}),
                    use_container_width=True)
       

    with tab_sim:
        st.subheader("유사도 검색 (고정 폴더 사용)")
        c1, c2 = st.columns(2)
        with c1:
            top_k = st.number_input("Top-K", min_value=1, max_value=50, value=5, step=1)
        with c2:
            per_pro = st.number_input("프로당 최대 결과(0=제한없음)", min_value=0, max_value=10, value=1, step=1)

        # 캐시해서 매번 폴더 스캔 비용 줄이기
        @st.cache_resource(show_spinner=True)
        def _build_db_cached(root_dir: str, pattern: str):
            return vec.build_db_from_fs(root_dir, pattern)

        try:
            db = _build_db_cached(PRO_DB_ROOT, PRO_FILE_PATTERN)
            st.success(f"DB 로드 완료 · 스윙 수={len(db['names'])}, 차원={db['Z'].shape[1]}")

            # 유사도 계산
            result = vec.search_similar(ama_arr, db, top_k=top_k)
            if per_pro > 0:
                result = vec.topk_per_pro(result, per_pro=per_pro)

            st.dataframe(result.style.format({"cosine_sim": "{:.4f}"}), use_container_width=True)


            # (옵션) 지금 DB를 파일로 저장 & 다운로드
            colA, colB = st.columns(2)
            with colA:
                if st.button("DB를 파일로 저장(.npz)"):
                    vec.save_db(db, SAVE_DB_AS)
                    st.success(f"저장됨: {SAVE_DB_AS}")
            with colB:
                try:
                    with open(SAVE_DB_AS, "rb") as f:
                        st.download_button("저장된 DB(.npz) 내려받기", data=f.read(), file_name=SAVE_DB_AS)
                except FileNotFoundError:
                    pass

            # (검증용) 내 스윙 벡터 보기
            with st.expander("내 스윙 벡터 보기(검증용)"):
                v, feat_names = vec.compute_feature_vector(ama_arr)
                st.write(f"벡터 차원: {len(v)}")
                st.dataframe(pd.DataFrame({"feature": feat_names, "value": v}), use_container_width=True)

        except Exception as e:
            st.error(f"DB 생성/로드 실패: {e}")
            st.caption(f"root={PRO_DB_ROOT}, pattern={PRO_FILE_PATTERN}")
