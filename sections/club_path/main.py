# sections/club_path/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from .features import _1basic as feat  # build_gs_pair_table, build_alignment_grip_table
from .features import _2CHD as chd
from .features import _3Yaw as yaw
from .features import _4vertical as vert
from .features import _5bot_sho as mid
from .features import _6t as bcax
from .features import _7swing_plane as sp
from .features import _8sho as stbl
from .features import _9t as shx
from .features import _10distance as ab


META = {"id": "club_path", "title": "Club Path", "icon": "⛳️", "order": 18}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

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

    # ── 표 생성 (각각) ─────────────────────────────────────────────────────
    df_gs = feat.build_gs_pair_table(gs_pro, gs_ama)              # ["항목","셀/식","프로","일반","차이(프로-일반)"]
    df_ag = feat.build_alignment_grip_table(base_pro, base_ama)   # ["항목","식","프로","일반","차이(프로-일반)"]

    # ── 하나로 합치기 ──────────────────────────────────────────────────────
    df_ag = df_ag.rename(columns={"식": "셀/식"})
    df_all = pd.concat([df_gs, df_ag], ignore_index=True)

    # ── 표/다운로드 ────────────────────────────────────────────────────────
    st.dataframe(
        df_all.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
        use_container_width=True
    )
    st.divider()
    st.subheader("CHD")
    df_cnax = chd.build_cn_ax_1_10_table(base_pro, base_ama)
    st.dataframe(
        df_cnax.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
        use_container_width=True
    )

    st.divider()
    st.subheader("Yaw")
    df_yaw = yaw.build_yaw_compare_table(ctx["pro_arr"], ctx["ama_arr"])
    st.dataframe(df_yaw.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)


    st.divider()
    st.subheader("Vertical")
    df_pitch = vert.build_pitch_compare_table(ctx["pro_arr"], ctx["ama_arr"])
    st.dataframe(
        df_pitch.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
        use_container_width=True
    )

    st.divider()
    df1, df2 = mid.build_midpoint_tables(base_pro, base_ama)

    st.subheader("2.2.4.6")
    st.dataframe(df1.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)

    st.subheader("2.2.4.7")
    st.dataframe(df2.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)

    
    st.subheader("BC4 − BC1")
    df_bc = bcax.build_bc4_minus_bc1_table(base_pro, base_ama)
    st.dataframe(df_bc.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)

    st.divider()
    st.subheader("AX/CN/CO/CP 6↔2 & 조합")
    df_grp = bcax.build_ax_cn_group_6_2_table(base_pro, base_ama)
    st.dataframe(df_grp.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)
    
    st.divider()
    st.markdown("#### case1…case11 ∠BAC + case6/7 판정")
    df3 = sp.build_bac_cases_table(base_pro, base_ama)
    st.dataframe(df3.style.format({"프로(°)":"{:.2f}","일반(°)":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)

    st.divider()
    st.markdown("#### 4.2.4 / 4.2.5 / 4.2.6 — 값만(프로·일반 동일 표)")

    st.caption("4.2.4  (L=AX−AR, R=BM−BG)")
    st.dataframe(stbl.build_cmp_ax_ar__bm_bg(base_pro, base_ama)
                .style.format({c:"{:.0f}" for c in map(str, range(1,10))}),
                use_container_width=True)

    st.caption("4.2.5  (L=AR−AL, R=BG−BA)")
    st.dataframe(stbl.build_cmp_ar_al__bg_ba(base_pro, base_ama)
                .style.format({c:"{:.0f}" for c in map(str, range(1,10))}),
                use_container_width=True)

    st.caption("4.2.6  (L=AX−AL, R=BM−BA)")
    st.dataframe(stbl.build_cmp_ax_al__bm_ba(base_pro, base_ama)
                .style.format({c:"{:.0f}" for c in map(str, range(1,10))}),
                use_container_width=True)

    st.divider()
    st.subheader("R Wrist–Shoulder (X)  —  BO1-BC1 / BMn-BAn / BO7-BC7")
    df_rws = shx.build_r_wrist_shoulder_x_table(base_pro, base_ama)
    st.dataframe(df_rws.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)

    st.divider()
    st.subheader("Shoulder / Elbow (X) — 가로형")

    df_L, df_R = shx.build_shoulder_elbow_x_table_wide(base_pro, base_ama)

    st.caption("L (ARn-ALn)")
    st.dataframe(df_L.style.format("{:.2f}"), use_container_width=True)

    st.caption("R (BGn-BAn)")
    st.dataframe(df_R.style.format("{:.2f}"), use_container_width=True)

    st.divider()

    st.subheader("거리만 비교")
    df_cmp = ab.build_ab_distance_compare(base_pro, base_ama)
    st.dataframe(df_cmp.style.format({"프로 |AB|":"{:.2f}","일반 |AB|":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)