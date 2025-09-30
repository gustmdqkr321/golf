from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from .features import _1center_gravity as feat
from .features import _2center_move as move
from .features import _3total_move as zmove
from .features import _4speed as speed

META = {"id": "center_move", "title": "Center Move", "icon": "🎯", "order": 41}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다."); return

    pro_arr: np.ndarray = ctx.get("pro_arr")
    ama_arr: np.ndarray = ctx.get("ama_arr")
    if pro_arr is None or ama_arr is None:
        st.warning("무지개(베이직) 엑셀 두 개(프로/일반)가 필요합니다."); return


    # 1) SMDI / MRMI
    st.markdown("### SMDI / MRMI")
    smdi = feat.build_smdi_mrmi_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(smdi.style.format({"SMDI":"{:.2f}","MRMI X":"{:.2f}","MRMI Y":"{:.2f}","MRMI Z":"{:.2f}"}),
                 use_container_width=True)


    # 2) ΔX
    st.markdown("### ΔX (COM vs BaseX)")
    dx = feat.build_delta_x_table(pro_arr, ama_arr)
    st.dataframe(dx.style.format({"프로":"{:.2f}","일반":"{:.2f}","프로 diff":"{:.2f}","일반 diff":"{:.2f}"}),
                 use_container_width=True)

    st.divider()

    # 3) ΔY
    st.markdown("### ΔY (COM Height)")
    dy = feat.build_delta_y_table(pro_arr, ama_arr)
    st.dataframe(dy.style.format({"프로":"{:.2f}","일반":"{:.2f}","프로 diff":"{:.2f}","일반 diff":"{:.2f}"}),
                 use_container_width=True)

    st.divider()

    # 4) ΔZ
    st.markdown("### ΔZ (Laterality)")
    dz = feat.build_delta_z_table(pro_arr, ama_arr)
    st.dataframe(dz.style.format({"프로":"{:.2f}","일반":"{:.2f}","프로 diff":"{:.2f}","일반 diff":"{:.2f}"}),
                 use_container_width=True)

    st.divider()

    # 5) Summary
    st.markdown("### Summary (Segments & Totals)")
    sm = feat.build_summary_table(pro_arr, ama_arr)
    st.dataframe(sm.style.format({"프로":"{:.2f}","일반":"{:.2f}"}),
                 use_container_width=True)
    st.download_button("CSV 내려받기 (Summary)", sm.to_csv(index=False).encode("utf-8-sig"),
                       "center_move_summary.csv", "text/csv", key="cm_summary")


    # ... run(ctx) 내부, 기존 표들 아래에 추가 --------------------------------
    st.divider()
    st.subheader("Part Movement (Δ between frames)")

    st.markdown("**Knee**")
    knee = move.build_movement_table_knee(pro_arr, ama_arr)
    st.dataframe(knee, use_container_width=True)

    st.markdown("**Hips**")
    hips = move.build_movement_table_hips(pro_arr, ama_arr)
    st.dataframe(hips, use_container_width=True)

    st.markdown("**Shoulder**")
    sho = move.build_movement_table_shoulder(pro_arr, ama_arr)
    st.dataframe(sho, use_container_width=True)

    st.markdown("**Head**")
    head = move.build_movement_table_head(pro_arr, ama_arr)
    st.dataframe(head, use_container_width=True)

    st.divider()
    st.subheader("Total Move (abs sum)")
    tm = move.build_total_move(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(tm.style.format({c:"{:.2f}" for c in tm.columns if c!="구간"}), use_container_width=True)

    st.divider()
    st.subheader("Move Ratio (%)")
    tr = move.build_total_move_ratio(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(tr.style.format({c:"{:.2f}" for c in tr.columns if c!="구간"}), use_container_width=True)

    st.divider()
    st.subheader("1-10 Abs Move (Σ|Δ|)")
    dfz = zmove.build_z_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(dfz, use_container_width=True)

    st.divider()
    st.markdown("### X Report")
    dfx = zmove.build_x_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(dfx, use_container_width=True)

    st.divider()
    st.markdown("### Y Report")
    dfy = zmove.build_y_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(dfy, use_container_width=True)

    # 프레임별 상세(원래 표)
    st.subheader("Tilt Report (per frame)")
    df_tilt = speed.compute_tilt_report(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(df_tilt, use_container_width=True)

    st.divider()
    st.subheader("Δθ Summary (Σ over segments)")
    df_delta = speed.build_tilt_delta_summary_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(df_delta.style.format({c:"{:.2f}" for c in df_delta.columns if c!="구간"}),
                 use_container_width=True)

    st.divider()
    st.subheader("Speed Summary (avg over segments)")
    df_speed = speed.build_tilt_speed_summary_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(df_speed.style.format({c:"{:.2f}" for c in df_speed.columns if c!="구간"}),
                 use_container_width=True)