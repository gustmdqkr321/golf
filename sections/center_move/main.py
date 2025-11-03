# sections/center_move/main.py
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from .features import _1center_gravity as feat
from .features import _2center_move as move
from .features import _3total_move as zmove
from .features import _4speed as speed

# ─────────────────────────────────────────────────────────────────────
# 마스터 합본용 세션 저장소
# {section_id: {"title": str, "tables": dict[str, DataFrame]}}
# ─────────────────────────────────────────────────────────────────────
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}

def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

# (엑셀) 한 섹션=한 시트로 내보내기 유틸
import io, re
from datetime import datetime

def _safe_sheet(name: str, used: set[str]) -> str:
    s = re.sub(r'[\\/\?\*\[\]\:\'"]', '', str(name)).strip()
    s = (s or "Sheet").replace(' ', '_')[:31]
    base, i = s, 1
    while s in used:
        suf = f"_{i}"
        s = (base[:31-len(suf)] if len(base) > 31-len(suf) else base) + suf
        i += 1
    used.add(s)
    return s

def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    """섹션 내부 여러 표를 한 시트에 제목+표 형태로 세로로 연속 기록"""
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 시트 핸들 확보
    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    cur_row = 0
    for name, df in tables.items():
        # 제목
        ws.write(cur_row, 0, str(name), title_fmt)
        cur_row += 1

        # 표
        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        # 헤더/숫자 포맷 + 열 너비
        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)
        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

        # 다음 표 위치: 본문 n_rows + 헤더 1 + 여백 2
        cur_row += n_rows + 1 + 2


META = {"id": "center_move", "title": "10. Center Move", "icon": "🎯", "order": 41}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다."); return

    pro_arr: np.ndarray = ctx.get("pro_arr")
    ama_arr: np.ndarray = ctx.get("ama_arr")
    if pro_arr is None or ama_arr is None:
        st.warning("무지개(베이직) 엑셀 두 개(프로/일반)가 필요합니다."); return

    # ─────────────────────────────────────────────────────────────
    # 1) 화면 표 생성 & 수집
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}

    # 1) SMDI / MRMI
    st.markdown("### 스윙이동평가지표(swing movement evalution indicators)")
    smdi = feat.build_smdi_mrmi_table(pro_arr, ama_arr, "Pro", "Ama")  # 열: Pro, Ama / 행: SMDI, MRMI X/Y/Z

    # 컬럼 기준 포맷 지정
    fmt = {col: "{:.2f}" for col in smdi.columns}  # {"Pro": "{:.2f}", "Ama": "{:.2f}"}
    st.dataframe(smdi.style.format(fmt), use_container_width=True)

    # 엑셀용 저장은 DataFrame 원본으로 그대로
    tables["스윙이동평가지표(swing movement evalution indicators)"] = smdi


    # 2) ΔX
    st.markdown("### 무게중심 X")
    dx = feat.build_delta_x_table(pro_arr, ama_arr)
    st.dataframe(dx.style.format({"프로":"{:.2f}","일반":"{:.2f}","프로 diff":"{:.2f}","일반 diff":"{:.2f}"}),
                 use_container_width=True)
    tables["무게중심 X"] = dx

    st.divider()

    # 3) ΔY
    st.markdown("### 수직중심 Y")
    dy = feat.build_delta_y_table(pro_arr, ama_arr)
    st.dataframe(dy.style.format({"프로":"{:.2f}","일반":"{:.2f}","프로 diff":"{:.2f}","일반 diff":"{:.2f}"}),
                 use_container_width=True)
    tables["수직중심 Y"] = dy

    st.divider()

    # 4) ΔZ
    st.markdown("### 무게중심 Z")
    dz = feat.build_delta_z_table(pro_arr, ama_arr)
    st.dataframe(dz.style.format({"프로":"{:.2f}","일반":"{:.2f}","프로 diff":"{:.2f}","일반 diff":"{:.2f}"}),
                 use_container_width=True)
    tables["무게중심 Z"] = dz

    st.divider()

    # 5) Summary
    st.markdown("### Summary")
    sm = feat.build_summary_table(pro_arr, ama_arr)
    st.dataframe(sm.style.format({"프로":"{:.2f}","일반":"{:.2f}"}),
                 use_container_width=True)
    st.download_button("CSV 내려받기 (Summary)", sm.to_csv(index=False).encode("utf-8-sig"),
                       "center_move_summary.csv", "text/csv", key="cm_summary")
    tables["Summary"] = sm

    # ── Part Movement ────────────────────────────────────────────
    st.divider()
    st.subheader("Part Movement (Δ between frames)")

    st.markdown("**Knee**")
    knee = move.build_movement_table_knee(pro_arr, ama_arr)
    st.dataframe(knee, use_container_width=True)
    tables["PartMovement_Knee"] = knee

    st.markdown("**Hips**")
    hips = move.build_movement_table_hips(pro_arr, ama_arr)
    st.dataframe(hips, use_container_width=True)
    tables["PartMovement_Hips"] = hips

    st.markdown("**Shoulder**")
    sho = move.build_movement_table_shoulder(pro_arr, ama_arr)
    st.dataframe(sho, use_container_width=True)
    tables["PartMovement_Shoulder"] = sho

    st.markdown("**Head**")
    head = move.build_movement_table_head(pro_arr, ama_arr)
    st.dataframe(head, use_container_width=True)
    tables["PartMovement_Head"] = head

    # ── Total Move / Ratio ───────────────────────────────────────
    st.divider()
    st.subheader("Total Move (abs sum)")
    tm = move.build_total_move(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(tm.style.format({c:"{:.2f}" for c in tm.columns if c!="구간"}), use_container_width=True)
    tables["신체분절 구간별 총 이동크기"] = tm

    st.divider()
    st.subheader("Move Ratio (%)")
    tr = move.build_total_move_ratio(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(tr.style.format({c:"{:.2f}" for c in tr.columns if c!="구간"}), use_container_width=True)
    tables["신체분절 이동 비율표"] = tr

    # ── 1-10 Abs Move & X/Y Report ───────────────────────────────
    st.divider()
    st.subheader("z축 변화량 최종표")
    dfz = zmove.build_z_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(dfz, use_container_width=True)
    tables["z축 변화량 최종표"] = dfz

    st.divider()
    st.markdown("### X 축 변화량 최종표")
    dfx = zmove.build_x_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(dfx, use_container_width=True)
    tables["X축 변화량 최종표"] = dfx

    st.divider()
    st.markdown("### Y 축 변화량 전체표")
    dfy = zmove.build_y_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(dfy, use_container_width=True)
    tables["Y축 변화량 최종표"] = dfy

    # ── Tilt / Speed ─────────────────────────────────────────────
    st.subheader("골반 어깨 좌우 높이 차이 및 속도와 힘")
    df_tilt = speed.compute_tilt_report(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(df_tilt.style.format({c:"{:.2f}" for c in df_tilt.columns if c!="Frame"}),
                    use_container_width=True)
    tables["골반 어깨 좌우 높이 차이 및 속도와 힘"] = df_tilt

    st.divider()
    st.subheader("골반 및 어깨 좌우 높이 차이와 속도, 힘")
    df_delta = speed.build_tilt_delta_summary_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(df_delta.style.format({c:"{:.2f}" for c in df_delta.columns if c!="구간"}),
                 use_container_width=True)
    tables["골반 및 어깨 좌우 높이 차이와 속도, 힘"] = df_delta

    st.divider()
    st.subheader("골반 및 어깨 좌우 높이 차이와 속도, 힘")
    df_speed = speed.build_tilt_speed_summary_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(df_speed.style.format({c:"{:.2f}" for c in df_speed.columns if c!="구간"}),
                 use_container_width=True)
    tables["골반 및 어깨 좌우 높이 차이와 속도, 힘"] = df_speed

    # ─────────────────────────────────────────────────────────────
    # 2) 섹션 단일 시트 엑셀 다운로드 + 마스터 합본 등록
    # ─────────────────────────────────────────────────────────────
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        _write_section_sheet(writer, "Center_Move", tables)
    xbuf.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Center Move (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"center_move_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # 마스터 합본(app.py)에서 섹션별 시트로 모을 수 있도록 등록
    register_section(META["id"], META["title"], tables)
