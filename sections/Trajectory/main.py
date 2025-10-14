# sections/trajectory/main.py
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from .features import _1basic as feat
from .features import _2loft as feat2
from .features import _3ang as ang
from .features import _4t as sum1
from .features import _5t as sum2
from .features import _6t as case

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

# (옵션) 시트명 안전화가 필요하면 사용
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


META = {"id": "trajectory", "title": "9. Trajectory", "icon": "🧭", "order": 40}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    pro_arr: np.ndarray = ctx.get("pro_arr")
    ama_arr: np.ndarray = ctx.get("ama_arr")
    gs_pro: pd.DataFrame = ctx.get("gs_pro_arr")
    gs_ama: pd.DataFrame = ctx.get("gs_ama_arr")

    if pro_arr is None or ama_arr is None or gs_pro is None or gs_ama is None:
        st.warning("무지개(베이직) 프로/일반 + GS(기어스) 프로/일반이 모두 필요합니다.")
        return

    # ─────────────────────────────────────────────────────────────
    # 1) 화면 표 생성 & 수집
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}

    # 4.4.1 Basic Data
    st.subheader("4.4.1 Basic Data")
    df_basic = feat.build_trajectory_table(gs_pro, gs_ama, pro_arr, ama_arr)
    st.dataframe(df_basic.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
                 use_container_width=True)
    tables["4_4_1_Basic_Data"] = df_basic

    st.download_button(
        "CSV 내려받기 (Trajectory - Basic)",
        data=df_basic.to_csv(index=False).encode("utf-8-sig"),
        file_name="trajectory_basic.csv",
        mime="text/csv",
        key="dl_trajectory_basic",
    )

    st.divider()
    st.subheader("4.4.2 Clubhead Loft")
    df_loft = feat2.build_dm_series_table(pro_arr, ama_arr)
    st.dataframe(df_loft.style.format("{:.2f}"), use_container_width=True)
    tables["4_4_2_Clubhead_Loft"] = df_loft

    st.divider()
    st.subheader("6/7/8 L WRI/CHD Y and Ang")
    df_ang_wri = ang.build_wri_chd_angle_table(pro_arr, ama_arr)
    st.dataframe(df_ang_wri.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["L_WRI_CHD_Y_and_Ang"] = df_ang_wri

    st.divider()
    st.subheader("손목, 양 어깨 Y차이각, 골반/어깨 틸트각, 어깨/팔 거리(XY거리)")
    df_metrics = sum1.build_metrics_table(pro_arr, ama_arr)
    st.dataframe(
        df_metrics.style.format({
            "Wrist Z Position":"{:.2f}",
            "Lateral Tilt (Y)":"{:.2f}",
            "Pelvis Z Tilt":"{:.2f}",
            "Shoulder Z Tilt":"{:.2f}",
            "Shoulder Z Tilt (Pelvis-based)":"{:.2f}",
            "Arm-Body Distance (XY)":"{:.2f}",
        }),
        use_container_width=True
    )
    tables["Wrist_ShoulderY_PelvisShoulderTilt_ArmBodyXY"] = df_metrics

    st.divider()
    st.subheader("Arm / Shoulder Angles")
    df_armsho = sum2.build_arm_shoulder_angle_table(pro_arr, ama_arr)
    st.dataframe(df_armsho.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["Arm_Shoulder_Angles"] = df_armsho

    st.divider()
    st.subheader("club plane")
    df_plane = case.build_bac_cases_table(pro_arr, ama_arr)
    st.dataframe(df_plane.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["Club_Plane"] = df_plane

    # ─────────────────────────────────────────────────────────────
    # 2) 섹션 단일 시트 엑셀 다운로드 + 마스터 합본 등록
    # ─────────────────────────────────────────────────────────────
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        _write_section_sheet(writer, "Trajectory", tables)
    xbuf.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Trajectory (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"trajectory_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # 마스터 합본(app.py)에서 섹션별 시트로 모을 수 있도록 등록
    register_section(META["id"], META["title"], tables)
