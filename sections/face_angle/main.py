# sections/face_angle/main.py
from __future__ import annotations
import streamlit as st
from .features import _1basic as feat
from .features import _2rolling as roll
from .features import _3dcocking as ck3
from .features import _4cocking2d as ck2
from .features import _5hinge as hinge
from .features import _6boncu as bc
from .features import _7tilt as tilt
from .features import _8foream as aux

import io, re
import pandas as pd
from datetime import datetime

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

# 시트명 안전화 (엑셀 금지문자 제거, 31자 제한, 중복 처리)
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

# 섹션 → 엑셀 단일 시트로 쓰기(섹션 내부 표 dict를 세로로 쌓음)
def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 빈 시트를 먼저 한 번 써서 핸들 확보
    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    cur_row = 0
    for name, df in tables.items():
        # 제목
        ws.write(cur_row, 0, str(name), title_fmt)
        cur_row += 1

        # 본문
        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        # 헤더/숫자 포맷 + 너비
        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)
        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

        # 다음 표 시작 위치로 이동: 본문 n_rows + 헤더 1 + 여백 2
        cur_row += n_rows + 1 + 2


META = {"id": "face_angle", "title": "7. Face Angle", "icon": "🎯", "order": 19}
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

    # ─────────────────────────────────────────────────────────────
    # 1) 표 생성
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}

    # 1. Face Angle 기본 표 (셀/식 숨김 버전)
    df_face = feat.build_face_angle_table(gs_pro, gs_ama, base_pro, base_ama)
    st.subheader("Face Angle (Summary)")
    st.dataframe(df_face.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["1_Basic Data"] = df_face

    # 2. Rolling
    st.divider(); st.subheader("Rolling")
    df_roll = roll.build_rolling_summary_table(base_pro, base_ama, alpha=2.0)
    st.dataframe(
        df_roll.style.format({
            "손목(프로)":"{:.2f}", "손목(일반)":"{:.2f}",
            "순수롤링(프로)":"{:.2f}", "순수롤링(일반)":"{:.2f}",
            "유사도(%)":"{:.2f}",
        }, na_rep=""),
        use_container_width=True
    )
    tables["2.Wrist Rolling Angle"] = df_roll

    # 3. 3D Cocking
    st.divider(); st.subheader("3D Cocking")
    df_ck3 = ck3.compute_cocking_table_from_arrays(base_pro, base_ama)
    num_cols = ["Pro ∠ABC","Ama ∠ABC","Pro Δ(°)","Ama Δ(°)","Similarity(0–100)"]
    st.dataframe(df_ck3.style.format({c: "{:.2f}" for c in num_cols}),
                 use_container_width=True)
    tables["3_3D_Cocking"] = df_ck3

    # 4. 2D Cocking
    st.divider(); st.subheader("2D Cocking")
    df_ck2 = ck2.build_yz_plane_compare_table(base_pro, base_ama)
    st.dataframe(df_ck2.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["4_2D_Cocking"] = df_ck2

    # 5. Hinging
    st.divider(); st.subheader("Hinging")
    df_hinge = hinge.build_hinging_compare_table(base_pro, base_ama, alpha=2.0)
    st.dataframe(
        df_hinge.style.format({
            "Pro Hinging(°)":"{:.2f}", "ΔPro(°)":"{:+.2f}",
            "Ama Hinging(°)":"{:.2f}", "ΔAma(°)":"{:+.2f}",
            "Similarity(0-100)":"{:.2f}",
        }, na_rep=""),
        use_container_width=True
    )
    tables["5_Hinging"] = df_hinge

    # 6. Bowing/Cupping
    st.divider(); st.subheader("Bowing / Cupping")
    df_bc = bc.build_bowing_table_from_arrays(base_pro, base_ama)
    st.dataframe(
        df_bc.style.format({
            "Pro Rel. Bowing (°)" : "{:.2f}",
            "Ama Rel. Bowing (°)" : "{:.2f}",
            "Pro ΔRel. Bowing"    : "{:.2f}",
            "Ama ΔRel. Bowing"    : "{:.2f}",
            "Similarity"          : "{:.2f}",
        }, na_rep=""),
        use_container_width=True
    )
    tables["6_Bowing_Cupping"] = df_bc

    # 7. Tilt
    st.divider(); st.subheader("Tilt")
    df_tilt = tilt.build_tilt_compare_table(base_pro, base_ama)
    st.dataframe(
        df_tilt.style.format({
            "Pro Tilt (°)" : "{:.2f}",
            "Ama Tilt (°)" : "{:.2f}",
            "similarity"   : "{:.2f}",
        }),
        use_container_width=True,
    )
    tables["7_Clubface : open/close(Heel/Toe Tilt) "] = df_tilt

    # 8. CLUB: (-) CLOSE, (+) OPEN
    st.divider(); st.subheader("CLUB  : (-): CLOSE, (+) : OPEN")
    df_club = aux.build_tilt_numerators_table(base_pro, base_ama)
    st.dataframe(df_club.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["8_Club_OpenClose"] = df_club

    # 9. Forearm Supination 1
    st.divider(); st.subheader("Forearm Supination 1")
    df_sup1 = aux.build_ay_bn_diffs_table(base_pro, base_ama)
    st.dataframe(df_sup1.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["9_Forearm_Supination_1"] = df_sup1

    # 10. Forearm Supination 2
    st.divider(); st.subheader("Forearm Supination 2")
    df_sup2 = aux.build_abc_angles_table(base_pro, base_ama)
    st.dataframe(df_sup2.style.format({"프로 ∠ABC(°)":"{:.2f}","일반 ∠ABC(°)":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                 use_container_width=True)
    tables["10_Forearm_Supination_2"] = df_sup2

    # ─────────────────────────────────────────────────────────────
    # 2) 섹션 단일 시트 엑셀 다운로드 + 마스터 합본 등록
    # ─────────────────────────────────────────────────────────────
    # 단일 시트 엑셀
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        sheet_name = "FaceAngle"
        _write_section_sheet(writer, sheet_name, tables)
    xbuf.seek(0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Face Angle (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"face_angle_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # 마스터 합본에 등록 (app.py의 “마스터 합본 다운로드”에서 한 번에 합쳐짐)
    register_section(META["id"], META["title"], tables)
