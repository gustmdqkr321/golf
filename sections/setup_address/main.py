# sections/setup_address/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from .features import _1t as pos  # build_* 함수들 사용
# app.py (상단 임포트 밑)
import io, re
import pandas as pd
from datetime import datetime
# 세션 저장소 초기화
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}   # {section_id: {"title": str, "tables": dict[str, DataFrame]}}

# 시트명 안전화
def _safe_sheet(name: str, used: set[str]) -> str:
    s = re.sub(r'[\\/\?\*\[\]\:\'"]', '', str(name)).strip()
    s = (s or "Sheet").replace(' ', '_')[:31]
    base, i = s, 1
    while s in used:
        suf = f"_{i}"
        s = (base[:31-len(suf)] if len(base) > 31-len(suf) else base) + suf
        i += 1
    used.add(s); 
    return s

# 섹션 → 시트 하나로 쓰기(섹션 내부 표 dict를 한 시트에 세로로 쌓음)
def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 먼저 빈 시트 한 번 만들어 핸들 확보
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

        # 다음 표 사이 여백 2줄
        cur_row += n_rows + 1 + 2

# 섹션에서 만든 표 dict를 마스터에 등록
def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

META = {"id": "setup", "title": "4. Setup Address", "icon": "🧭", "order": 6}
def get_metadata(): return META

def _export_single_sheet_excel(tables: dict[str, pd.DataFrame], sheet_name: str = "All") -> bytes:
    """이 섹션의 모든 표를 하나의 시트에 세로로 쌓아 엑셀 바이너리 반환"""
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        # 시트 핸들 확보
        pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
        wb = writer.book
        ws = writer.sheets[sheet_name]

        title_fmt = wb.add_format({'bold': True, 'font_size': 12})
        header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
        num_fmt = wb.add_format({'num_format': '0.00'})

        cur_row = 0
        for name, df in tables.items():
            # 제목
            ws.write(cur_row, 0, str(name), title_fmt)
            cur_row += 1

            # 데이터프레임 쓰기
            df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

            # 서식/너비
            n_rows, n_cols = df.shape
            for c in range(n_cols):
                ws.write(cur_row, c, df.columns[c], header_fmt)
            ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

            # 다음 표 간 여백
            cur_row += n_rows + 1 + 2

        ws.freeze_panes(1, 0)
    xbuf.seek(0)
    return xbuf.getvalue()

def _register_to_master(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    """마스터 엑셀에 섹션 등록 (register_section 있으면 호출, 없으면 세션에 직접 저장)"""
    reg = globals().get("register_section")
    if callable(reg):
        reg(section_id, section_title, tables)
        return
    # 폴백: 세션에 직접
    if "section_tables" not in st.session_state:
        st.session_state["section_tables"] = {}
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]

    (tab_only,) = st.tabs(["Grip & L WRI/CLU"])

    with tab_only:
        # 1) Grip face angle (BN1 − AY1)
        st.subheader("2.1.1.1 Grip face angle")
        grip_df = pos.build_grip_compare(pro_arr, ama_arr)  # 항목 / 프로 / 일반 / 차이
        st.dataframe(
            grip_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        st.divider()

        # 2) L WRI/CLU (CP1 − AZ1) — 기존 posture 비교표 활용
        st.subheader("2.1.1.2 Posture all")
        df_posture = pos.build_posture_compare(pro_arr, ama_arr)
        st.dataframe(
            df_posture.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # 3) 2.1.1.3 Alignment (L/R)
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("2.1.1.3 Alignment")
        align_df = pos.build_alignment_compare(pro_arr, ama_arr)
        st.dataframe(
            align_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # 4) 2.1.1.4 Stance & Ball Position (ALL)
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("2.1.1.4 Stance & Ball Position — ALL")
        sb_df = pos.build_stance_ball_compare(pro_arr, ama_arr)
        st.dataframe(
            sb_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # 5) 2.1.1.5 Basic Body Data (Length, cm) — ALL
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("2.1.1.5 Basic Body Data (Length, cm) — ALL")
        body_df = pos.build_basic_body_compare(pro_arr, ama_arr)
        st.dataframe(
            body_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )

        # ───────────────────────────────────────────────────────────
        # ▶ 이 섹션 표들을 단일 시트(All)에 담아 엑셀 다운로드 + 마스터에 등록
        # ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📦 이 섹션 엑셀 내보내기 (단일 시트)")

        # 이 섹션의 표를 모두 모아서 dict 구성
        tables = {
            "2.1.1.1 Grip face angle": grip_df,
            "2.1.1.2 Posture all": df_posture,
            "2.1.1.3 Alignment": align_df,
            "2.1.1.4 Stance & Ball Position — ALL": sb_df,
            "2.1.1.5 Basic Body Data (Length, cm) — ALL": body_df,
        }

        # 단일 시트 엑셀 생성
        excel_bytes = _export_single_sheet_excel(tables, sheet_name="All")
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📥 섹션 엑셀 다운로드 (All 시트 1장)",
            data=excel_bytes,
            file_name=f"setup_address_all_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # 마스터 엑셀에 이 섹션 등록(시트=섹션)
        _register_to_master(META["id"], META["title"], tables)