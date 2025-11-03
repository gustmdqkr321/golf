from __future__ import annotations
import streamlit as st

from .features import _1distance as gs
from .features import _2direction as dir
from .features import _3etc as etc

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

META = {"id": "gs", "title": "5. Ball & Club", "icon": "📑", "order": 17}
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

    # ─────────────────────────────────────────────────────────────
    # 1) 표 생성
    # ─────────────────────────────────────────────────────────────
    df_mix = gs.build_gs_mixed_compare(gs_pro, gs_ama, base_pro, base_ama)
    st.dataframe(
        df_mix.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )

    st.divider()
    df_dir = dir.build_gs_club_table(gs_pro, gs_ama, base_pro, base_ama)
    st.dataframe(
        df_dir.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}, na_rep=""),
        use_container_width=True
    )

    st.divider()
    df_etc = etc.build_gs_b48_b55_table(gs_pro, gs_ama)
    st.dataframe(
        df_etc.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
        use_container_width=True
    )

    # 이 섹션의 모든 표를 dict로 모으기 (→ 단일시트 내보내기 + 마스터 병합용)
    tables: dict[str, pd.DataFrame] = {
        "Mixed Compare": df_mix,
        "Club Direction": df_dir,
        "GS B48~B55": df_etc,
    }

    # 미리보기(옵션)
    with st.expander("전체 표 미리보기", expanded=False):
        for name, df in tables.items():
            st.markdown(f"**{name}**")
            fmt = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
            st.dataframe(df.style.format(fmt), use_container_width=True)
            st.divider()

    # ─────────────────────────────────────────────────────────────
    # 2) 단일 시트(이 섹션 전용) 엑셀 다운로드
    # ─────────────────────────────────────────────────────────────
    import io
    from datetime import datetime

    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        # 시트명은 섹션 타이틀로(안전화 필요시 _safe_sheet 사용)
        sheet_name = _safe_sheet(META["title"], set())
        _write_section_sheet(writer, sheet_name, tables)

    xbuf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – 4. Ball & Club (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"ball_club_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # ─────────────────────────────────────────────────────────────
    # 3) 마스터 엑셀(섹션별 시트) 병합을 위해 세션에 등록
    #    - app.py의 “마스터 엑셀 만들기” 버튼에서 이 저장소를 읽어
    #      섹션마다 시트 하나씩 쌓아 최종 파일을 생성할 수 있음.
    # ─────────────────────────────────────────────────────────────
    register_section(META["id"], META["title"], tables)
    st.success("마스터 엑셀에 이 섹션이 추가되었습니다. (앱 상단의 마스터 다운로드 버튼으로 전체를 받을 수 있어요)")
