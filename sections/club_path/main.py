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


# app.py (상단 임포트 밑)
import io, re
import pandas as pd

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



META = {"id": "club_path", "title": "6. Club Path", "icon": "⛳️", "order": 18}
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

    
    st.subheader("4.2.7  Short Sho Back Turn")
    df_bc = bcax.build_bc4_minus_bc1_table(base_pro, base_ama)
    st.dataframe(df_bc.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)

    st.divider()
    st.subheader("4.2.8 Downswing path compared to backswing ")
    df_grp = bcax.build_ax_cn_group_6_2_table(base_pro, base_ama)
    st.dataframe(df_grp.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                use_container_width=True)
    
    st.divider()
    st.subheader("4.2.9 Swing Plane")
    df3 = sp.build_bac_cases_table(base_pro, base_ama)
    st.dataframe(df3.style.format({"프로(°)":"{:.2f}","일반(°)":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)

    st.divider()

    st.subheader("4.2.4 Elbow/ Wrist X")
    st.dataframe(stbl.build_cmp_ax_ar__bm_bg(base_pro, base_ama)
                .style.format({c:"{:.0f}" for c in map(str, range(1,10))}),
                use_container_width=True)

    st.subheader("4.2.5 Shoulder / Elbow X")
    st.dataframe(stbl.build_cmp_ar_al__bg_ba(base_pro, base_ama)
                .style.format({c:"{:.0f}" for c in map(str, range(1,10))}),
                use_container_width=True)

    st.subheader("4.2.6 Shoulder/ Wrist X")
    st.dataframe(stbl.build_cmp_ax_al__bm_ba(base_pro, base_ama)
                .style.format({c:"{:.0f}" for c in map(str, range(1,10))}),
                use_container_width=True)

    st.divider()
    st.subheader("2.2.4.8. R SHO/WRI X, Z")
    df_rws = shx.build_r_wrist_shoulder_x_table(base_pro, base_ama)
    st.dataframe(df_rws.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)

    st.divider()
    st.subheader("2.2.4.9 Shoulder / Elbow(X)")

    df_L, df_R = shx.build_shoulder_elbow_x_table_wide(base_pro, base_ama)

    st.caption("L")
    st.dataframe(df_L.style.format("{:.2f}"), use_container_width=True)

    st.caption("R")
    st.dataframe(df_R.style.format("{:.2f}"), use_container_width=True)

    st.divider()

    st.subheader("양 어깨 중심 축과 양 손목 중심 거리")
    df_cmp = ab.build_ab_distance_compare(base_pro, base_ama)
    st.dataframe(df_cmp.style.format({"프로 |AB|":"{:.2f}","일반 |AB|":"{:.2f}","차이(프로-일반)":"{:+.2f}"}), use_container_width=True)


        # ─────────────────────────────────────────────────────────────
    # (중략) — 여기까지는 화면 출력용 st.dataframe들
    # ─────────────────────────────────────────────────────────────

    # ── 1) 이 섹션의 모든 표를 dict로 모으기
    tables: dict[str, pd.DataFrame] = {
        "4.2.1 Basic": df_all,     # df_gs + df_ag 합친 표
        "4.2.3 L Wri/ CHD X":            df_cnax,
        "Both Sho Center/Wri. Horizon Rot Ang. Z(Yaw Ang)":               df_yaw,
        "Both Sho Center/Wri. Vertical Rot Ang. Z(Yaw Ang)":          df_pitch,
        "2.2.4.6 Both Sho Center/Elb X, Z":          df1,
        "2.2.4.7 BOT SHO CENTER/WRI X,Z":          df2,
        "4.2.7 Short Sho Back Turn": df_bc,
        "4.2.8 Backswing/ Downswing path": df_grp,
        "4.2.9 Swing Plane":         df3,
        "4.2.4 Elbow/Wrist X":       stbl.build_cmp_ax_ar__bm_bg(base_pro, base_ama),
        "4.2.5 Shoulder/Elbow X":    stbl.build_cmp_ar_al__bg_ba(base_pro, base_ama),
        "4.2.6 Shoulder/Wrist X":    stbl.build_cmp_ax_al__bm_ba(base_pro, base_ama),
        "2.2.4.8 R SHO/WRI X,Z":     df_rws,
        "2.2.4.9 Shoulder/Elbow X (L)": df_L,
        "2.2.4.9 Shoulder/Elbow X (R)": df_R,
        "Sho Center/Wri Center Distance": df_cmp,
    }

    # (옵션) 미리보기
    # with st.expander("전체 표 미리보기", expanded=False):
    #     for name, _df in tables.items():
    #         st.markdown(f"**{name}**")
    #         fmt = {c: "{:.2f}" for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])}
    #         st.dataframe(_df.style.format(fmt), use_container_width=True)
    #         st.divider()

    # ── 2) 단일 시트(이 섹션 전용)로 엑셀 다운로드
    import io
    from datetime import datetime

    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        sheet_name = _safe_sheet(META["title"], set())
        _write_section_sheet(writer, sheet_name, tables)

    xbuf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – 6. Club Path (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"club_path_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # ── 3) 마스터 엑셀 병합용으로 섹션 등록
    register_section(META["id"], META["title"], tables)
    st.success("마스터 엑셀에 이 섹션이 추가되었습니다. (메인 화면의 마스터 다운로드 버튼으로 전체를 받을 수 있어요)")
