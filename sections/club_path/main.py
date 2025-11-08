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

# ─────────────────────────────────────────────────────────
# 세션 저장소/엑셀 유틸 (기존 그대로)
# ─────────────────────────────────────────────────────────
import io, re

if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}   # {section_id: {"title": str, "tables": dict[str, DataFrame]}}

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

def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    cur_row = 0
    for name, df in tables.items():
        ws.write(cur_row, 0, str(name), title_fmt)
        cur_row += 1

        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)
        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

        cur_row += n_rows + 1 + 2

def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

# ─────────────────────────────────────────────────────────
# ✅ 화면 하이라이트 유틸 (인덱스로 라벨 컬럼만 색칠)
# ─────────────────────────────────────────────────────────
def _norm_indices(n: int, idxs: list[int]) -> list[int]:
    out = []
    for i in idxs:
        j = n + i if i < 0 else i
        if 0 <= j < n:
            out.append(j)
    return sorted(set(out))

def _style_highlight_rows_by_index(df: pd.DataFrame,
                                   row_indices: list[int],
                                   target_cols: list[str] | tuple[str, ...] = (),
                                   color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    if not row_indices:
        return df.style
    if not target_cols:
        target_cols = (df.columns[0],)
    elif isinstance(target_cols, str):
        target_cols = (target_cols,)
    target_cols = [c for c in target_cols if c in df.columns]
    if not target_cols:
        target_cols = (df.columns[0],)

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    n = len(df)
    for idx in row_indices:
        if 0 <= idx < n:
            for c in target_cols:
                styles.iat[idx, df.columns.get_loc(c)] = f"background-color: {color}"
    return df.style.apply(lambda _df: styles, axis=None)

def _apply_2f(styler: pd.io.formats.style.Styler, df: pd.DataFrame) -> pd.io.formats.style.Styler:
    fmt = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return styler.format(fmt)

# ─────────────────────────────────────────────────────────
# ✅ 표별 인덱스/라벨 컬럼 매핑
# ─────────────────────────────────────────────────────────
# 여기에 “하이라이트하고 싶은 행 인덱스(0-based; 음수 OK)” 넣어줘
IDX_BASIC   = []      # 4.2.1 Basic (항목 라벨)
IDX_CHD     = [3,4,5]      # 4.2.3 L Wri/ CHD X (Frame 라벨일 가능성 높음)
IDX_YAW     = []      # Yaw (Frame)
IDX_VERT    = []      # Vertical (Frame)
IDX_2246    = []      # 2.2.4.6 표
IDX_2247    = []      # 2.2.4.7 표
IDX_BC      = [0]      # 4.2.7 Short Sho Back Turn (Frame/항목)
IDX_GRP     = [0,2,5]      # 4.2.8 Backswing/Downswing path (Frame/항목)
IDX_PLANE   = [5,6]      # 4.2.9 Swing Plane (항목)
IDX_EW_X    = []      # 4.2.4 Elbow/Wrist X (항목)
IDX_SE_X    = []      # 4.2.5 Shoulder/Elbow X (항목)
IDX_SW_X    = []      # 4.2.6 Shoulder/Wrist X (항목)
IDX_RWS     = []      # 2.2.4.8 R SHO/WRI X,Z (Frame/항목)
IDX_2249_L  = []      # 2.2.4.9 Shoulder/Elbow X (L)
IDX_2249_R  = []      # 2.2.4.9 Shoulder/Elbow X (R)
IDX_ABDIST  = []      # Sho Center/Wri Center Distance (항목)

# 표 키 → (라벨컬럼, 인덱스리스트)
CP_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "4.2.1 Basic": ("항목", IDX_BASIC),

    "4.2.3 L Wri/ CHD X": ("Frame", IDX_CHD),
    "Both Sho Center/Wri. Horizon Rot Ang. Z(Yaw Ang)": ("Frame", IDX_YAW),
    "Both Sho Center/Wri. Vertical Rot Ang. Z(Yaw Ang)": ("Frame", IDX_VERT),

    "2.2.4.6 Both Sho Center/Elb X, Z": ("항목", IDX_2246),
    "2.2.4.7 BOT SHO CENTER/WRI X,Z": ("항목", IDX_2247),

    "4.2.7 Short Sho Back Turn": ("Frame", IDX_BC),
    "4.2.8 Backswing/ Downswing path": ("Frame", IDX_GRP),
    "4.2.9 Swing Plane": ("항목", IDX_PLANE),

    "4.2.4 Elbow/Wrist X": ("항목", IDX_EW_X),
    "4.2.5 Shoulder/Elbow X": ("항목", IDX_SE_X),
    "4.2.6 Shoulder/Wrist X": ("항목", IDX_SW_X),

    "2.2.4.8 R SHO/WRI X,Z": ("Frame", IDX_RWS),

    "2.2.4.9 Shoulder/Elbow X (L)": ("항목", IDX_2249_L),
    "2.2.4.9 Shoulder/Elbow X (R)": ("항목", IDX_2249_R),

    "Sho Center/Wri Center Distance": ("항목", IDX_ABDIST),
}

def _style_with_key(table_key: str, df: pd.DataFrame, color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    label_col, idxs = CP_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    return _apply_2f(_style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color), df)

# ─────────────────────────────────────────────────────────
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

    # ── 표 생성 ─────────────────────────────────────────────────────
    df_gs = feat.build_gs_pair_table(gs_pro, gs_ama)
    df_ag = feat.build_alignment_grip_table(base_pro, base_ama)
    df_ag = df_ag.rename(columns={"식": "셀/식"})
    df_all = pd.concat([df_gs, df_ag], ignore_index=True)

    # ── 표/다운로드 (하이라이트 적용) ───────────────────────────────
    st.dataframe(
        _style_with_key("4.2.1 Basic", df_all),
        use_container_width=True
    )

    st.divider()
    st.subheader("CHD")
    df_cnax = chd.build_cn_ax_1_10_table(base_pro, base_ama)
    st.dataframe(
        _style_with_key("4.2.3 L Wri/ CHD X", df_cnax),
        use_container_width=True
    )

    st.divider()
    st.subheader("Yaw")
    df_yaw = yaw.build_yaw_compare_table(ctx["pro_arr"], ctx["ama_arr"])
    st.dataframe(_style_with_key("Both Sho Center/Wri. Horizon Rot Ang. Z(Yaw Ang)", df_yaw),
                 use_container_width=True)

    st.divider()
    st.subheader("Vertical")
    df_pitch = vert.build_pitch_compare_table(ctx["pro_arr"], ctx["ama_arr"])
    st.dataframe(_style_with_key("Both Sho Center/Wri. Vertical Rot Ang. Z(Yaw Ang)", df_pitch),
                 use_container_width=True)

    st.divider()
    df1, df2 = mid.build_midpoint_tables(base_pro, base_ama)

    st.subheader("2.2.4.6")
    st.dataframe(_style_with_key("2.2.4.6 Both Sho Center/Elb X, Z", df1),
                 use_container_width=True)

    st.subheader("2.2.4.7")
    st.dataframe(_style_with_key("2.2.4.7 BOT SHO CENTER/WRI X,Z", df2),
                 use_container_width=True)

    st.subheader("4.2.7  Short Sho Back Turn")
    df_bc = bcax.build_bc4_minus_bc1_table(base_pro, base_ama)
    st.dataframe(_style_with_key("4.2.7 Short Sho Back Turn", df_bc),
                 use_container_width=True)

    st.divider()
    st.subheader("4.2.8 Downswing path compared to backswing ")
    df_grp = bcax.build_ax_cn_group_6_2_table(base_pro, base_ama)
    st.dataframe(_style_with_key("4.2.8 Backswing/ Downswing path", df_grp),
                 use_container_width=True)
    
    st.divider()
    st.subheader("4.2.9 Swing Plane")
    df3 = sp.build_bac_cases_table(base_pro, base_ama)
    st.dataframe(_style_with_key("4.2.9 Swing Plane", df3),
                 use_container_width=True)

    st.divider()
    st.subheader("4.2.4 Elbow/ Wrist X")
    df_ew = stbl.build_cmp_ax_ar__bm_bg(base_pro, base_ama)
    st.dataframe(_style_with_key("4.2.4 Elbow/Wrist X", df_ew), use_container_width=True)

    st.subheader("4.2.5 Shoulder / Elbow X")
    df_se = stbl.build_cmp_ar_al__bg_ba(base_pro, base_ama)
    st.dataframe(_style_with_key("4.2.5 Shoulder/Elbow X", df_se), use_container_width=True)

    st.subheader("4.2.6 Shoulder/ Wrist X")
    df_sw = stbl.build_cmp_ax_al__bm_ba(base_pro, base_ama)
    st.dataframe(_style_with_key("4.2.6 Shoulder/Wrist X", df_sw), use_container_width=True)

    st.divider()
    st.subheader("2.2.4.8. R SHO/WRI X, Z")
    df_rws = shx.build_r_wrist_shoulder_x_table(base_pro, base_ama)
    st.dataframe(_style_with_key("2.2.4.8 R SHO/WRI X,Z", df_rws), use_container_width=True)

    st.divider()
    st.subheader("2.2.4.9 Shoulder / Elbow(X)")
    df_L, df_R = shx.build_shoulder_elbow_x_table_wide(base_pro, base_ama)
    st.caption("L")
    st.dataframe(_style_with_key("2.2.4.9 Shoulder/Elbow X (L)", df_L), use_container_width=True)
    st.caption("R")
    st.dataframe(_style_with_key("2.2.4.9 Shoulder/Elbow X (R)", df_R), use_container_width=True)

    st.divider()
    st.subheader("양 어깨 중심 축과 양 손목 중심 거리")
    df_cmp = ab.build_ab_distance_compare(base_pro, base_ama)
    st.dataframe(_style_with_key("Sho Center/Wri Center Distance", df_cmp),
                 use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # 표 dict 수집(엑셀/마스터)
    # ─────────────────────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {
        "4.2.1 Basic": df_all,
        "4.2.3 L Wri/ CHD X": df_cnax,
        "Both Sho Center/Wri. Horizon Rot Ang. Z(Yaw Ang)": df_yaw,
        "Both Sho Center/Wri. Vertical Rot Ang. Z(Yaw Ang)": df_pitch,
        "2.2.4.6 Both Sho Center/Elb X, Z": df1,
        "2.2.4.7 BOT SHO CENTER/WRI X,Z": df2,
        "4.2.7 Short Sho Back Turn": df_bc,
        "4.2.8 Backswing/ Downswing path": df_grp,
        "4.2.9 Swing Plane": df3,
        "4.2.4 Elbow/Wrist X": df_ew,
        "4.2.5 Shoulder/Elbow X": df_se,
        "4.2.6 Shoulder/Wrist X": df_sw,
        "2.2.4.8 R SHO/WRI X,Z": df_rws,
        "2.2.4.9 Shoulder/Elbow X (L)": df_L,
        "2.2.4.9 Shoulder/Elbow X (R)": df_R,
        "Sho Center/Wri Center Distance": df_cmp,
    }

    # ── 단일 시트 엑셀 다운로드
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

    # ── 마스터 등록
    register_section(META["id"], META["title"], tables)
    st.success("마스터 엑셀에 이 섹션이 추가되었습니다. (메인 화면의 마스터 다운로드 버튼으로 전체를 받을 수 있어요)")
