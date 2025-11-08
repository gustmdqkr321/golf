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

# ─────────────────────────────────────────────────────────────
# ✅ 화면 하이라이트 유틸 (인덱스로 '라벨 열'만 색칠)
# ─────────────────────────────────────────────────────────────
def _norm_indices(n: int, idxs: list[int]) -> list[int]:
    """음수 인덱스(-1: 마지막 행 등) 허용 → 정규화"""
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
    """
    row_indices: 0-based 인덱스 리스트 (예: [0,3,-1]).
    target_cols: 색칠할 '라벨 열'만 지정. 비우면 첫 번째 열을 라벨로 간주.
    """
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
    """숫자열만 소수 둘째 자리 포맷"""
    fmt = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return styler.format(fmt)

# ─────────────────────────────────────────────────────────────
# ✅ Trajectory 표별 인덱스 / 라벨 열 매핑
# (label_col을 ""로 두면 첫 열을 라벨로 자동 지정)
# 필요 시 아래 인덱스는 네 기준으로 자유롭게 수정!
# ─────────────────────────────────────────────────────────────
IDX_BASIC      = []     # 4.4.1
IDX_LOFT       = [0, 1, 2]      # 4.4.2
IDX_WRI_CHD    = [2]   # 6/7/8 표
IDX_METRICS    = [0,1,2,3,6,7]   # 손목/틸트/거리
IDX_ARM_SHO    = []   # Arm/Shoulder
IDX_PLANE      = []  # club plane

TRAJ_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "4_4_1_Basic_Data": ("", IDX_BASIC),
    "4_4_2_Clubhead_Loft": ("", IDX_LOFT),
    "L_WRI_CHD_Y_and_Ang": ("", IDX_WRI_CHD),
    "손목, 어깨 y차이각, 골반/어깨 틸트, 어깨/팔 거리": ("", IDX_METRICS),
    "4 Flat/Upright": ("", IDX_ARM_SHO),
    "Club_Plane": ("", IDX_PLANE),
}

def _style_with_key(table_key: str, df: pd.DataFrame, color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    label_col, idxs = TRAJ_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    return _apply_2f(_style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color), df)

# ─────────────────────────────────────────────────────────────
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
    st.dataframe(_style_with_key("4_4_1_Basic_Data", df_basic), use_container_width=True)
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
    st.dataframe(_style_with_key("4_4_2_Clubhead_Loft", df_loft), use_container_width=True)
    tables["4_4_2_Clubhead_Loft"] = df_loft

    st.divider()
    st.subheader("6/7/8 L WRI/CHD Y and Ang")
    df_ang_wri = ang.build_wri_chd_angle_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("L_WRI_CHD_Y_and_Ang", df_ang_wri), use_container_width=True)
    tables["L_WRI_CHD_Y_and_Ang"] = df_ang_wri

    st.divider()
    st.subheader("손목, 양 어깨 Y차이각, 골반/어깨 틸트각, 어깨/팔 거리(XY거리)")
    df_metrics = sum1.build_metrics_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("손목, 어깨 y차이각, 골반/어깨 틸트, 어깨/팔 거리", df_metrics), use_container_width=True)
    tables["손목, 어깨 y차이각, 골반/어깨 틸트, 어깨/팔 거리"] = df_metrics

    st.divider()
    st.subheader("Arm / Shoulder Angles")
    df_armsho = sum2.build_arm_shoulder_angle_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("4 Flat/Upright", df_armsho), use_container_width=True)
    tables["4 Flat/Upright"] = df_armsho

    st.divider()
    st.subheader("club plane")
    df_plane = case.build_bac_cases_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("Club_Plane", df_plane), use_container_width=True)
    tables["Club_Plane"] = df_plane

    # ─────────────────────────────────────────────────────────────
    # 2) 섹션 단일 시트 엑셀 다운로드 + 마스터 합본 등록
    # (엑셀은 값만 저장 — 색칠은 화면 전용)
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
