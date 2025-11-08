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

# ─────────────────────────────────────────────────────────────────────
# 화면 하이라이트 유틸 (인덱스로 라벨 컬럼만 색칠)
# ─────────────────────────────────────────────────────────────────────
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
    """row_indices(0-based) 행에서 target_cols만 칠함. target_cols가 비면 첫 컬럼을 라벨로 가정."""
    if not row_indices:
        return df.style
    if not target_cols:
        target_cols = (df.columns[0],)
    elif isinstance(target_cols, str):
        target_cols = (target_cols,)
    # 실제 존재하는 컬럼만
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

# ─────────────────────────────────────────────────────────────────────
# 표별 인덱스/라벨컬럼 매핑 (여기 숫자만 바꾸면 전체 화면 하이라이트가 따라감)
# ─────────────────────────────────────────────────────────────────────
# 네가 원하는 인덱스로 자유롭게 수정해서 써!
IDX_SMDI       = [0,1,2,3]          # 스윙이동평가지표
IDX_DX         = []          # Mass Center X
IDX_DY         = []          # Mass Center Y
IDX_DZ         = []          # Mass Center Z
IDX_SUMMARY    = [3,7,11,12]          # Summary(X,Y,Z)
IDX_PM_KNEE    = []          # PartMovement_Knee
IDX_PM_HIPS    = []          # PartMovement_Hips
IDX_PM_SHO     = []          # PartMovement_Shoulder
IDX_PM_HEAD    = []          # PartMovement_Head
IDX_TOTAL_MOVE = [0,1,2,3]          # Total PartMovement X,Y,Z Sum
IDX_RATIO      = []          # Total PartMovement X,Y,Z Sum Percentile
IDX_Z_REPORT   = [10,11,12,13]          # z Change
IDX_X_REPORT   = [10,11,12,13]          # X Change
IDX_Y_REPORT   = [10,11,12,13]          # Y Change
IDX_TILT1      = []          # Tilt report 1 (Frame 라벨)
IDX_TILT2      = [0,1,2]          # Tilt report 2 (구간 라벨)
IDX_TILT3      = [0,1,2]          # Tilt report 3 (구간 라벨)

# 표 키 → (라벨컬럼, 인덱스리스트)
CM_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "스윙이동평가지표(swing movement evalution indicators)": ("지표" , IDX_SMDI),   # df가 갖는 첫 컬럼명을 모르면 빈 값이면 첫 컬럼 자동
    "Mass Center X": ("Frame", IDX_DX),
    "Mass Center Y": ("Frame", IDX_DY),
    "Mass Center Z": ("Frame", IDX_DZ),
    "Mass Center X,Y, Z Summary": ("항목", IDX_SUMMARY),

    "PartMovement_Knee": ("Frame", IDX_PM_KNEE),
    "PartMovement_Hips": ("Frame", IDX_PM_HIPS),
    "PartMovement_Shoulder": ("Frame", IDX_PM_SHO),
    "PartMovement_Head": ("Frame", IDX_PM_HEAD),

    "Total PartMovement X,Y,Z Sum": ("구간", IDX_TOTAL_MOVE),
    "Total PartMovement X,Y,Z Sum Percentile": ("구간", IDX_RATIO),

    "z Change": ("구간", IDX_Z_REPORT),
    "X Change": ("구간", IDX_X_REPORT),
    "Y Change": ("구간", IDX_Y_REPORT),

    "Tilt of Pelvic and Shoulder and Velocity & Force 1": ("Frame", IDX_TILT1),
    "Tilt of Pelvic and Shoulder and Velocity & Force 2": ("구간", IDX_TILT2),
    "Tilt of Pelvic and Shoulder and Velocity & Force 3": ("구간", IDX_TILT3),
}

def _style_with_key(table_key: str, df: pd.DataFrame, color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    label_col, idxs = CM_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    return _apply_2f(_style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color), df)

# ─────────────────────────────────────────────────────────────────────
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
    smdi = feat.build_smdi_mrmi_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("스윙이동평가지표(swing movement evalution indicators)", smdi), use_container_width=True)
    tables["스윙이동평가지표(swing movement evalution indicators)"] = smdi

    # 2) ΔX
    st.markdown("### 무게중심 X")
    dx = feat.build_delta_x_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("Mass Center X", dx), use_container_width=True)
    tables["Mass Center X"] = dx

    st.divider()

    # 3) ΔY
    st.markdown("### 수직중심 Y")
    dy = feat.build_delta_y_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("Mass Center Y", dy), use_container_width=True)
    tables["Mass Center Y"] = dy

    st.divider()

    # 4) ΔZ
    st.markdown("### 무게중심 Z")
    dz = feat.build_delta_z_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("Mass Center Z", dz), use_container_width=True)
    tables["Mass Center Z"] = dz

    st.divider()

    # 5) Summary
    st.markdown("### Summary")
    sm = feat.build_summary_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("Mass Center X,Y, Z Summary", sm), use_container_width=True)
    st.download_button("CSV 내려받기 (Summary)", sm.to_csv(index=False).encode("utf-8-sig"),
                       "center_move_summary.csv", "text/csv", key="cm_summary")
    tables["Mass Center X,Y, Z Summary"] = sm

    # ── Part Movement ────────────────────────────────────────────
    st.divider()
    st.subheader("Part Movement (Δ between frames)")

    st.markdown("**Knee**")
    knee = move.build_movement_table_knee(pro_arr, ama_arr)
    st.dataframe(_style_with_key("PartMovement_Knee", knee), use_container_width=True)
    tables["PartMovement_Knee"] = knee

    st.markdown("**Hips**")
    hips = move.build_movement_table_hips(pro_arr, ama_arr)
    st.dataframe(_style_with_key("PartMovement_Hips", hips), use_container_width=True)
    tables["PartMovement_Hips"] = hips

    st.markdown("**Shoulder**")
    sho = move.build_movement_table_shoulder(pro_arr, ama_arr)
    st.dataframe(_style_with_key("PartMovement_Shoulder", sho), use_container_width=True)
    tables["PartMovement_Shoulder"] = sho

    st.markdown("**Head**")
    head = move.build_movement_table_head(pro_arr, ama_arr)
    st.dataframe(_style_with_key("PartMovement_Head", head), use_container_width=True)
    tables["PartMovement_Head"] = head

    # ── Total Move / Ratio ───────────────────────────────────────
    st.divider()
    st.subheader("Total Move (abs sum)")
    tm = move.build_total_move(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("Total PartMovement X,Y,Z Sum", tm), use_container_width=True)
    tables["Total PartMovement X,Y,Z Sum"] = tm

    st.divider()
    st.subheader("Move Ratio (%)")
    tr = move.build_total_move_ratio(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("Total PartMovement X,Y,Z Sum Percentile", tr), use_container_width=True)
    tables["Total PartMovement X,Y,Z Sum Percentile"] = tr

    # ── 1-10 Abs Move & X/Y Report ───────────────────────────────
    st.divider()
    st.subheader("z축 변화량 최종표")
    dfz = zmove.build_z_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("z Change", dfz), use_container_width=True)
    tables["z Change"] = dfz

    st.divider()
    st.markdown("### X 축 변화량 최종표")
    dfx = zmove.build_x_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("X Change", dfx), use_container_width=True)
    tables["X Change"] = dfx

    st.divider()
    st.markdown("### Y 축 변화량 전체표")
    dfy = zmove.build_y_report_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("Y Change", dfy), use_container_width=True)
    tables["Y Change"] = dfy

    # ── Tilt / Speed ─────────────────────────────────────────────
    st.subheader("골반 어깨 좌우 높이 차이 및 속도와 힘")
    df_tilt = speed.compute_tilt_report(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("Tilt of Pelvic and Shoulder and Velocity & Force 1", df_tilt),
                 use_container_width=True)
    tables["Tilt of Pelvic and Shoulder and Velocity & Force 1"] = df_tilt

    st.divider()
    st.subheader("골반 및 어깨 좌우 높이 차이와 속도, 힘")
    df_delta = speed.build_tilt_delta_summary_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("Tilt of Pelvic and Shoulder and Velocity & Force 2", df_delta),
                 use_container_width=True)
    tables["Tilt of Pelvic and Shoulder and Velocity & Force 2"] = df_delta

    st.divider()
    st.subheader("골반 및 어깨 좌우 높이 차이와 속도, 힘")
    df_speed = speed.build_tilt_speed_summary_table(pro_arr, ama_arr, "Pro", "Ama")
    st.dataframe(_style_with_key("Tilt of Pelvic and Shoulder and Velocity & Force 3", df_speed),
                 use_container_width=True)
    tables["Tilt of Pelvic and Shoulder and Velocity & Force 3"] = df_speed

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
