# sections/forces/main.py
from __future__ import annotations
import streamlit as st
import re
import io
import pandas as pd

from .features import _1power as fc   # Force
from .features import _2torque as fc2 # Torque (요약 abs-sum 고정판)

# ───────────────────────── 공통: 섹션 결과를 마스터 엑셀에 합치기 ─────────────────────────
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
    used.add(s)
    return s

def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 빈 시트 확보
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

# ───────────────────────── 메타 ─────────────────────────
META = {"id": "forces", "title": "12. 힘/토크 비교", "icon": "🧲", "order": 15}
def get_metadata(): return META

_FORCE_PARTS = [
    ("knee", "무릎"),
    ("pelvis", "골반"),
    ("shoulder", "어깨"),
    ("wrist", "손목"),
    ("clubhead", "클럽헤드"),
]
_TORQUE_PARTS = [
    ("knee", "무릎"),
    ("pelvis", "골반"),
    ("shoulder", "어깨"),
]

# ───────────────────────── 메인: UI 없이 표만 출력 ─────────────────────────
def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if not ctx or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]
    # ctx에 mass가 있으면 사용, 없으면 60.0kg
    mass = float(ctx.get("mass", 60.0))

    # 섹션 시트 병합용 표 수집
    section_tables: dict[str, pd.DataFrame] = {}

    # ── Force 표들 (순서대로 쭉 출력) ──
    for part, label in _FORCE_PARTS:
        try:
            res = fc.build_all_tables(pro_arr, ama_arr, part=part, mass=mass)
            st.markdown(f"### Force — {label}")
            st.dataframe(res.table_main, use_container_width=True)
            section_tables[f"Force/{label} - Main"] = res.table_main

            st.dataframe(res.table_opposite, use_container_width=True)
            section_tables[f"Force/{label} - Opposite-sign"] = res.table_opposite

            st.dataframe(res.table_same_top3, use_container_width=True)
            section_tables[f"Force/{label} - Same-sign Top3"] = res.table_same_top3
        except Exception as e:
            st.warning(f"{label} Force 계산 중 오류: {e}")

    try:
        df_tot = fc.build_totals_ratio_table(
            pro_arr, ama_arr,
            mass=mass,
            pro_label="pro",
            ama_label="ama",
        )
        st.subheader("힘 스타일")
        st.dataframe(df_tot, use_container_width=True)
        section_tables["힘 스타일"] = df_tot
    except Exception as e:
        st.warning(f"Force 요약표 생성 중 오류: {e}")

    # ── Torque 표들 (순서대로 쭉 출력, 요약=abs-sum 고정) ──
    for part, label in _TORQUE_PARTS:
        try:
            tres = fc2.build_torque_tables(pro_arr, ama_arr, part=part, mass=mass)
            st.markdown(f"### Torque — {label}")
            st.dataframe(tres.table_main, use_container_width=True)
            section_tables[f"Torque/{label} - Main"] = tres.table_main

            st.dataframe(tres.table_opposite, use_container_width=True)
            section_tables[f"Torque/{label} - Opposite-sign"] = tres.table_opposite

            st.dataframe(tres.table_same_top3, use_container_width=True)
            section_tables[f"Torque/{label} - Same-sign Top3"] = tres.table_same_top3
        except Exception as e:
            st.warning(f"{label} Torque 계산 중 오류: {e}")

    # 마스터 병합 등록 (별도 다운로드 UI 제거)
    register_section(META["id"], META["title"], section_tables)
