# sections/forces/main.py
from __future__ import annotations
import streamlit as st
import re, io
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

    # 빈 시트를 먼저 만들어 워크시트 핸들을 확보
    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    cur_row = 0
    for name, df in tables.items():
        # 제목
        ws.write(cur_row, 0, str(name), title_fmt)
        cur_row += 1

        # 본문
        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        # 헤더/숫자 포맷 + 열 너비
        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)
        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

        # 다음 표 간 간격 2줄
        cur_row += n_rows + 1 + 2

def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

# ───────────────────────── 메타 ─────────────────────────
META = {"id": "forces", "title": "12. 힘/토크 비교", "icon": "🧲", "order": 15}
def get_metadata(): return META

# 표기 라벨
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

# ───────────────────────── UI / 메인 로직 ─────────────────────────
def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if not ctx or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("메인에서 프로/일반 엑셀을 업로드하면 여기에서 표가 생성됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]

    # 전역 설정: 질량만 받기 (요약은 abs-sum 고정)
    mass = st.number_input("질량(kg)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)

    # 섹션 전체를 엑셀 한 시트로 저장하기 위해 표를 모아둘 dict
    section_tables: dict[str, pd.DataFrame] = {}

    # ───────── Force (모든 부위) ─────────
    st.markdown("## 🧠 힘(Force) 비교 — 모든 부위")
    for part, label in _FORCE_PARTS:
        with st.expander(f"🔹 {label} — Force", expanded=False):
            try:
                res = fc.build_all_tables(pro_arr, ama_arr, part=part, mass=mass)

                title_main = f"Force/{label} - Main"
                title_opp  = f"Force/{label} - Opposite-sign only"
                title_top3 = f"Force/{label} - Same-sign Top3(|Diff|)"

                st.markdown("**표 1. 전체 힘 비교표 (요약·지표 포함)**")
                st.dataframe(res.table_main, use_container_width=True)
                section_tables[title_main] = res.table_main

                st.markdown("**표 2. 부호 반대 항목만 (차이 큰 순, 요약 제외)**")
                st.dataframe(res.table_opposite, use_container_width=True)
                section_tables[title_opp] = res.table_opposite

                st.markdown("**표 3. 부호 같고 차이 큰 상위 3 (xyz 무구분, 요약 제외)**")
                st.dataframe(res.table_same_top3, use_container_width=True)
                section_tables[title_top3] = res.table_same_top3

            except Exception as e:
                st.warning(f"{label} Force 계산 중 오류: {e}")

    # ───────── Torque (무릎/골반/어깨, 요약=abs-sum 고정) ─────────
    st.divider()
    st.markdown("## 🔧 토크(Torque) 비교 — 무릎/골반/어깨")

    for part, label in _TORQUE_PARTS:
        with st.expander(f"🔹 {label} — Torque", expanded=False):
            try:
                # 요약 방식 선택 제거, 내부는 abs-sum 고정 구현판을 사용
                tres = fc2.build_torque_tables(pro_arr, ama_arr, part=part, mass=mass)

                title_main = f"Torque/{label} - Main"
                title_opp  = f"Torque/{label} - Opposite-sign only"
                title_top3 = f"Torque/{label} - Same-sign Top3(|Diff|)"

                st.markdown("**표 1. 전체 토크 비교표 (요약·지표 포함)**")
                st.dataframe(tres.table_main, use_container_width=True)
                section_tables[title_main] = tres.table_main

                st.markdown("**표 2. 부호 반대 항목만 (차이 큰 순, 요약 제외)**")
                st.dataframe(tres.table_opposite, use_container_width=True)
                section_tables[title_opp] = tres.table_opposite

                st.markdown("**표 3. 부호 같고 차이 큰 상위 3 (xyz 무구분, 요약 제외)**")
                st.dataframe(tres.table_same_top3, use_container_width=True)
                section_tables[title_top3] = tres.table_same_top3

            except Exception as e:
                st.warning(f"{label} Torque 계산 중 오류: {e}")

    # ───────── 이 섹션 전용 엑셀(단일 시트) 다운로드 + 마스터 등록 ─────────
    st.divider()
    st.subheader("📦 Forces 섹션 다운로드 / 마스터 병합 등록")

    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        used = set()
        sheet_name = _safe_sheet("Forces", used)
        _write_section_sheet(writer, sheet_name, section_tables)
    xbuf.seek(0)

    st.download_button(
        "⬇️ Excel 내려받기 (Forces 섹션 – 단일 시트)",
        data=xbuf.getvalue(),
        file_name="forces_section.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_forces_section",
    )

    # 마스터 병합용으로 현재 섹션 표들을 세션에 등록
    register_section(META["id"], META["title"], section_tables)
    st.success("이 섹션의 표들을 마스터 병합 목록에 등록했습니다. (메인에서 전체 합치기 버튼으로 병합 가능)")
