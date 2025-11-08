# sections/club_hand/main.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import io, re
from datetime import datetime

from .features import _1distance as dis
from .features import _2rot_ang as rot
from .features import _3TDD as tdd
from .features import _4rot_center as rc
from .features import _5summ as misc

# ── 세션 저장소 초기화 (마스터 병합용) ────────────────────────────────────────
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}   # {section_id: {"title": str, "tables": dict[str, DataFrame]}}

# ── 유틸: 시트명 안전화 ─────────────────────────────────────────────────────
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

# ── 유틸: 섹션 → 단일 시트에 세로로 쌓아 쓰기 ────────────────────────────────
def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

    # 빈 시트 한번 만들어 핸들 확보
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

# ── 유틸: 섹션 표 dict를 마스터에 등록 ───────────────────────────────────────
def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

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

def _style_highlight_rows_by_index(
    df: pd.DataFrame,
    row_indices: list[int],
    target_cols: list[str] | tuple[str, ...] = (),
    color: str = "#A9D08E",
) -> pd.io.formats.style.Styler:
    """
    row_indices: 0-based 인덱스 리스트.
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

def _style_with_key(table_key: str, df: pd.DataFrame, fmt: dict | None = None, color: str = "#A9D08E"):
    label_col, idxs = CH_TABLE_STYLES.get(table_key, ("", []))
    norm = _norm_indices(len(df), idxs)
    target_cols = (label_col,) if label_col else ()
    sty = _style_highlight_rows_by_index(df, norm, target_cols=target_cols, color=color)
    if fmt:
        sty = sty.format(fmt)
    return sty

# ─────────────────────────────────────────────────────────────
# ✅ Club & Hand 표별 인덱스 / 라벨 열 매핑
# (label_col을 ""로 두면 첫 열을 라벨로 자동 지정)
# 필요 시 아래 인덱스는 네 기준에 맞게 자유롭게 수정해!
# ─────────────────────────────────────────────────────────────
IDX_BASIC     = [0,1,2,3]         # "클럽헤드/손 운동량과 힘"
IDX_LEFT      = []         # "왼팔 수평/수직 회전각도"
IDX_CLUB      = []         # "클럽 수평/수직 회전각도"
IDX_KNEE_TDD  = []         # "무릎 TDD"
IDX_KNEE_ROT  = []         # "무릎 수평/수직 회전각도"
IDX_PELVIS_TDD= []         # "골반 TDD"
IDX_HIP_ROT   = []         # "골반 수평/수직 회전각도"
IDX_SHO_TDD   = []         # "어깨 TDD"
IDX_SHO_ROT   = []         # "어깨 수평/수직 회전각도"
IDX_PELVIS_C  = [0,1,2,3]         # "골반 회전 중심"
IDX_SHO_C     = [0,1,2,3]         # "어깨 회전 중심"
IDX_KNEE_C    = [0,1,2,3]         # "무릎 회전 중심"
IDX_SUMMARY   = []         # "통합표"

CH_TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "클럽헤드/손 운동량과 힘": ("", IDX_BASIC),
    "왼팔 수평/수직 회전각도": ("", IDX_LEFT),
    "클럽 수평/수직 회전각도": ("", IDX_CLUB),
    "무릎 TDD": ("", IDX_KNEE_TDD),
    "무릎 수평/수직 회전각도": ("", IDX_KNEE_ROT),
    "골반 TDD": ("", IDX_PELVIS_TDD),
    "골반 수평/수직 회전각도": ("", IDX_HIP_ROT),
    "어깨 TDD": ("", IDX_SHO_TDD),
    "어깨 수평/수직 회전각도": ("", IDX_SHO_ROT),
    "골반 회전 중심": ("", IDX_PELVIS_C),
    "어깨 회전 중심": ("", IDX_SHO_C),
    "무릎 회전 중심": ("", IDX_KNEE_C),
    "통합표": ("", IDX_SUMMARY),
}

META = {"id": "club_hand", "title": "11. Club & Hand", "icon": "🤝", "order": 41}
def get_metadata(): return META

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")
    if ctx is None:
        st.info("메인앱 컨텍스트가 없습니다.")
        return

    pro_arr = ctx.get("pro_arr")
    ama_arr = ctx.get("ama_arr")
    if pro_arr is None or ama_arr is None:
        st.warning("무지개(기존) 엑셀 두 개(프로/일반)가 필요합니다.")
        return

    # ── 표 생성 ─────────────────────────────────────────────────────────────
    df_basic = dis.build_club_hand_table(pro_arr, ama_arr, pro_label="Pro", ama_label="Ama")
    st.dataframe(
        _style_with_key(
            "클럽헤드/손 운동량과 힘",
            df_basic,
            fmt={
                "ADD→TOP 이동거리(m)": "{:.2f}",
                "ADD→TOP 평균속도(m/s)": "{:.2f}",
                "TOP→IMP 이동거리(m)": "{:.2f}",
                "TOP→IMP 평균속도(m/s)": "{:.2f}",
                "TOP→IMP 평균가속도(m/s²)": "{:.2f}",
                "임팩트 순간 힘(N)": "{:.2f}",
                "ADD→TOP 평균속도(m/s) 비율(로리=100)": "{:.2f}",
                "임팩트 순간 힘(N) 비율(로리=100)": "{:.2f}",
            },
        ),
        use_container_width=True
    )

    st.divider()
    st.subheader("왼팔 회전각 (Left Arm)")
    df_left = rot.build_left_arm_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "왼팔 수평/수직 회전각도",
            df_left,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.subheader("클럽 회전각")
    df_club = rot.build_club_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "클럽 수평/수직 회전각도",
            df_club,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.subheader("무릎 TDD")
    df_knee = tdd.build_knee_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(_style_with_key("무릎 TDD", df_knee), use_container_width=True)

    st.divider()
    st.markdown("무릎 수평 수직")
    df_knee_rot = rot.build_knee_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "무릎 수평/수직 회전각도",
            df_knee_rot,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )
    
    st.divider()
    st.markdown("골반 TDD")
    df_pelvis = tdd.build_hip_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(_style_with_key("골반 TDD", df_pelvis), use_container_width=True)

    st.divider()
    st.markdown("골반 수평 수직")
    df_hip_rot = rot.build_hip_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "골반 수평/수직 회전각도",
            df_hip_rot,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.markdown("어깨 TDD")
    df_shoulder = tdd.build_shoulder_tdd_table(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(_style_with_key("어깨 TDD", df_shoulder), use_container_width=True)

    st.divider()
    st.markdown("어깨 수평 수직")
    df_sho_rot = rot.build_shoulder_rotation_table(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "어깨 수평/수직 회전각도",
            df_sho_rot,
            fmt={"수평(Pro)":"{:.2f}","수평(Ama)":"{:.2f}","수직(Pro)":"{:.2f}","수직(Ama)":"{:.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.markdown("회전 중심")

    st.subheader("골반")
    df_p = rc.build_pelvis_center_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("골반 회전 중심", df_p), use_container_width=True)

    st.subheader("어깨")
    df_s = rc.build_shoulder_center_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("어깨 회전 중심", df_s), use_container_width=True)

    st.subheader("무릎")
    df_k = rc.build_knee_center_table(pro_arr, ama_arr)
    st.dataframe(_style_with_key("무릎 회전 중심", df_k), use_container_width=True)

    st.divider()
    st.subheader("회전 중심 구간차")
    df_center = misc.build_rotation_center_diff_all(pro_arr, ama_arr)
    st.dataframe(
        _style_with_key(
            "통합표",
            df_center,
            fmt={"X 차이 (Ama - Pro)":"{:+.2f}","Y 차이 (Ama - Pro)":"{:+.2f}","Z 차이 (Ama - Pro)":"{:+.2f}"},
        ),
        use_container_width=True
    )

    st.divider()
    st.subheader("회전각 요약 (구간별: 1-4 / 4-7 / 7-10 / 합계)")

    df_rot_summary = rot.build_rotation_summary_all(pro_arr, ama_arr, pro_label="Pro", ama_label="Ama")
    st.dataframe(
        df_rot_summary.style.format({
            "Pro 수평회전각": "{:.2f}", "Ama 수평회전각": "{:.2f}",
            "Pro 수직회전각": "{:.2f}", "Ama 수직회전각": "{:.2f}",
        }),
        use_container_width=True
    )
    st.divider()
    st.subheader("TDD 요약 (Knee / Pelvis / Shoulder, 구간별)")

    df_tdd_summary = tdd.build_tdd_summary_all(pro_arr, ama_arr, rot_to_m=0.01)
    st.dataframe(
        df_tdd_summary.style.format({
            "이동(Pro,m)": "{:.2f}", "이동(Ama,m)": "{:.2f}",
            "회전량(Pro,deg)": "{:.2f}", "회전량(Ama,deg)": "{:.2f}",
            "TDD(Pro,m)": "{:.2f}", "TDD(Ama,m)": "{:.2f}",
        }),
        use_container_width=True
    )


    # ── 단일 시트 엑셀 다운로드 + 마스터 등록 ────────────────────────────────
    # 섹션 내 모든 표를 dict로 모아 순서대로 한 시트에 쌓아 쓴다
    tables = {
        "클럽헤드/손 운동량과 힘": df_basic,
        "왼팔 수평/수직 회전각도":            df_left,
        "클럽 수평/수직 회전각도":                df_club,
        "무릎 TDD":                     df_knee,
        "무릎 수평/수직 회전각도":          df_knee_rot,
        "골반 TDD":                   df_pelvis,
        "골반 수평/수직 회전각도":        df_hip_rot,
        "어깨 TDD":                 df_shoulder,
        "어깨 수평/수직 회전각도":      df_sho_rot,
        "골반 회전 중심":                df_p,
        "어깨 회전 중심":              df_s,
        "무릎 회전 중심":                  df_k,
        "통합표":      df_center,
        "회전각 요약(구간별)": df_rot_summary,
        "TDD 요약(구간별)": df_tdd_summary,  # ✅ 추가
    }

    # 1) 단일 시트(All) 엑셀 다운로드 버튼
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        _write_section_sheet(writer, sheet_name="All", tables=tables)
    xbuf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📦 Excel 다운로드 – Club & Hand (단일 시트)",
        data=xbuf.getvalue(),
        file_name=f"club_hand_all_in_one_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_club_hand_all"
    )

    # 2) 마스터 엑셀 병합용 등록 버튼
    if st.button("➕ 이 섹션을 마스터 엑셀에 추가", use_container_width=True, key="reg_club_hand_master"):
        register_section(META["id"], META["title"], tables)
        st.success("Club & Hand 섹션을 마스터 엑셀에 등록했습니다. (사이드바/메인에서 '모든 섹션 합쳐서 다운로드' 버튼으로 병합 파일을 받을 수 있어요.)")
