from __future__ import annotations
import streamlit as st
import pandas as pd

from .features import _1FB as FB
from .features import _2BH as BH
from .features import _3LH as LH
from .features import _4SB as SB
from .features import _5Trust as TR
from .features import _6OTT as OTT
from .features import _7Trust2 as TR2

# ───────────────────────────────── 공통 유틸 (색칠/내보내기 등) ─────────────────────────────────
import re as _re

# ───────────────────── Master Excel 누적 유틸 ─────────────────────
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}   # {section_id: {"title": str, "tables": dict[str, DataFrame]}}

import re as _re2

def _safe_sheet(name: str, used: set[str]) -> str:
    """엑셀 시트명 안전화 + 31자 제한 + 중복 회피"""
    s = _re2.sub(r'[\\/\?\*\[\]\:\'"]', '', str(name)).strip()
    s = (s or "Sheet").replace(' ', '_')[:31]
    base, i = s, 1
    while s in used:
        suf = f"_{i}"
        s = (base[:31-len(suf)] if len(base) > 31-len(suf) else base) + suf
        i += 1
    used.add(s)
    return s

def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    """섹션 하나를 '한 시트'에 여러 표를 위→아래로 쌓아서 기록"""
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

        # 본문
        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        # 헤더/숫자 포맷 + 기본 너비
        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)
        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)

        # 다음 표 여백 2줄
        cur_row += n_rows + 1 + 2

def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    """현재 섹션(페이지)의 표 dict를 마스터에 등록"""
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

def style_highlight_rows_by_index(df: pd.DataFrame,
                                  row_indices: list[int],
                                  target_cols: list[str] | tuple[str, ...] = ("Frame",),
                                  color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    if not row_indices:
        return df.style
    if isinstance(target_cols, str):
        target_cols = (target_cols,)
    target_cols = [c for c in target_cols if c in df.columns]
    if not target_cols:
        return df.style

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    n = len(df)
    for idx in row_indices:
        if 0 <= idx < n:
            for c in target_cols:
                styles.iat[idx, df.columns.get_loc(c)] = f"background-color: {color}"
    return df.style.apply(lambda _df: styles, axis=None)

def _norm_indices(n: int, idxs: list[int]) -> list[int]:
    """음수 인덱스(-1: 마지막 행 등) 허용 → 정규화하여 반환"""
    out = []
    for i in idxs:
        j = n + i if i < 0 else i
        if 0 <= j < n:
            out.append(j)
    return sorted(set(out))

def _apply_2f(styler: pd.io.formats.style.Styler, df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """숫자열만 소수 둘째자리 포맷"""
    fmt = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return styler.format(fmt)

# ── 엑셀/PDF 내보내기 ─────────────────────────────────────────
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm

def export_tables_pdf(tables: dict[str, pd.DataFrame],
                      title: str = "Swing Error – All-in-One",
                      landscape_mode: bool = True) -> io.BytesIO:
    buf = io.BytesIO()
    page_size = landscape(A4) if landscape_mode else A4
    doc = SimpleDocTemplate(
        buf, pagesize=page_size,
        leftMargin=12*mm, rightMargin=12*mm, topMargin=12*mm, bottomMargin=12*mm
    )
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles['Title']), Spacer(1, 6)]

    for name, df in tables.items():
        story.append(Paragraph(str(name), styles['Heading2']))
        story.append(Spacer(1, 4))

        df2 = df.copy()
        for c in df2.columns:
            if pd.api.types.is_numeric_dtype(df2[c]):
                df2[c] = df2[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

        data = [list(df2.columns)] + df2.fillna("").astype(str).values.tolist()

        def _col_width(series):
            max_chars = max((len(s) for s in series), default=5)
            return max(18*mm, min(60*mm, max_chars * 2.5*mm))

        col_widths = [_col_width([str(h)] + df2[col].astype(str).tolist()) for col, h in zip(df2.columns, df2.columns)]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F2F2")),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))
    doc.build(story)
    buf.seek(0)
    return buf

# ── 섹션 메타/조합 유틸 ────────────────────────────────────────────────────
META = {"id": "swing_error", "title": "3. Swing Error", "icon": "⚠️", "order": 16}
def get_metadata(): return META

def combine_pro_ama_table(builder, pro_arr, ama_arr, *, key_col: str | None = None):
    if isinstance(builder, tuple):
        build_p, build_a = builder
        df_p = build_p(pro_arr)
        df_a = build_a(ama_arr)
    else:
        df_p = builder(pro_arr)
        df_a = builder(ama_arr)

    def _ensure_seg(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "seg" not in df.columns:
            base = "검사명" if "검사명" in df.columns else df.columns[0]
            df.insert(0, "seg", df[base].astype(str))
        return df.drop(columns=["검사명"], errors="ignore")

    df_p = _ensure_seg(df_p)
    df_a = _ensure_seg(df_a)

    if key_col is None:
        key_col = "seg"

    p = df_p.set_index(key_col, drop=False)
    a = df_a.set_index(key_col, drop=False)
    out = p.join(a, lsuffix="_프로", rsuffix="_일반", how="outer")
    out.insert(0, "seg", out.index.astype(str))
    cols_to_drop = [c for c in out.columns if c.startswith("seg_") or c.endswith("검사명")]
    out = out.drop(columns=cols_to_drop, errors="ignore")
    out.reset_index(drop=True, inplace=True)
    cols = ["seg"] + [c for c in out.columns if c != "seg"]
    return out[cols]

# ─────────────────────── “하이라이트 인덱스” 공통 정의 ───────────────────────
IDX_FB      = [0, 3, 10, 11, 12, -1]
IDX_BH      = [0, 10, 11, 12, 13]
IDX_LH      = []
IDX_SB      = [0, 10, 11, 12, 13]

IDX_TR_WAIST_X    = [3, 4, 5]
IDX_TR_SHOULDER_X = [4, 5, 6]
IDX_TR_HEAD_X     = []
IDX_TR_WAIST_Y    = []
IDX_TR_SHOULDER_Y = []
IDX_TR_HEAD_Y     = []

IDX_OTT           = [0, 1, 2, 9, 10, 11]

IDX_TR2_33        = []
IDX_TR2_EACH      = []

TABLE_STYLES: dict[str, tuple[str, list[int]]] = {
    "FrontalBend": ("Frame", IDX_FB),
    "BodyHinge":   ("Frame", IDX_BH),
    "LegHinge":    ("Frame", IDX_LH),
    "SideBend":    ("Frame", IDX_SB),

    "Thrust_Waist_X":    ("항목", IDX_TR_WAIST_X),
    "Thrust_Shoulder_X": ("항목", IDX_TR_SHOULDER_X),
    "Thrust_Head_X":     ("항목", IDX_TR_HEAD_X),
    "Lift_Waist_Y":      ("항목", IDX_TR_WAIST_Y),
    "Lift_Shoulder_Y":   ("항목", IDX_TR_SHOULDER_Y),
    "Lift_Head_Y":       ("항목", IDX_TR_HEAD_Y),

    "OverTheTop": ("항목", IDX_OTT),

    "3_3_EarlyExtension(Waist Thrust(X, cm))": ("seg", IDX_TR2_33),
    "3_4_FlatShoPlane":    ("seg", IDX_TR2_EACH),
    "3_5_FlyingElbow":     ("seg", IDX_TR2_EACH),
    "3_6_Sway":            ("seg", IDX_TR2_EACH),
    "3_7_Casting":         ("seg", IDX_TR2_EACH),
    "3_8_HangingBack":     ("seg", IDX_TR2_EACH),
    "3_9_Slide":           ("seg", IDX_TR2_EACH),
    "3_10_Overswing_Y":    ("seg", IDX_TR2_EACH),
    "3_11_CrossOver_X":    ("seg", IDX_TR2_EACH),
    "3_12_ReverseSpine":   ("seg", IDX_TR2_EACH),
    "3_13_ChickenWing":    ("seg", IDX_TR2_EACH),
    "3_14_Scooping":       ("seg", IDX_TR2_EACH),
    "3_15_ReverseCFinish": ("seg", IDX_TR2_EACH),
}

def style_with_table_key(table_key: str, df: pd.DataFrame, color: str = "#A9D08E") -> pd.io.formats.style.Styler:
    label_col, idxs = TABLE_STYLES.get(table_key, ("Frame", []))
    norm = _norm_indices(len(df), idxs)
    return style_highlight_rows_by_index(df, norm, target_cols=(label_col,), color=color)

# ─────────────────────────────────────────── 앱 메인 ───────────────────────────────────────────
def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("상단 메인앱에서 프로/일반 엑셀을 업로드하면 여기서 자동으로 비교가 실행됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]

    (tab_fb, tab_bh, tab_lh, tab_sb, tab_TR, tab_ott, tab_TR2, tab_all) = st.tabs(
        ["Frontal Bend", "Body Hinge", "Leg Hinge", "Side Bend",
         "Trust", "Over The Top", "Trust2", "전체 비교표"]
    )

    # ───────── Frontal Bend ─────────
    with tab_fb:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
        df = FB.build_frontal_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels,
            pro_name="프로", ama_name="일반"
        )
        st.dataframe(
            _apply_2f(
                style_with_table_key("FrontalBend", df, color="#A9D08E"), df
            ),
            use_container_width=True,
        )

    # ───────── Body Hinge ─────────
    with tab_bh:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
        cmp = BH.build_body_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels,
            pro_name="프로", ama_name="일반"
        )
        st.dataframe(
            _apply_2f(
                style_with_table_key("BodyHinge", cmp, color="#A9D08E"), cmp
            ),
            use_container_width=True
        )
        st.download_button(
            "CSV (Body Hinge 비교표)",
            data=cmp.to_csv(index=False).encode("utf-8-sig"),
            file_name="body_hinge_compare.csv",
            mime="text/csv"
        )

    # ───────── Leg Hinge ─────────
    with tab_lh:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
        cmp_lh = LH.build_leg_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels,
            pro_name="프로", ama_name="일반"
        )
        st.dataframe(
            _apply_2f(
                style_with_table_key("LegHinge", cmp_lh, color="#A9D08E"), cmp_lh
            ),
            use_container_width=True
        )
        st.download_button(
            "CSV (Leg Hinge 비교표)",
            data=cmp_lh.to_csv(index=False).encode("utf-8-sig"),
            file_name="leg_hinge_compare.csv",
            mime="text/csv"
        )

    # ───────── Side Bend ─────────
    with tab_sb:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
        cmp_sb = SB.build_side_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels,
            pro_name="프로", ama_name="일반"
        )
        st.dataframe(
            _apply_2f(
                style_with_table_key("SideBend", cmp_sb, color="#A9D08E"), cmp_sb
            ),
            use_container_width=True
        )
        st.download_button(
            "CSV (Side Bend 비교표)",
            data=cmp_sb.to_csv(index=False).encode("utf-8-sig"),
            file_name="side_bend_compare.csv",
            mime="text/csv"
        )

    # ───────── TR (Thrust/Lift) ─────────
    with tab_TR:
        st.subheader("Thrust(X, cm)")

        st.subheader("3.1.6.1 Waist")
        df_px = TR.build_compare_table(pro_arr, ama_arr)
        st.dataframe(
            _apply_2f(
                style_with_table_key("Thrust_Waist_X", df_px), df_px
            ),
            use_container_width=True
        )
        st.download_button("CSV (Pelvis X Shift 비교)",
                           data=df_px.to_csv(index=False).encode("utf-8-sig"),
                           file_name="pelvis_x_shift_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.6.2 Shoulder")
        df_sx = TR.build_shoulder_x_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        st.dataframe(
            _apply_2f(
                style_with_table_key("Thrust_Shoulder_X", df_sx), df_sx
            ),
            use_container_width=True
        )
        st.download_button("CSV (Shoulder X Center 비교)",
                           data=df_sx.to_csv(index=False).encode("utf-8-sig"),
                           file_name="shoulder_x_center_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.6.3 Head")
        df_hx = TR.build_head_x_compare(pro_arr, ama_arr)
        st.dataframe(
            _apply_2f(
                style_with_table_key("Thrust_Head_X", df_hx), df_hx
            ),
            use_container_width=True
        )
        st.download_button("CSV (Head X 비교)",
                           data=df_hx.to_csv(index=False).encode("utf-8-sig"),
                           file_name="head_x_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.1 Waist Lift (Y, cm)")
        df_wy = TR.build_waist_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        st.dataframe(
            _apply_2f(
                style_with_table_key("Lift_Waist_Y", df_wy), df_wy
            ),
            use_container_width=True
        )
        st.download_button("CSV (Waist Lift Y 비교)",
                           data=df_wy.to_csv(index=False).encode("utf-8-sig"),
                           file_name="waist_lift_y_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.2 Shoulder Lift (Y, cm)")
        df_sy = TR.build_shoulder_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        st.dataframe(
            _apply_2f(
                style_with_table_key("Lift_Shoulder_Y", df_sy), df_sy
            ),
            use_container_width=True
        )
        st.download_button("CSV (Shoulder Lift Y 비교)",
                           data=df_sy.to_csv(index=False).encode("utf-8-sig"),
                           file_name="shoulder_lift_y_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.3 Head (Y)")
        df_hy = TR.build_head_y_compare(pro_arr, ama_arr)
        st.dataframe(
            _apply_2f(
                style_with_table_key("Lift_Head_Y", df_hy), df_hy
            ),
            use_container_width=True
        )
        st.download_button("CSV (Head Y 비교)",
                           data=df_hy.to_csv(index=False).encode("utf-8-sig"),
                           file_name="head_y_compare.csv", mime="text/csv")

    # ───────── OTT ─────────
    with tab_ott:
        st.caption("프레임: 4, 5, 6 기준 (방법 1/2 포함)")
        cmp = OTT.build_over_the_top_compare(pro_arr, ama_arr, frames=(4,5,6), chd_col="CN", wrist_r_col="BM")
        st.dataframe(
            _apply_2f(
                style_with_table_key("OverTheTop", cmp), cmp
            ),
            use_container_width=True
        )
        st.download_button(
            "CSV (Over The Top 비교)",
            data=cmp.to_csv(index=False).encode("utf-8-sig"),
            file_name="over_the_top_compare.csv",
            mime="text/csv"
        )

    # ───────── TR2 (3.3 ~ 3.15) ─────────
    with tab_TR2:
        st.subheader("3.3 Early Extension (Waist Thrust X)")
        df33 = combine_pro_ama_table(
            (
                lambda a: TR2.build_33_early_extension(a, ctx["gs_pro_arr"]),
                lambda a: TR2.build_33_early_extension(a, ctx["gs_ama_arr"]),
            ),
            pro_arr, ama_arr, key_col=None
        )
        st.dataframe(
            _apply_2f(
                style_with_table_key("3_3_EarlyExtension(Waist Thrust(X, cm))", df33), df33
            ),
            use_container_width=True
        )
        st.download_button(
            "CSV (3.3·프로/일반 단일표)",
            df33.to_csv(index=False).encode("utf-8-sig"),
            "3_3_early_extension_compare.csv",
            "text/csv"
        )

        st.divider()
        items = [
            (TR2.build_34_flat_sho_plane, "3.4 Flat Sho Plane",               "3_4_FlatShoPlane"),
            (TR2.build_35_flying_elbow,   "3.5 Flying Elbow",                  "3_5_FlyingElbow"),
            (TR2.build_36_sway,           "3.6 Sway",                          "3_6_Sway"),
            (TR2.build_37_casting,        "3.7 Casting",                       "3_7_Casting"),
            (TR2.build_38_hanging_back,   "3.8 Hanging Back (Z, − Greater)",   "3_8_HangingBack"),
            (TR2.build_39_slide,          "3.9 Slide (Z, + Greater)",          "3_9_Slide"),
            (TR2.build_310_overswing_y,   "3.10 Overswing (Y, − Greater)",     "3_10_Overswing_Y"),
            (TR2.build_311_cross_over_x,  "3.11 Cross Over (X, − Greater)",    "3_11_CrossOver_X"),
            (TR2.build_312_reverse_spine, "3.12 Reverse Spine (Z, + Greater)", "3_12_ReverseSpine"),
            (TR2.build_313_chicken_wing,  "3.13 Chicken Wing",                 "3_13_ChickenWing"),
            (TR2.build_314_scooping,      "3.14 Scooping",                     "3_14_Scooping"),
            (TR2.build_315_reverse_c_finish,"3.15 Reverse C Finish",           "3_15_ReverseCFinish"),
        ]
        for fn, title, key in items:
            st.subheader(title)
            dfc = combine_pro_ama_table(fn, pro_arr, ama_arr, key_col=None)
            st.dataframe(
                _apply_2f(
                    style_with_table_key(key, dfc), dfc
                ),
                use_container_width=True
            )
            st.download_button(
                f"CSV ({title}·프로/일반 단일표)",
                dfc.to_csv(index=False).encode("utf-8-sig"),
                f"{key}_compare.csv",
                "text/csv"
            )
            st.divider()

    # ───────── All-in-One (미리보기 + 파일/마스터 내보내기) ─────────
    with tab_all:
        st.subheader("All-in-One (모든 비교표 한 번에)")

        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

        tables: dict[str, pd.DataFrame] = {}
        # Frame 라벨 표
        tables["FrontalBend"] = FB.build_frontal_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )
        tables["BodyHinge"] = BH.build_body_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )
        tables["LegHinge"] = LH.build_leg_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )
        tables["SideBend"] = SB.build_side_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )

        # TR/Lift (항목 라벨)
        tables["Thrust_Waist_X"]    = TR.build_compare_table(pro_arr, ama_arr)
        tables["Thrust_Shoulder_X"] = TR.build_shoulder_x_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        tables["Thrust_Head_X"]     = TR.build_head_x_compare(pro_arr, ama_arr)
        tables["Lift_Waist_Y"]      = TR.build_waist_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        tables["Lift_Shoulder_Y"]   = TR.build_shoulder_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        tables["Lift_Head_Y"]       = TR.build_head_y_compare(pro_arr, ama_arr)

        # OTT
        tables["OverTheTop"] = OTT.build_over_the_top_compare(
            pro_arr, ama_arr, frames=(4,5,6), chd_col="CN", wrist_r_col="BM"
        )

        # TR2(seg 라벨)
        tables["3_3_EarlyExtension(Waist Thrust(X, cm))"] = combine_pro_ama_table(
            (lambda a: TR2.build_33_early_extension(a, ctx["gs_pro_arr"]),
             lambda a: TR2.build_33_early_extension(a, ctx["gs_ama_arr"])),
            pro_arr, ama_arr
        )
        tables["3_4_FlatShoPlane"]    = combine_pro_ama_table(TR2.build_34_flat_sho_plane,    pro_arr, ama_arr)
        tables["3_5_FlyingElbow"]     = combine_pro_ama_table(TR2.build_35_flying_elbow,      pro_arr, ama_arr)
        tables["3_6_Sway"]            = combine_pro_ama_table(TR2.build_36_sway,              pro_arr, ama_arr)
        tables["3_7_Casting"]         = combine_pro_ama_table(TR2.build_37_casting,           pro_arr, ama_arr)
        tables["3_8_HangingBack"]     = combine_pro_ama_table(TR2.build_38_hanging_back,      pro_arr, ama_arr)
        tables["3_9_Slide"]           = combine_pro_ama_table(TR2.build_39_slide,             pro_arr, ama_arr)
        tables["3_10_Overswing_Y"]    = combine_pro_ama_table(TR2.build_310_overswing_y,      pro_arr, ama_arr)
        tables["3_11_CrossOver_X"]    = combine_pro_ama_table(TR2.build_311_cross_over_x,     pro_arr, ama_arr)
        tables["3_12_ReverseSpine"]   = combine_pro_ama_table(TR2.build_312_reverse_spine,    pro_arr, ama_arr)
        tables["3_13_ChickenWing"]    = combine_pro_ama_table(TR2.build_313_chicken_wing,     pro_arr, ama_arr)
        tables["3_14_Scooping"]       = combine_pro_ama_table(TR2.build_314_scooping,         pro_arr, ama_arr)
        tables["3_15_ReverseCFinish"] = combine_pro_ama_table(TR2.build_315_reverse_c_finish, pro_arr, ama_arr)

        # ✅ 현재 섹션 표들을 "마스터"에 등록 (세션 누적)
        register_section(META["id"], META["title"], tables)

        # 화면 프리뷰: 탭과 동일 인덱스 규칙 적용
# 화면 프리뷰: 탭과 동일 인덱스 규칙 적용
        # 🔧 expander 제거 → 바로 표시
        for key, df in tables.items():
            st.markdown(f"**{key}**")
            sty = style_with_table_key(key, df)
            st.dataframe(_apply_2f(sty, df), use_container_width=True)
            st.divider()


        # ── (A) 올인원 단독 Excel (원래 그대로)
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
            sheet_name = "All"
            cur_row = 0
            wb  = writer.book
            title_fmt  = wb.add_format({'bold': True, 'font_size': 12, 'align': 'left', 'valign': 'vcenter'})
            header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
            num_fmt    = wb.add_format({'num_format': '0.00'})

            pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            for name, df in tables.items():
                ws.write(cur_row, 0, str(name), title_fmt)
                cur_row += 1

                df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

                n_rows, n_cols = df.shape
                for c in range(n_cols):
                    ws.write(cur_row, c, df.columns[c], header_fmt)
                ws.set_column(0, n_cols - 1, 14, num_fmt)

                cur_row += n_rows + 1 + 2
            ws.freeze_panes(1, 0)

        xbuf.seek(0)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📦 Excel 다운로드 – All-in-One (단일 시트)",
            data=xbuf.getvalue(),
            file_name=f"swing_error_all_in_one_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # ── (B) Master Excel: ✅ All 시트 + 섹션별 시트들 한 파일에!
        m_buf = io.BytesIO()
        with pd.ExcelWriter(m_buf, engine="xlsxwriter") as writer:
            used_names: set[str] = set()

            # 1) All 시트 먼저 작성
            all_sheet = _safe_sheet("All", used_names)
            cur_row = 0
            wb  = writer.book
            title_fmt  = wb.add_format({'bold': True, 'font_size': 12, 'align': 'left', 'valign': 'vcenter'})
            header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
            num_fmt    = wb.add_format({'num_format': '0.00'})

            pd.DataFrame().to_excel(writer, sheet_name=all_sheet, index=False)
            ws_all = writer.sheets[all_sheet]

            for name, df in tables.items():
                ws_all.write(cur_row, 0, str(name), title_fmt)
                cur_row += 1

                df.to_excel(writer, sheet_name=all_sheet, startrow=cur_row, startcol=0, index=False, header=True)

                n_rows, n_cols = df.shape
                for c in range(n_cols):
                    ws_all.write(cur_row, c, df.columns[c], header_fmt)
                ws_all.set_column(0, n_cols - 1, 14, num_fmt)

                cur_row += n_rows + 1 + 2
            ws_all.freeze_panes(1, 0)

            # 2) 섹션별 시트들 추가
            for sec_id, meta in st.session_state["section_tables"].items():
                sheet = _safe_sheet(meta.get("title") or sec_id, used_names)
                _write_section_sheet(writer, sheet, meta["tables"])

        m_buf.seek(0)
        st.download_button(
            "📚 Master Excel 다운로드 (All 시트 + 섹션별 시트)",
            data=m_buf.getvalue(),
            file_name=f"master_with_all_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
