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



# 공용 헬퍼 (어딘가 공통 utils나 해당 main 위쪽에)
def _insert_index_column(df: pd.DataFrame, col_name: str = "seg") -> pd.DataFrame:
    out = df.copy()
    # 인덱스 → 문자열로 안전 변환 (MultiIndex도 처리)
    if isinstance(out.index, pd.MultiIndex):
        idx_series = out.index.map(lambda t: " | ".join(map(str, t)))
    else:
        idx_series = out.index.astype(str)
    # 맨 앞에 삽입 후, RangeIndex로 리셋
    out.insert(0, col_name, idx_series)
    out.reset_index(drop=True, inplace=True)
    return out

def combine_with_index(builder, pro_arr, ama_arr, idx_col: str = "seg") -> pd.DataFrame:
    df = combine_pro_ama_table(builder, pro_arr, ama_arr)
    return _insert_index_column(df, idx_col)


# utils/excel_export.py (같은 파일 상단에 둬도 OK)
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



# ── PDF Exporter ─────────────────────────────────────────────────────────
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm
import pandas as pd

def export_tables_pdf(tables: dict[str, pd.DataFrame],
                      title: str = "Swing Error – All-in-One",
                      landscape_mode: bool = True) -> io.BytesIO:
    """
    tables: {"표이름": DataFrame, ...}
    - 숫자 컬럼은 소수점 둘째자리 고정
    - 표 제목(H2), 헤더 회색, 그리드, 페이지 자동 분할
    """
    buf = io.BytesIO()
    page_size = landscape(A4) if landscape_mode else A4

    doc = SimpleDocTemplate(
        buf, pagesize=page_size,
        leftMargin=12*mm, rightMargin=12*mm, topMargin=12*mm, bottomMargin=12*mm
    )
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles['Title']), Spacer(1, 6)]

    for name, df in tables.items():
        # 제목
        story.append(Paragraph(str(name), styles['Heading2']))
        story.append(Spacer(1, 4))

        # 숫자 2f 포맷 적용본
        df2 = df.copy()
        for c in df2.columns:
            if pd.api.types.is_numeric_dtype(df2[c]):
                df2[c] = df2[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

        data = [list(df2.columns)] + df2.fillna("").astype(str).values.tolist()

        # 폭: 글자수 기반 대략치(가독성용), [18mm, 60mm] 사이로 제한
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
        story.append(Spacer(1, 10))  # 표 간 여백

    doc.build(story)
    buf.seek(0)
    return buf



META = {"id": "swing_error", "title": "3. Swing Error", "icon": "⚠️", "order": 16}
def get_metadata(): return META


def combine_pro_ama_table(builder, pro_arr, ama_arr, *, key_col: str | None = None):
    # 1) 프로/일반 테이블 생성 (builder가 (pro, ama) 튜플일 수도 있음)
    if isinstance(builder, tuple):
        build_p, build_a = builder
        df_p = build_p(pro_arr)
        df_a = build_a(ama_arr)
    else:
        df_p = builder(pro_arr)
        df_a = builder(ama_arr)

    # 2) seg 보장: 없으면 '검사명'→seg, 그것도 없으면 첫 컬럼으로 seg 생성
    def _ensure_seg(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "seg" not in df.columns:
            base = "검사명" if "검사명" in df.columns else df.columns[0]
            df.insert(0, "seg", df[base].astype(str))
        # ✅ 여기서부터 표에 '검사명'이 안 들어가도록 제거
        df = df.drop(columns=["검사명"], errors="ignore")
        return df

    df_p = _ensure_seg(df_p)
    df_a = _ensure_seg(df_a)

    # 3) 조인 키 결정 (무조건 seg 우선)
    if key_col is None:
        key_col = "seg"

    # 4) 인덱스 조인 (seg 컬럼은 남겨두고, 인덱스로도 사용)
    p = df_p.set_index(key_col, drop=False)
    a = df_a.set_index(key_col, drop=False)

    out = p.join(a, lsuffix="_프로", rsuffix="_일반", how="outer")

    # 5) 조인 후 seg를 인덱스에서 확실히 복원하고, 잡스러운 seg_* / 검사명 잔여 제거
    out.insert(0, "seg", out.index.astype(str))
    # seg 변형 컬럼/검사명 류는 싹 제거
    cols_to_drop = [c for c in out.columns if c.startswith("seg_") or c.endswith("검사명")]
    out = out.drop(columns=cols_to_drop, errors="ignore")

    # 6) 인덱스 리셋 + seg 맨 앞 고정
    out.reset_index(drop=True, inplace=True)
    cols = ["seg"] + [c for c in out.columns if c != "seg"]
    return out[cols]





def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("상단 메인앱에서 프로/일반 엑셀을 업로드하면 여기서 자동으로 비교가 실행됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]


    (tab_fb, tab_bh, tab_lh, tab_sb, tab_TR, tab_ott,tab_TR2, tab_all) = st.tabs(
        ["Frontal Bend", "Body Hinge", "Leg Hinge", "Side Bend", "Trust", "Over The Top","Trust2", "전체 비교표"]
    )


    with tab_fb:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]
        df = FB.build_frontal_bend_compare(pro_arr, ama_arr, start=1, end=10, labels=labels,
                                        pro_name="프로", ama_name="일반")
        st.dataframe(
            df.style.format({
                "프로 Frontal Bend (deg)":"{:.2f}",
                "프로 Section Change (deg)":"{:+.2f}",
                "일반 Frontal Bend (deg)":"{:.2f}",
                "일반 Section Change (deg)":"{:+.2f}",
                "Frontal Bend Δ(프로-일반)":"{:+.2f}",
                "Section Change Δ(프로-일반)":"{:+.2f}",
            }),
            use_container_width=True,
        )

    
    with tab_bh:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

        cmp = BH.build_body_hinge_compare(pro_arr, ama_arr, start=1, end=10, labels=labels,
                                        pro_name="프로", ama_name="일반")

        st.dataframe(
            cmp.style.format({
                "프로 Body Hinge (deg)":"{:.2f}",
                "프로 Section Change (deg)":"{:.2f}",
                "일반 Body Hinge (deg)":"{:.2f}",
                "일반 Section Change (deg)":"{:.2f}",
                "Body Hinge Δ(프로-일반)":"{:+.2f}",
                "Section Change Δ(프로-일반)":"{:+.2f}",
            }),
            use_container_width=True
        )

        st.download_button(
            "CSV (Body Hinge 비교표)",
            data=cmp.to_csv(index=False).encode("utf-8-sig"),
            file_name="body_hinge_compare.csv",
            mime="text/csv"
        )
    with tab_lh:
        # (선택) 프레임 라벨
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

        # ── 프로/일반 단일 리포트(프레임 1~10 + 섹션 합계) ──
        p_rep = LH.build_leg_hinge_report(pro_arr, start=1, end=10, labels=labels)
        a_rep = LH.build_leg_hinge_report(ama_arr, start=1, end=10, labels=labels)

        st.subheader("Leg Hinge")

        cmp_lh = LH.build_leg_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels,
            pro_name="프로", ama_name="일반"
        )
        st.dataframe(
            cmp_lh.style.format({
                "프로 Leg Hinge (deg)":"{:.2f}",
                "프로 Section Change (deg)":"{:+.2f}",
                "일반 Leg Hinge (deg)":"{:.2f}",
                "일반 Section Change (deg)":"{:+.2f}",
                "Leg Hinge Δ(프로-일반)":"{:+.2f}",
                "Section Change Δ(프로-일반)":"{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV (Leg Hinge 비교표)",
            data=cmp_lh.to_csv(index=False).encode("utf-8-sig"),
            file_name="leg_hinge_compare.csv",
            mime="text/csv"
        )


    with tab_sb:
        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]  # 선택

        # 프로/일반 리포트
        p_rep = SB.build_side_bend_report(pro_arr, start=1, end=10, labels=labels)
        a_rep = SB.build_side_bend_report(ama_arr, start=1, end=10, labels=labels)

        # c1, c2 = st.columns(2)
        # with c1:
        #     st.caption("프로")
        #     st.dataframe(
        #         p_rep.style.format({
        #             "Side Bend (deg)": "{:.2f}",
        #             "Section Change (deg)": "{:+.2f}",
        #         }),
        #         use_container_width=True
        #     )
        #     st.download_button(
        #         "CSV (프로 Side Bend 리포트)",
        #         data=p_rep.to_csv(index=False).encode("utf-8-sig"),
        #         file_name="side_bend_report_pro.csv",
        #         mime="text/csv"
        #     )

        # with c2:
        #     st.caption("일반")
        #     st.dataframe(
        #         a_rep.style.format({
        #             "Side Bend (deg)": "{:.2f}",
        #             "Section Change (deg)": "{:+.2f}",
        #         }),
        #         use_container_width=True
        #     )
        #     st.download_button(
        #         "CSV (일반 Side Bend 리포트)",
        #         data=a_rep.to_csv(index=False).encode("utf-8-sig"),
        #         file_name="side_bend_report_ama.csv",
        #         mime="text/csv"
        #     )

        # st.divider()
        st.subheader("Side Bend")
        cmp_sb = SB.build_side_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels,
            pro_name="프로", ama_name="일반"
        )
        st.dataframe(
            cmp_sb.style.format({
                "프로 Side Bend (deg)":"{:.2f}",
                "프로 Section Change (deg)":"{:+.2f}",
                "일반 Side Bend (deg)":"{:.2f}",
                "일반 Section Change (deg)":"{:+.2f}",
                "Side Bend Δ(프로-일반)":"{:+.2f}",
                "Section Change Δ(프로-일반)":"{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV (Side Bend 비교표)",
            data=cmp_sb.to_csv(index=False).encode("utf-8-sig"),
            file_name="side_bend_compare.csv",
            mime="text/csv"
        )

    with tab_TR:
        st.subheader("Thrust(X, cm)")
        st.subheader("3.1.6.1 Waist")
        df_px = TR.build_compare_table(pro_arr, ama_arr)
        st.dataframe(df_px.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Pelvis X Shift 비교)",
                        data=df_px.to_csv(index=False).encode("utf-8-sig"),
                        file_name="pelvis_x_shift_compare.csv", mime="text/csv")


        st.divider()
        st.subheader("3.1.6.2 Shoulder")
        df_sx = TR.build_shoulder_x_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])

        st.dataframe(df_sx.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Shoulder X Center 비교)",
                        data=df_sx.to_csv(index=False).encode("utf-8-sig"),
                        file_name="shoulder_x_center_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.6.3 Head")
        df_hx = TR.build_head_x_compare(pro_arr, ama_arr)
        st.dataframe(df_hx.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Head X 비교)",
                        data=df_hx.to_csv(index=False).encode("utf-8-sig"),
                        file_name="head_x_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.1 Waist Lift (Y, cm)")
        df_wy = TR.build_waist_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        st.dataframe(df_wy.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Waist Lift Y 비교)",
                        data=df_wy.to_csv(index=False).encode("utf-8-sig"),
                        file_name="waist_lift_y_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.2 Shoulder Lift (Y, cm)")
        df_sy = TR.build_shoulder_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        st.dataframe(df_sy.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Shoulder Lift Y 비교)",
                        data=df_sy.to_csv(index=False).encode("utf-8-sig"),
                        file_name="shoulder_lift_y_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.3 Head (Y)")
        df_hy = TR.build_head_y_compare(pro_arr, ama_arr)
        st.dataframe(df_hy.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Head Y 비교)",
                        data=df_hy.to_csv(index=False).encode("utf-8-sig"),
                        file_name="head_y_compare.csv", mime="text/csv")

    
    with tab_ott:
        st.caption("프레임: 4, 5, 6 기준 (방법 1/2 포함)")
        cmp = OTT.build_over_the_top_compare(pro_arr, ama_arr, frames=(4,5,6), chd_col="CN", wrist_r_col="BM")
        st.dataframe(
            cmp.style.format({"프로":"{:.2f}", "일반":"{:.2f}", "차이(프로-일반)":"{:+.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "CSV (Over The Top 비교)",
            data=cmp.to_csv(index=False).encode("utf-8-sig"),
            file_name="over_the_top_compare.csv",
            mime="text/csv"
        )

        # st.divider()
        # st.caption("프로/일반 단일표 (참고)")
        # c1, c2 = st.columns(2)
        # with c1:
        #     st.caption("프로")
        #     df_p = OTT.build_over_the_top_table(pro_arr)
        #     st.dataframe(df_p.style.format({"값":"{:.2f}"}), use_container_width=True)
        # with c2:
        #     st.caption("일반")
        #     df_a = OTT.build_over_the_top_table(ama_arr)
        #     st.dataframe(df_a.style.format({"값":"{:.2f}"}), use_container_width=True)

    with tab_TR2:
        st.subheader("3.3 Early Extension (Waist Thrust X)")
        df = combine_pro_ama_table(
            (
                lambda a: TR2.build_33_early_extension(a, ctx["gs_pro_arr"]),
                lambda a: TR2.build_33_early_extension(a, ctx["gs_ama_arr"]),
            ),
            pro_arr, ama_arr, key_col=None
        )

        st.dataframe(
            df.style.format({
                "프로": "{:.2f}", "일반": "{:.2f}",
                "차(프로-일반)": "{:+.2f}", 
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV (3.3·프로/일반 단일표)",
            df.to_csv(index=False).encode("utf-8-sig"),
            "3_3_early_extension_compare.csv",
            "text/csv"
        )

        st.divider()

        items = [
            
            (TR2.build_34_flat_sho_plane, "3.4 Flat Sho Plane",               "3_4_flat_sho_plane"),
            (TR2.build_35_flying_elbow,   "3.5 Flying Elbow",                  "3_5_flying_elbow"),
            (TR2.build_36_sway,           "3.6 Sway",                          "3_6_sway"),
            (TR2.build_37_casting,        "3.7 Casting",                       "3_7_casting"),
            (TR2.build_38_hanging_back,   "3.8 Hanging Back (Z, − Greater)",   "3_8_hanging_back"),
            (TR2.build_39_slide,          "3.9 Slide (Z, + Greater)",          "3_9_slide"),
            (TR2.build_310_overswing_y,   "3.10 Overswing (Y, − Greater)",     "3_10_overswing"),
            (TR2.build_311_cross_over_x,  "3.11 Cross Over (X, − Greater)",    "3_11_cross_over"),
            (TR2.build_312_reverse_spine, "3.12 Reverse Spine (Z, + Greater)", "3_12_reverse_spine"),
            (TR2.build_313_chicken_wing,  "3.13 Chicken Wing",                 "3_13_chicken_wing"),
            (TR2.build_314_scooping,      "3.14 Scooping",                     "3_14_scooping"),
            (TR2.build_315_reverse_c_finish,"3.15 Reverse C Finish",           "3_15_reverse_c_finish"),
        ]

        for fn, title, fname in items:
            st.subheader(title)
            dfc = combine_pro_ama_table(fn, pro_arr, ama_arr, key_col=None)
            st.dataframe(
                dfc.style.format({
                "프로": "{:.2f}", "일반": "{:.2f}",
                "차(프로-일반)": "{:+.2f}", 
            }),
                use_container_width=True
            )
            st.download_button(
                f"CSV ({title}·프로/일반 단일표)",
                dfc.to_csv(index=False).encode("utf-8-sig"),
                f"{fname}_compare.csv",
                "text/csv"
            )
            st.divider()



    with tab_all:
        st.subheader("All-in-One (모든 비교표 한 번에)")

        # 통일 포맷 헬퍼: 숫자만 {:.2f}
        def style_2f(df: pd.DataFrame):
            fmt = {c: "{:.2f}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
            return df.style.format(fmt)

        labels = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","FIN"]

        # 1) 모든 표를 '평면 dict'로 모으기 (카테고리 없이 한번에)
        tables: dict[str, pd.DataFrame] = {}

        # Frontal Bend
        tables["FrontalBend"] = FB.build_frontal_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )

        # Body / Leg / Side
        tables["BodyHinge"] = BH.build_body_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )
        tables["LegHinge"] = LH.build_leg_hinge_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )
        tables["SideBend"] = SB.build_side_bend_compare(
            pro_arr, ama_arr, start=1, end=10, labels=labels, pro_name="프로", ama_name="일반"
        )

        # Thrust / Lift
        tables["Thrust_Waist_X"]    = TR.build_compare_table(pro_arr, ama_arr)            # 3.1.6.1
        tables["Thrust_Shoulder_X"] = TR.build_shoulder_x_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        tables["Thrust_Head_X"]     = TR.build_head_x_compare(pro_arr, ama_arr)           # 3.1.6.3
        tables["Lift_Waist_Y"]      = TR.build_waist_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        tables["Lift_Shoulder_Y"]   = TR.build_shoulder_lifty_compare(pro_arr, ama_arr, ctx["gs_pro_arr"], ctx["gs_ama_arr"])
        tables["Lift_Head_Y"]       = TR.build_head_y_compare(pro_arr, ama_arr)           # 3.1.7.3

        # Over The Top
        tables["OverTheTop"] = OTT.build_over_the_top_compare(
            pro_arr, ama_arr, frames=(4,5,6), chd_col="CN", wrist_r_col="BM"
        )

        # Trust2 (3.3~3.15) — 3.3은 GS 필요 → tuple 빌더로 주입
        tables["3_3_EarlyExtension(Waist Thrust(X, cm))"] = combine_pro_ama_table(
            (lambda a: TR2.build_33_early_extension(a, ctx["gs_pro_arr"]),
            lambda a: TR2.build_33_early_extension(a, ctx["gs_ama_arr"])),
            pro_arr, ama_arr
        )
        tables["3_4_FlatShoPlane"]    = combine_pro_ama_table(TR2.build_34_flat_sho_plane, pro_arr, ama_arr)
        tables["3_5_FlyingElbow"]     = combine_pro_ama_table(TR2.build_35_flying_elbow,   pro_arr, ama_arr)
        tables["3_6_Sway"]            = combine_pro_ama_table(TR2.build_36_sway,           pro_arr, ama_arr)
        tables["3_7_Casting"]         = combine_pro_ama_table(TR2.build_37_casting,        pro_arr, ama_arr)
        tables["3_8_HangingBack"]     = combine_pro_ama_table(TR2.build_38_hanging_back,   pro_arr, ama_arr)
        tables["3_9_Slide"]           = combine_pro_ama_table(TR2.build_39_slide,          pro_arr, ama_arr)
        tables["3_10_Overswing_Y"]    = combine_pro_ama_table(TR2.build_310_overswing_y,   pro_arr, ama_arr)
        tables["3_11_CrossOver_X"]    = combine_pro_ama_table(TR2.build_311_cross_over_x,  pro_arr, ama_arr)
        tables["3_12_ReverseSpine"]   = combine_pro_ama_table(TR2.build_312_reverse_spine, pro_arr, ama_arr)
        tables["3_13_ChickenWing"]    = combine_pro_ama_table(TR2.build_313_chicken_wing,  pro_arr, ama_arr)
        tables["3_14_Scooping"]       = combine_pro_ama_table(TR2.build_314_scooping,      pro_arr, ama_arr)
        tables["3_15_ReverseCFinish"] = combine_pro_ama_table(TR2.build_315_reverse_c_finish, pro_arr, ama_arr)

        register_section(META["id"], META["title"], tables)

        # (옵션) 화면에서 한두 개 프리뷰
        with st.expander("전체 표", expanded=False):
            preview_keys = list(tables.keys()) #전부
            for k in preview_keys:
                st.markdown(f"**{k}**")
                st.dataframe(style_2f(tables[k]), use_container_width=True)
                st.divider()

        # 2) 안전한 시트명 생성기: 금지문자 제거/치환 + 31자 제한 + 중복 고유화
        import re, io
        from datetime import datetime

        def _safe_sheet_name(name: str, used: set[str]) -> str:
            # 금지문자: \ / ? * [ ] : ' " (따옴표류도 제거)
            bad = r'[\\/\?\*\[\]\:\'"]'
            s = re.sub(bad, '', name)          # 전부 제거
            s = s.replace(' ', '_')            # 공백은 언더스코어
            s = s[:31] if len(s) > 31 else s   # 길이 제한
            if not s: s = "Sheet"
            base = s
            i = 1
            while s in used:
                # 접미사 붙일 자리 확보(최대 31자)
                suffix = f"_{i}"
                cut = 31 - len(suffix)
                s = (base[:cut] if len(base) > cut else base) + suffix
                i += 1
            used.add(s)
            return s

                # 3) 엑셀로 한번에 내보내기 (단일 시트: All)
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
            sheet_name = "All"
            # 시트 생성
            # 첫 행/열 위치
            cur_row = 0
            # 서식 준비
            wb  = writer.book
            title_fmt = wb.add_format({
                'bold': True, 'font_size': 12, 'align': 'left', 'valign': 'vcenter'
            })
            header_fmt = wb.add_format({
                'bold': True, 'bg_color': '#F2F2F2', 'border': 1
            })
            num_fmt = wb.add_format({'num_format': '0.00'})
            sep_fmt = wb.add_format({'bg_color': '#FFFFFF'})
            # 시트 객체 얻기 위해 우선 빈 DF 한 번 써두고 바로 덮어씀
            pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            # 표들을 하나의 시트에 순서대로 기록
            for name, df in tables.items():
                # 3-1) 제목 라인
                ws.write(cur_row, 0, str(name), title_fmt)
                cur_row += 1

                # 3-2) 데이터프레임 기록 (헤더 서식 지정)
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=cur_row,
                    startcol=0,
                    index=False,
                    header=True
                )

                # 3-3) 숫자열 0.00 포맷, 열 너비
                n_rows, n_cols = df.shape
                # 헤더 서식
                for c in range(n_cols):
                    ws.write(cur_row, c, df.columns[c], header_fmt)
                # 본문 서식 + 너비
                # 간단히 모든 열 너비 14로(필요시 문자열 길이 기반 자동화 가능)
                ws.set_column(0, n_cols - 1, 14, num_fmt)

                # 3-4) 다음 표 시작 위치: (데이터 n_rows) + (헤더 1) + 공백 2줄
                cur_row += n_rows + 1 + 2

            # 보기 편하게 맨 위 고정(제목들만이라 큰 의미 없지만 유지)
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

        pdf_buf = export_tables_pdf(tables, title="Swing Error – All-in-One", landscape_mode=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "🧾 PDF 다운로드 – All-in-One",
            data=pdf_buf.getvalue(),
            file_name=f"swing_error_all_in_one_{stamp}.pdf",
            mime="application/pdf",
            use_container_width=True
)