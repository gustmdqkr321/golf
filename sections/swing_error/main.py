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

META = {"id": "swing_error", "title": "Swing Error", "icon": "⚠️", "order": 16}
def get_metadata(): return META


def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("상단 메인앱에서 프로/일반 엑셀을 업로드하면 여기서 자동으로 비교가 실행됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]


    (tab_fb, tab_bh, tab_lh, tab_sb, tab_TR, tab_ott,tab_TR2) = st.tabs(
        ["Frontal Bend", "Body Hinge", "Leg Hinge", "Side Bend", "Trust", "Over The Top","Trust2"]
    )


    with tab_fb:


        # 프로/일반 단일표
        p_rep = FB.build_fb_report_table(pro_arr, start=1, end=10)
        a_rep = FB.build_fb_report_table(ama_arr, start=1, end=10)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("프로")
            st.dataframe(
                p_rep.style.format({
                    "Frontal Bend (deg)": "{:+.2f}",
                    "Frontal Bend Section (deg)": "{:+.2f}",
                }),
                use_container_width=True
            )
            st.download_button(
                "CSV (프로 Frontal Bend 리포트)",
                data=p_rep.to_csv(index=False).encode("utf-8-sig"),
                file_name="frontal_bend_report_pro.csv",
                mime="text/csv"
            )

        with c2:
            st.caption("일반")
            st.dataframe(
                a_rep.style.format({
                    "Frontal Bend (deg)": "{:+.2f}",
                    "Frontal Bend Section (deg)": "{:+.2f}",
                }),
                use_container_width=True
            )
            st.download_button(
                "CSV (일반 Frontal Bend 리포트)",
                data=a_rep.to_csv(index=False).encode("utf-8-sig"),
                file_name="frontal_bend_report_ama.csv",
                mime="text/csv"
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

        c1, c2 = st.columns(2)
        with c1:
            st.caption("프로")
            st.dataframe(
                p_rep.style.format({
                    "Leg Hinge (deg)": "{:.2f}",
                    "Section Change (deg)": "{:+.2f}",
                }),
                use_container_width=True
            )
            st.download_button(
                "CSV (프로 Leg Hinge 리포트)",
                data=p_rep.to_csv(index=False).encode("utf-8-sig"),
                file_name="leg_hinge_report_pro.csv",
                mime="text/csv"
            )

        with c2:
            st.caption("일반")
            st.dataframe(
                a_rep.style.format({
                    "Leg Hinge (deg)": "{:.2f}",
                    "Section Change (deg)": "{:+.2f}",
                }),
                use_container_width=True
            )
            st.download_button(
                "CSV (일반 Leg Hinge 리포트)",
                data=a_rep.to_csv(index=False).encode("utf-8-sig"),
                file_name="leg_hinge_report_ama.csv",
                mime="text/csv"
            )

        st.divider()
        st.subheader("프로 vs 일반 비교")

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

        c1, c2 = st.columns(2)
        with c1:
            st.caption("프로")
            st.dataframe(
                p_rep.style.format({
                    "Side Bend (deg)": "{:.2f}",
                    "Section Change (deg)": "{:+.2f}",
                }),
                use_container_width=True
            )
            st.download_button(
                "CSV (프로 Side Bend 리포트)",
                data=p_rep.to_csv(index=False).encode("utf-8-sig"),
                file_name="side_bend_report_pro.csv",
                mime="text/csv"
            )

        with c2:
            st.caption("일반")
            st.dataframe(
                a_rep.style.format({
                    "Side Bend (deg)": "{:.2f}",
                    "Section Change (deg)": "{:+.2f}",
                }),
                use_container_width=True
            )
            st.download_button(
                "CSV (일반 Side Bend 리포트)",
                data=a_rep.to_csv(index=False).encode("utf-8-sig"),
                file_name="side_bend_report_ama.csv",
                mime="text/csv"
            )

        st.divider()
        st.subheader("프로 vs 일반 비교")
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
        st.subheader("Pelvis X Shift 표")
        st.subheader("3.1.6.1")
        df_px = TR.build_compare_table(pro_arr, ama_arr)
        st.dataframe(df_px.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Pelvis X Shift 비교)",
                        data=df_px.to_csv(index=False).encode("utf-8-sig"),
                        file_name="pelvis_x_shift_compare.csv", mime="text/csv")


        st.divider()
        st.subheader("3.1.6.2 Shoulder (X Center)")
        df_sx = TR.build_shoulder_x_compare(pro_arr, ama_arr)
        st.dataframe(df_sx.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Shoulder X Center 비교)",
                        data=df_sx.to_csv(index=False).encode("utf-8-sig"),
                        file_name="shoulder_x_center_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.6.3 Head (X)")
        df_hx = TR.build_head_x_compare(pro_arr, ama_arr)
        st.dataframe(df_hx.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Head X 비교)",
                        data=df_hx.to_csv(index=False).encode("utf-8-sig"),
                        file_name="head_x_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.1 Waist Lift (Y, cm)")
        df_wy = TR.build_waist_lifty_compare(pro_arr, ama_arr)
        st.dataframe(df_wy.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (Waist Lift Y 비교)",
                        data=df_wy.to_csv(index=False).encode("utf-8-sig"),
                        file_name="waist_lift_y_compare.csv", mime="text/csv")

        st.divider()
        st.subheader("3.1.7.2 Shoulder Lift (Y, cm)")
        df_sy = TR.build_shoulder_lifty_compare(pro_arr, ama_arr)
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

        st.divider()
        st.caption("프로/일반 단일표 (참고)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("프로")
            df_p = OTT.build_over_the_top_table(pro_arr)
            st.dataframe(df_p.style.format({"값":"{:.2f}"}), use_container_width=True)
        with c2:
            st.caption("일반")
            df_a = OTT.build_over_the_top_table(ama_arr)
            st.dataframe(df_a.style.format({"값":"{:.2f}"}), use_container_width=True)

    with tab_TR2:
        st.subheader("3.3 Early Extension (Waist Thrust X)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("프로")
            df = TR2.build_33_early_extension(pro_arr)
            st.dataframe(df.style.format({"현재결과":"{:.2f}","종전결과":"{:.2f}"}), use_container_width=True)
            st.download_button("CSV (3.3·프로)", df.to_csv(index=False).encode("utf-8-sig"),
                            "3_3_early_extension_pro.csv", "text/csv")
        with c2:
            st.caption("일반")
            df = TR2.build_33_early_extension(ama_arr)
            st.dataframe(df.style.format({"현재결과":"{:.2f}","종전결과":"{:.2f}"}), use_container_width=True)
            st.download_button("CSV (3.3·일반)", df.to_csv(index=False).encode("utf-8-sig"),
                            "3_3_early_extension_ama.csv", "text/csv")

        st.divider()
        st.subheader("3.4 Flat Sho Plane")
        for fn, title, fname in [
            (TR2.build_34_flat_sho_plane, "3.4 Flat Sho Plane", "3_4_flat_sho_plane"),
            (TR2.build_35_flying_elbow,   "3.5 Flying Elbow",   "3_5_flying_elbow"),
            (TR2.build_36_sway,           "3.6 Sway",           "3_6_sway"),
            (TR2.build_37_casting,        "3.7 Casting",        "3_7_casting"),
            (TR2.build_38_hanging_back,   "3.8 Hanging Back (Z, − Greater)", "3_8_hanging_back"),
            (TR2.build_39_slide,          "3.9 Slide (Z, + Greater)", "3_9_slide"),
            (TR2.build_310_overswing_y,   "3.10 Overswing (Y, − Greater)", "3_10_overswing"),
            (TR2.build_311_cross_over_x,  "3.11 Cross Over (X, − Greater)", "3_11_cross_over"),
            (TR2.build_312_reverse_spine, "3.12 Reverse Spine (Z, + Greater)", "3_12_reverse_spine"),
            (TR2.build_313_chicken_wing,  "3.13 Chicken Wing",  "3_13_chicken_wing"),
            (TR2.build_314_scooping,      "3.14 Scooping",      "3_14_scooping"),
            (TR2.build_315_reverse_c_finish,"3.15 Reverse C Finish","3_15_reverse_c_finish"),
        ]:
            st.subheader(title)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("프로")
                dfp = fn(pro_arr)
                st.dataframe(dfp.style.format({"현재결과":"{:.2f}","종전결과":"{:.2f}"}), use_container_width=True)
                st.download_button(f"CSV ({title}·프로)",
                                data=dfp.to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"{fname}_pro.csv",
                                mime="text/csv")
            with c2:
                st.caption("일반")
                dfa = fn(ama_arr)
                st.dataframe(dfa.style.format({"현재결과":"{:.2f}","종전결과":"{:.2f}"}), use_container_width=True)
                st.download_button(f"CSV ({title}·일반)",
                                data=dfa.to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"{fname}_ama.csv",
                                mime="text/csv")
            st.divider()
