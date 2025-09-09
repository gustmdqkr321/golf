# sections/swing/main.py
from pathlib import Path
import streamlit as st
import pandas as pd

from .features import _1hand_hight as hand
from .features import _2swing_tempo as swing
from .features import _3body_arm as fc   # ← 같은 섹션(features)에서 import
from .features import _4center as rot
from .features import _6arc as rasi
from .features import _7takeback as wri_chd
from .features import _8top as top
from .features import _9top2 as top2
from .features import _10sho_turn as sho_turn
from .features import _11x_factor as xfac
from .features import _12club_head as chd
from .features import _13cocking as coc
from .features import _14lean as lean
from .features import _15side_bend as bend
from .features import _16ankle as ank
from .features import _17opn as opn
from .features import _18_chd_clo as clo
from .features import _19to23 as t1923

META = {"id": "swing", "title": "스윙 비교", "icon": "🏌️", "order": 10}
def get_metadata(): return META

def _to_num(x):
    if isinstance(x, str):
        x = x.replace("❗", "").strip()
    try:
        return float(x)
    except Exception:
        return float("nan")

def _get_last_xyz(df_main: pd.DataFrame, cols: list[str]) -> tuple[float,float,float]:
    """메인표에서 '1-9' 행(없으면 마지막 행)의 X/Y/Z 숫자 추출"""
    if "Frame" in df_main.columns:
        target = df_main[df_main["Frame"] == "1-9"]
        row = (target.iloc[0] if not target.empty else df_main.tail(1).iloc[0])
    else:
        row = df_main.tail(1).iloc[0]
    return _to_num(row[cols[0]]), _to_num(row[cols[1]]), _to_num(row[cols[2]])

def run(ctx=None):
    st.subheader(f"{META['icon']} {META['title']}")

    if ctx is None or ctx.get("pro_arr") is None or ctx.get("ama_arr") is None:
        st.info("상단 메인앱에서 프로/일반 엑셀을 업로드하면 여기서 자동으로 비교가 실행됩니다.")
        return

    pro_arr = ctx["pro_arr"]
    ama_arr = ctx["ama_arr"]




    ###
    # 새 탭 추가: 📋 비율 표
    ###
    tab1, tab2, tab3, tab4, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs(["손높이", "스윙 템포", "비율 표", "중심", "아크", "테이크백", "top", "cocking","lean","side bend","ankle","opn","chd clo","19-23"])







    # ── 탭 1: 손높이/각도 비교 ────────────────────────────────────────────────
    with tab1:
        row = st.number_input("계산 행 번호", min_value=1, value=4, step=1, key="swing_row")
        pro = hand.compute_metrics(pro_arr, row=row)
        ama = hand.compute_metrics(ama_arr, row=row)
        df  = hand.build_compare_df(pro, ama)
        st.dataframe(df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:.2f}"}),
                     use_container_width=True)
        st.markdown(f"- **프로 분류:** `{pro['class']}` &nbsp;&nbsp; "
                    f"- **일반 분류:** `{ama['class']}`")
        st.download_button("CSV 다운로드(손높이/각도)",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hand_compare.csv", mime="text/csv", key="dl_hand")

    # ── 탭 2: 템포/리듬 ───────────────────────────────────────────────────────
    with tab2:
        colA, colB, colC, colD = st.columns(4)
        # with colA:
        #     tempo_std = st.number_input("템포 표준(프로, 2번)", value=1.14, step=0.01, format="%.2f")
        # with colB:
        #     tempo_tol = st.number_input("템포 middle 허용오차", value=0.05, step=0.01, format="%.2f")
        # with colC:
        #     rhythm_std = st.number_input("리듬 표준(프로, 3번)", value=2.80, step=0.05, format="%.2f")
        # with colD:
        #     rhythm_tol = st.number_input("리듬 middle 허용오차", value=0.20, step=0.01, format="%.2f")

        pro_m = swing.compute_tempo_rhythm(
            pro_arr
        )
        ama_m = swing.compute_tempo_rhythm(
            ama_arr
        )

        cmp_df = swing.build_tempo_rhythm_compare(pro_m, ama_m)
        st.dataframe(cmp_df.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:.2f}"}),
                     use_container_width=True)

        st.markdown(
            f"- **프로 스타일:** 템포=`{pro_m['tempo_style']}`, 리듬=`{pro_m['rhythm_style']}`  \n"
            f"- **일반 스타일:** 템포=`{ama_m['tempo_style']}`, 리듬=`{ama_m['rhythm_style']}`"
        )

        with st.expander("상세: 프로/일반 단일표 보기"):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("프로")
                st.dataframe(swing.build_tempo_rhythm_table(pro_m).style.format({"값": "{:.2f}"}),
                             use_container_width=True)
            with c2:
                st.caption("일반")
                st.dataframe(swing.build_tempo_rhythm_table(ama_m).style.format({"값": "{:.2f}"}),
                             use_container_width=True)

        st.download_button("CSV 다운로드(템포·리듬)",
                           data=cmp_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="tempo_rhythm_compare.csv", mime="text/csv", key="dl_tempo")

   # ── 탭 3: 📋 비율 표 — 15행(1–9) X+Y+Z 합의 분포로 프로/일반 비율과 차이 표시 ──
    with tab3:
        # 4부위
        parts = [("knee","KNEE"), ("pelvis","WAIST"), ("shoulder","SHOULDER"), ("wrist","WRIST")]

        # 각 부위의 1–9(X+Y+Z 합) 가져오는 헬퍼
        def part_sum(arr_pro, arr_ama, part_key, colset):
            res = fc.build_all_tables(arr_pro, arr_ama, part=part_key, mass=60.0, summary_mode="mean")
            # 메인표의 '1-9' 행에서 X/Y/Z 추출
            def _to_num(x):
                if isinstance(x, str): x = x.replace("❗", "").strip()
                try: return float(x)
                except: return float("nan")
            df = res.table_main
            row = df[df["Frame"] == "1-9"].iloc[0] if "1-9" in df["Frame"].values else df.tail(1).iloc[0]
            x = _to_num(row[colset[0]]) or 0.0
            y = _to_num(row[colset[1]]) or 0.0
            z = _to_num(row[colset[2]]) or 0.0
            return x + y + z

        # 프로/일반 합계 → 비율(%)
        sums_pro, sums_ama = {}, {}
        for key, _ in parts:
            sums_pro[key] = part_sum(pro_arr, ama_arr, key, ["Rory_X","Rory_Y","Rory_Z"])
            sums_ama[key] = part_sum(pro_arr, ama_arr, key, ["Hong_X","Hong_Y","Hong_Z"])

        def to_ratio(d: dict) -> dict:
            tot = sum(d.values())
            return {k: (0.0 if tot == 0 else v/tot*100.0) for k, v in d.items()}

        ratio_pro = to_ratio(sums_pro)
        ratio_ama = to_ratio(sums_ama)

        # 표 구성: 프로/일반 나란히 + 차이(프로-일반)
        rows = []
        for key, label in parts:
            p = round(ratio_pro.get(key, 0.0), 1)
            a = round(ratio_ama.get(key, 0.0), 1)
            d = round(p - a, 1)
            rows.append([label, p, a, d, abs(d)])

        table = pd.DataFrame(rows, columns=["", "프로(%)", "일반(%)", "차이(프로-일반)", "_abs"])
        # 보기 좋게: 절대차이 기준 정렬 옵션
        sort_by_diff = st.checkbox("차이 큰 순으로 정렬", value=True)
        if sort_by_diff:
            table = table.sort_values("_abs", ascending=False, ignore_index=True)
        table = table.drop(columns="_abs")

        st.dataframe(table.style.format({"프로(%)":"{:.1f}", "일반(%)":"{:.1f}", "차이(프로-일반)":"{:.1f}"}),
                     use_container_width=True)

        st.download_button(
            "CSV 다운로드(프로·일반 비율표)",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name="ratio_pro_vs_ama.csv",
            mime="text/csv",
        )

    # 탭 4: 🔄 회전 각도 — 골반/어깨 수평·수직 회전각 (프로/일반)
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("회전 요약 (1-4 구간, 설명 없이)")
        spec_df = rot.build_rotation_spec_table_simple(pro_arr, ama_arr, start=1, end=4)
        abs_df  = rot.build_abs_1_10_table(pro_arr, ama_arr)

        # 테이블 합치기
        spec_df = pd.concat([spec_df, abs_df], ignore_index=True)

        # ★ 프로-일반 차이 컬럼 추가
        for col in ("프로", "일반"):
            spec_df[col] = pd.to_numeric(spec_df[col], errors="coerce")
        spec_df["차이(프로-일반)"] = (spec_df["프로"] - spec_df["일반"]).round(2)

        st.dataframe(
            spec_df.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )

    # ── 탭 6: 🎯 RASI — 상대적 아크 크기 지수 ─────────────────────────────────
    with tab6:
        st.subheader("RASI (Relative Arc Size Index) — 시트 기반")

        # 팔/클럽 길이 입력 (m)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**프로 길이 입력 (m)**")
            p_arm  = st.number_input("팔 길이 (m) · 프로",   value=0.75, step=0.01, format="%.2f", key="rasi_p_arm")
            p_club = st.number_input("클럽 길이 (m) · 프로", value=1.00, step=0.01, format="%.2f", key="rasi_p_club")
        with c2:
            st.markdown("**일반 길이 입력 (m)**")
            a_arm  = st.number_input("팔 길이 (m) · 일반",   value=0.78, step=0.01, format="%.2f", key="rasi_a_arm")
            a_club = st.number_input("클럽 길이 (m) · 일반", value=1.02, step=0.01, format="%.2f", key="rasi_a_club")

        # 표 생성: 총 아크는 시트(CN/CO/CP)에서 자동 계산
        rasi_df = rasi.build_rasi_table_from_arrays(
            pro_arr, ama_arr,
            arm_len_pro=p_arm, club_len_pro=p_club,
            arm_len_ama=a_arm, club_len_ama=a_club,
        )

        st.dataframe(
            rasi_df.style.format({
                "프로": "{:.3f}",
                "일반": "{:.3f}",
                "차이(프로-일반)": "{:+.3f}",
            }),
            use_container_width=True
        )

        # (옵션) 세부 구간 Di 확인
        with st.expander("구간별 클럽헤드 이동거리(Di) 보기"):
            c3, c4 = st.columns(2)
            with c3:
                st.caption("프로")
                st.dataframe(rasi.build_rasi_segments_table(pro_arr), use_container_width=True)
            with c4:
                st.caption("일반")
                st.dataframe(rasi.build_rasi_segments_table(ama_arr), use_container_width=True)

        st.download_button(
            "CSV 다운로드(RASI)",
            data=rasi_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="rasi_compare.csv",
            mime="text/csv",
        )

    with tab7:
        st.subheader("손목–클럽헤드 표 (2프레임 & 삼각합)")

        # 프로/일반 비교표
        cmp_df = wri_chd.build_wri_chd_table_compare(pro_arr, ama_arr)
        st.dataframe(
            cmp_df.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )

        # CSV
        st.download_button(
            "CSV 다운로드(손목·클럽헤드)",
            data=cmp_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="wri_chd_compare.csv",
            mime="text/csv",
        )

    with tab8:
        st.subheader("프레임 4 분석")

        # 표 1: CN4-AX4 / CO4-AY4 / CP4-AZ4
        st.markdown("**프레임 4 벡터 차**")
        df_f4 = top.build_frame4_cnax_table(pro_arr, ama_arr)
        st.dataframe(
            df_f4.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(프레임4 벡터 차)",
            data=df_f4.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame4_vector_diff.csv",
            mime="text/csv",
            key="dl_f4_vec"
        )

        st.divider()  # 또는 st.markdown("---")

        # 표 2: ∠ABC (deg)
        st.markdown("**프레임 4 ∠ABC**")
        df_ang = top2.build_frame4_angle_table(pro_arr, ama_arr)
        st.dataframe(
            df_ang.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(프레임4 ∠ABC)",
            data=df_ang.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame4_angle_ABC.csv",
            mime="text/csv",
            key="dl_f4_ang"
        )

        st.divider()

        # 표 3: BB4-AM4 / AN4-BC4
        st.markdown("**프레임 4 추가 비교: BB4-AM4 / AN4-BC4**")
        df_extra = sho_turn.build_frame4_bbam_anbc_table(pro_arr, ama_arr)
        st.dataframe(
            df_extra.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(프레임4 BB/AM & AN/BC)",
            data=df_extra.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame4_bbam_anbc_compare.csv",
            mime="text/csv",
            key="dl_f4_extra"
        )

        st.divider()

        # 표 4: (AN4-BC4) - ((J1-M1) - (J4-M4))
        st.markdown("**프레임 4: AN4-BC4 - ((J1-M1) - (J4-M4))**")
        df_anbcjm = xfac.build_frame4_anbc_minus_jm_delta_table(pro_arr, ama_arr)
        st.dataframe(
            df_anbcjm.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(프레임4 AN/BC & J/M 조합)",
            data=df_anbcjm.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame4_anbc_minus_jm_delta.csv",
            mime="text/csv",
            key="dl_f4_anbcjm"
        )

        st.divider()

        # 표 5: CQ4 - CN4
        st.markdown("**프레임 4: CQ4 - CN4**")
        df_cqcn = chd.build_frame4_cqcn_table(pro_arr, ama_arr)
        st.dataframe(
            df_cqcn.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(프레임4 CQ4-CN4)",
            data=df_cqcn.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame4_cq4_minus_cn4.csv",
            mime="text/csv",
            key="dl_f4_cqcn"
        )

    with tab9:
        st.subheader("프레임 4·6·8 ∠ABC")
        df_468 = coc.build_frames_angle_ABC_table(pro_arr, ama_arr)  # ← 위 feature 함수 호출
        st.dataframe(
            df_468.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(프레임 4·6·8 ∠ABC)",
            data=df_468.to_csv(index=False).encode("utf-8-sig"),
            file_name="angles_ABC_f468.csv",
            mime="text/csv",
            key="dl_f468_angles"
        )
    
    with tab10:
        st.subheader("프레임 7: CP7 - AZ7")
        df_cp7 = lean.build_cp7_minus_az7_table(pro_arr, ama_arr)
        st.dataframe(
            df_cp7.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(CP7-AZ7)",
            data=df_cp7.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame7_cp7_minus_az7.csv",
            mime="text/csv",
            key="dl_cp7az7"
        )

    # ── 새 탭 2: (AM7 - BB7) + (AM8 - BB8) ─────────────────────────────────────
    with tab11:
        st.subheader("(AM7 - BB7) + (AM8 - BB8)")
        df_sum = bend.build_am_bb_7_8_sum_table(pro_arr, ama_arr)
        st.dataframe(
            df_sum.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(AM7/8-BB7/8 합)",
            data=df_sum.to_csv(index=False).encode("utf-8-sig"),
            file_name="am_bb_7_8_sum.csv",
            mime="text/csv",
            key="dl_am_bb_7_8"
        )
    
    with tab12:
        st.subheader("CL7 - CL1")
        df_cl = ank.build_cl7_minus_cl1_table(pro_arr, ama_arr)  # top = _6wri_chd
        st.dataframe(
            df_cl.style.format({"프로": "{:.2f}", "일반": "{:.2f}", "차이(프로-일반)": "{:+.2f}"}),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(CL7-CL1)",
            data=df_cl.to_csv(index=False).encode("utf-8-sig"),
            file_name="cl7_minus_cl1.csv",
            mime="text/csv",
            key="dl_cl7cl1"
        )

    with tab13:
        st.subheader("프레임 7: H7-K7 / AL7-BA7 / (AL7-BA7)-(H7-K7)")
        df_hk = opn.build_hk_alba_table(pro_arr, ama_arr)
        st.dataframe(
            df_hk.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(H7-K7 · AL7-BA7 · 조합)",
            data=df_hk.to_csv(index=False).encode("utf-8-sig"),
            file_name="frame7_hk_alba_table.csv",
            mime="text/csv",
        )
    
    with tab14:
        st.subheader("CN−CQ 스타일 (프레임 8 & 6)")
        df_cn_cq = clo.build_cn_cq_style_table(pro_arr, ama_arr)
        st.dataframe(
            df_cn_cq.style.format({
                "프로": "{:.2f}",
                "일반": "{:.2f}",
                "차이(프로-일반)": "{:+.2f}",
            }),
            use_container_width=True
        )
        st.download_button(
            "CSV 다운로드(CN−CQ 스타일)",
            data=df_cn_cq.to_csv(index=False).encode("utf-8-sig"),
            file_name="cn_cq_style_f8_f6.csv",
            mime="text/csv",
            key="dl_cn_cq_style"
        )

    with tab15:
        st.subheader("항목 19–23 표")
            
        t19 = t1923.build_19_r_wri_elb_x(pro_arr, ama_arr)
        st.dataframe(t19.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (19)", t19.to_csv(index=False).encode("utf-8-sig"),
                        "item19_r_wri_elb_x.csv", "text/csv")

        st.divider()
        st.subheader("20) 1/4 Head Y, Z")
        t20 = t1923.build_20_head_quarter(pro_arr, ama_arr)
        st.dataframe(t20.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (20)", t20.to_csv(index=False).encode("utf-8-sig"),
                        "item20_head_quarter.csv", "text/csv")

        st.divider()
        st.subheader("21) 8 CHD Y")
        t21 = t1923.build_21_8_chd_y(pro_arr, ama_arr)
        st.dataframe(t21.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (21)", t21.to_csv(index=False).encode("utf-8-sig"),
                        "item21_8_chd_y.csv", "text/csv")

        st.divider()
        st.subheader("22) 4/5 CHD SHALLOWING")
        t22 = t1923.build_22_chd_shallowing(pro_arr, ama_arr)
        st.dataframe(t22.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (22)", t22.to_csv(index=False).encode("utf-8-sig"),
                        "item22_shallowing.csv", "text/csv")

        st.divider()
        st.subheader("23) 4 R KNE X")
        t23 = t1923.build_23_4_r_kne_x(pro_arr, ama_arr)
        st.dataframe(t23.style.format({"프로":"{:.2f}","일반":"{:.2f}","차이(프로-일반)":"{:+.2f}"}),
                    use_container_width=True)
        st.download_button("CSV (23)", t23.to_csv(index=False).encode("utf-8-sig"),
                        "item23_4_r_kne_x.csv", "text/csv")