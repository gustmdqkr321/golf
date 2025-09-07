# sections/swing/main.py
from pathlib import Path
import streamlit as st
import pandas as pd

from .features import _1hand_hight as hand
from .features import _2swing_tempo as swing
from .features import _3body_arm as fc   # ← 같은 섹션(features)에서 import
from .features import _4center as rot

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

    # 새 탭 추가: 📋 비율 표
    tab1, tab2, tab3, tab4 = st.tabs(["🖐 손높이/각도 비교", "⏱ 템포 · 리듬", "📋 비율 표", "중심"])

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
