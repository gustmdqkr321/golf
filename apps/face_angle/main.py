#!/usr/bin/env python3
# main.py

from pathlib import Path
import numpy as np
import pandas as pd

from .wrist_rolling import summarize_player
from .cocking        import compute_table as compute_cocking_table
from .similarity     import rolling_sim, cocking_sim, hinging_sim, bowing_sim, club_sim
from .cocking_2d     import compute_yz_plane_angles
from .hinge          import calculate_full_hinging_table
from .bo_cu          import compute_table as generate_bowing_table
from .club_face      import compute_tilt_angles
from .forearm        import compute_tilt_numerators, compute_ay_bn_diffs, compute_abc_angles


def build_rolling_df(summary: dict[str, dict]) -> pd.DataFrame:
    rows = [
        "ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2",
        "1-4","4-7","7-9","1-7","STD","Total Δ"
    ]
    data: dict[str, list] = {}
    for name, stats in summary.items():
        data[f"{name} Wrist (°)"] = stats["wrist"] + [None] * 6
        data[f"{name} Pure Rolling (°)"] = (
            stats["pure_roll"]
            + [stats["sum1_4"], stats["sum4_7"], stats["sum7_9"],
               stats["diff1_7"], stats["std"], stats["total_delta"]]
        )
    df = pd.DataFrame(data, index=rows)
    df.index.name = "Frame"

    # rolling similarity
    r = np.array(summary["Rory"]["pure_roll"], float)
    h = np.array(summary["Hong"]["pure_roll"], float)
    mask = ~np.isnan(r) & ~np.isnan(h)
    sim = rolling_sim(r[mask], h[mask], alpha=2.0)
    df.loc["Similarity"] = [None] * (df.shape[1] - 1) + [sim]

    return df


def main(
    pro_file: Path,
    golfer_file: Path,
    times: list[int],
    out_path: Path | None = None
) -> dict[str, pd.DataFrame]:
    """
    -- pro_file:    기준(프로) 엑셀 파일 경로
    -- golfer_file: 실제(골퍼) 엑셀 파일 경로
    -- times:       이 앱에선 사용하지 않지만, 시그니처 통일용
    -- out_path:    결과를 Excel로 저장할 경로 (None이면 저장 안 함)
    → 반환값: { "섹션명": DataFrame, … }
    """

    # 1) Rolling 요약
    summary = {
        "Rory": summarize_player(pro_file),
        "Hong": summarize_player(golfer_file),
    }
    df_rolling = build_rolling_df(summary)

    # 2) Cocking ∠ABC & similarity
    df_cocking = compute_cocking_table(pro_file, golfer_file)
    r_delta = df_cocking["Rory_Δ(°)"].iloc[1:9].astype(float).to_numpy()
    h_delta = df_cocking["Hong_Δ(°)"].iloc[1:9].astype(float).to_numpy()
    cock_sim, _ = cocking_sim(r_delta, h_delta)
    df_cocking.loc["Similarity"] = [None] * (df_cocking.shape[1] - 1) + [cock_sim]

    # 3) Frontal (YZ-Plane)
    yz_r = compute_yz_plane_angles(pro_file)
    yz_h = compute_yz_plane_angles(golfer_file)
    frames10 = list(range(1, 11))
    df_frontal = pd.DataFrame({
        "Rory Frontal (°)": yz_r,
        "Hong Frontal (°)": yz_h
    }, index=frames10)
    df_frontal.index.name = "Frame"

    # 4) Hinging & similarity
    df_hinge = calculate_full_hinging_table(pro_file, golfer_file)
    r_hinge = df_hinge["Rory Hinging (deg)"].iloc[:10].astype(float).to_numpy()
    h_hinge = df_hinge["Hong Hinging (deg)"].iloc[:10].astype(float).to_numpy()
    hinge_sim = hinging_sim(r_hinge, h_hinge, alpha=2.0)
    df_hinge.loc["Similarity"] = [None] * (df_hinge.shape[1] - 1) + [hinge_sim]

    # 5) Bowing & similarity
    df_bowing = generate_bowing_table(pro_file, golfer_file)
    r_bow = df_bowing["Rory Rel. Bowing (°)"].iloc[:10].astype(float).to_numpy()
    h_bow = df_bowing["Hong Rel. Bowing (°)"].iloc[:10].astype(float).to_numpy()
    bow_sim = bowing_sim(r_bow, h_bow)
    df_bowing.loc["Similarity"] = [None] * (df_bowing.shape[1] - 1) + [bow_sim]

    # 6) Tilt & similarity
    tilt_r = compute_tilt_angles(pro_file)
    tilt_h = compute_tilt_angles(golfer_file)
    frames9 = list(range(1, 10))
    df_tilt = pd.DataFrame({
        "Rory Tilt (°)": tilt_r,
        "Hong Tilt (°)": tilt_h
    }, index=frames9)
    df_tilt.index.name = "Frame"
    tilt_sim = club_sim(np.array(tilt_r), np.array(tilt_h), alpha=2.0)
    df_tilt.loc["Similarity"] = [None] * (df_tilt.shape[1] - 1) + [tilt_sim]

    # 7) Tilt numerators
    nums_r = compute_tilt_numerators(pro_file)
    nums_h = compute_tilt_numerators(golfer_file)
    df_tilt_num = pd.DataFrame({
        "Rory Tilt Num": nums_r,
        "Hong Tilt Num": nums_h
    }, index=frames9)
    df_tilt_num.index.name = "Frame"

    # 8) AY-BN differences
    aybn_r = compute_ay_bn_diffs(pro_file)
    aybn_h = compute_ay_bn_diffs(golfer_file)
    frames11 = list(range(1, len(aybn_r) + 1))
    df_aybn = pd.DataFrame({
        "Rory AY-BN": aybn_r,
        "Hong AY-BN": aybn_h
    }, index=frames11)
    df_aybn.index.name = "Frame"

    # 9) ∠ABC angles
    abc_r = compute_abc_angles(pro_file)
    abc_h = compute_abc_angles(golfer_file)
    frames11 = list(range(1, len(abc_r) + 1))
    df_abc = pd.DataFrame({
        "Rory ∠ABC (°)": abc_r,
        "Hong ∠ABC (°)": abc_h
    }, index=frames11)
    df_abc.index.name = "Frame"

    # 10) 결과 모으기
    dfs = {
        "Wrist Rolling Summary":      df_rolling,
        "Cocking ∠ABC & Similarity":  df_cocking,
        "Frontal (YZ-Plane)":         df_frontal,
        "Hinging & Similarity":       df_hinge,
        "Bowing & Similarity":        df_bowing,
        "Tilt & Similarity":          df_tilt,
        "Tilt Numerators":            df_tilt_num,
        "AY-BN Differences":          df_aybn,
        "∠ABC Angles":                df_abc,
    }

    # 11) out_path이 주어지면 Excel에 각 DataFrame을 별도 시트로 저장
    if out_path:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            for title, df in dfs.items():
                sheet = title[:31]  # 엑셀 시트명 31자 제한
                df.to_excel(writer, sheet_name=sheet)
        print(f"▶️ Excel saved to {out_path.resolve()}")

    return dfs


if __name__ == "__main__":
    base = Path("/Users/park_sh/Desktop/sim_pro")
    pro = base / "driver/Rory McIlroy/first_data_transition.xlsx"
    gol = base / "test/sample_first.xlsx"
    # times는 통일 시그니처용이므로 빈 리스트 전달
    tables = main(pro, gol, times=[], out_path=Path("summary_all.xlsx"))
    for title, df in tables.items():
        print(f"\n== {title} ==")
        print(df.head())
