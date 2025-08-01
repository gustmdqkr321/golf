from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wrist_rolling import summarize_player
from cocking        import compute_table            # Cocking ∠ABC
from similarity     import rolling_sim, cocking_sim, hinging_sim, bowing_sim, club_sim  # 유사도 함수
from cocking_2d     import compute_yz_plane_angles  # Frontal (YZ)
from hinge          import calculate_full_hinging_table    # Hinging
from bo_cu          import compute_table as generate_bowing_table    # Bowing
from club_face     import compute_tilt_angles    # Tilt
from forearm import (
    compute_tilt_numerators,
    compute_ay_bn_diffs,
    compute_abc_angles
)

def build_rolling_df(summary: dict[str, dict]) -> pd.DataFrame:
    rows = [
        "ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2",
        "1-4","4-7","7-9","1-7","STD","Total Δ"
    ]
    data = {}
    for name, stats in summary.items():
        data[f"{name} Wrist (°)"]        = stats["wrist"] + [None]*6
        data[f"{name} Pure Rolling (°)"] = (
            stats["pure_roll"]
            + [stats["sum1_4"], stats["sum4_7"], stats["sum7_9"],
               stats["diff1_7"], stats["std"], stats["total_delta"]]
        )
    df = pd.DataFrame(data, index=rows)
    df.index.name = "Frame"

    # Rolling similarity
    r = np.array(summary["Rory"]["pure_roll"], float)
    h = np.array(summary["Hong"]["pure_roll"], float)
    mask = ~np.isnan(r) & ~np.isnan(h)
    sim = rolling_sim(r[mask], h[mask], alpha=2.0)
    sim_row = [None]*(df.shape[1]-1) + [sim]
    df.loc["Similarity"] = sim_row

    return df


def main():
    # 1) 파일 경로 지정
    file_rory = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    file_hong = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # 2) Rolling 요약
    rolling_summary = {
        "Rory": summarize_player(file_rory),
        "Hong": summarize_player(file_hong),
    }
    df_rolling = build_rolling_df(rolling_summary)

    # 3) Cocking ∠ABC 요약 + similarity
    df_cocking = compute_table(file_rory, file_hong)
    r_delta = df_cocking["Rory_Δ(°)"].iloc[1:9].astype(float).to_numpy()
    h_delta = df_cocking["Hong_Δ(°)"].iloc[1:9].astype(float).to_numpy()
    cock_sim, _ = cocking_sim(r_delta, h_delta)
    df_cocking.loc["Similarity"] = [None]*(df_cocking.shape[1]-1) + [cock_sim]

    # 4) Frontal(YZ) 요약
    yz_r = compute_yz_plane_angles(file_rory)
    yz_h = compute_yz_plane_angles(file_hong)
    frames10 = list(range(1,11))
    df_frontal = pd.DataFrame({
        "Rory Frontal (°)": yz_r,
        "Hong Frontal (°)": yz_h
    }, index=frames10)
    df_frontal.index.name = "Frame"

    # 5) Hinging 요약 + similarity
    df_hinge = calculate_full_hinging_table(file_rory, file_hong)
    r_hinge = df_hinge["Rory Hinging (deg)"].iloc[:10].astype(float).to_numpy()
    h_hinge = df_hinge["Hong Hinging (deg)"].iloc[:10].astype(float).to_numpy()
    hinge_sim = hinging_sim(r_hinge, h_hinge, alpha=2.0)
    df_hinge.loc["Similarity"] = [None]*(df_hinge.shape[1]-1) + [hinge_sim]

    # 6) Bowing 요약 + similarity
    df_bowing = generate_bowing_table(file_rory, file_hong)
    r_bow = df_bowing["Rory Rel. Bowing (°)"].iloc[:10].astype(float).to_numpy()
    h_bow = df_bowing["Hong Rel. Bowing (°)"].iloc[:10].astype(float).to_numpy()
    bow_sim = bowing_sim(r_bow, h_bow)
    df_bowing.loc["Similarity"] = [None]*(df_bowing.shape[1]-1) + [bow_sim]

    # 7) Tilt 요약 + similarity
    tilt_r = compute_tilt_angles(file_rory)
    tilt_h = compute_tilt_angles(file_hong)
    df_tilt = pd.DataFrame({
        "Rory Tilt (°)": tilt_r,
        "Hong Tilt (°)": tilt_h
    }, index=list(range(1,10)))
    df_tilt.index.name = "Frame"
    tilt_sim = club_sim(np.array(tilt_r), np.array(tilt_h), alpha=2.0)
    df_tilt.loc["Similarity"] = [None]*(df_tilt.shape[1]-1) + [tilt_sim]

    # 8) Tilt numerators, AY-BN diffs, ABC angles
    nums_r = compute_tilt_numerators(file_rory)
    nums_h = compute_tilt_numerators(file_hong)
    df_tilt_num = pd.DataFrame({
        "Rory Tilt Num": nums_r,
        "Hong Tilt Num": nums_h
    }, index=list(range(1,10)))
    df_tilt_num.index.name = "Frame"

    aybn_r = compute_ay_bn_diffs(file_rory)
    aybn_h = compute_ay_bn_diffs(file_hong)
    df_aybn = pd.DataFrame({
        "Rory AY-BN": aybn_r,
        "Hong AY-BN": aybn_h
    }, index=list(range(1,11)))
    df_aybn.index.name = "Frame"

    abc_r = compute_abc_angles(file_rory)
    abc_h = compute_abc_angles(file_hong)
    df_abc = pd.DataFrame({
        "Rory ∠ABC (°)": abc_r,
        "Hong ∠ABC (°)": abc_h
    }, index=list(range(1,11)))
    df_abc.index.name = "Frame"

    # 9) 단일 시트에 모두 쓰기
    out = Path("summary_all.xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        title_fmt = wb.add_format({"bold": True, "align": "left"})

        # write sections with spacing
        sections = [
            ("▶ Wrist Rolling Summary", df_rolling),
            ("▶ Cocking ∠ABC & Similarity", df_cocking),
            ("▶ Frontal (YZ-Plane) Angles", df_frontal),
            ("▶ Hinging Angles & Similarity", df_hinge),
            ("▶ Bowing Angles & Similarity", df_bowing),
            ("▶ Tilt Angles & Similarity", df_tilt),
            ("▶ Tilt Numerators", df_tilt_num),
            ("▶ AY-BN Differences", df_aybn),
            ("▶ ∠ABC Angles (A-B-C)", df_abc)
        ]
        row = 0
        for title, df in sections:
            ws.write(row, 0, title, title_fmt)
            df.to_excel(writer, "Summary", startrow=row+1, startcol=0)
            row += len(df) + 3

    print(f"▶️ '{out.name}'에 저장되었습니다.")

    # 10) plt
    fig, axes = plt.subplots(6, 1, figsize=(6, 24))

    # (a) Pure Rolling
    ax = axes[0]
    seq_r = np.array(rolling_summary["Rory"]["pure_roll"], float)
    seq_h = np.array(rolling_summary["Hong"]["pure_roll"], float)
    mask = ~np.isnan(seq_r) & ~np.isnan(seq_h)
    x_rr = np.arange(1, mask.sum() + 1)
    ax.plot(x_rr, seq_r[mask], marker='o', label="Rory")
    ax.plot(x_rr, seq_h[mask], marker='s', label="Hong")
    ax.set_title("Pure Rolling Comparison (Frames 1–9)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Δ Rolling (°)")
    ax.grid(True); ax.legend()

    # (b) Cocking ∠ABC
    ax = axes[1]
    for name in ["Rory", "Hong"]:
        ang = compute_table(file_rory, file_rory)[f"{name}_∠ABC"].iloc[:10]
        ax.plot(frames10, ang, marker='o', label=name)
    ax.set_title("Cocking ∠ABC (1–10)"); ax.set_xlabel("Frame"); ax.set_ylabel("∠ABC (°)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

    # (c) Frontal YZ
    ax = axes[2]
    ax.plot(frames10, yz_r, marker='o', label="Rory")
    ax.plot(frames10, yz_h, marker='s', label="Hong")
    ax.set_title("Frontal (YZ-Plane) Angles (1–10)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Angle (°)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

    # (d) Hinging
    ax = axes[3]
    hinge_r = df_hinge["Rory Hinging (deg)"].iloc[:10].astype(float)
    hinge_h = df_hinge["Hong Hinging (deg)"].iloc[:10].astype(float)
    ax.plot(frames10, hinge_r, marker='o', label="Rory")
    ax.plot(frames10, hinge_h, marker='s', label="Hong")
    ax.set_title("Hinging Angles (1–10)")
    ax.set_xlabel("Frame"); ax.set_ylabel("θ_hinging (°)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

    # (e) Bowing
    ax = axes[4]
    bowing_r = df_bowing["Rory Rel. Bowing (°)"].iloc[:10].astype(float)
    bowing_h = df_bowing["Hong Rel. Bowing (°)"].iloc[:10].astype(float)
    ax.plot(frames10, bowing_r, marker='o', label="Rory")
    ax.plot(frames10, bowing_h, marker='s', label="Hong")
    ax.set_title("Bowing Angles (1–10)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Bowing (°)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

    # (f) Tilt
    ax = axes[5]
    frames9 = list(range(1, 10))
    ax.plot(frames9, tilt_r, marker='o', label="Rory")
    ax.plot(frames9, tilt_h, marker='s', label="Hong")
    ax.set_title("Tilt Angles (1–9)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Tilt (°)")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

    plt.tight_layout()
    img = Path("summary_all.png")
    fig.savefig(img)
    print(f"▶️ '{img.name}'에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    main()
