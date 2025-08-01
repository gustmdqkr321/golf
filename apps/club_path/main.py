# main.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from CHD import compute_cn_4_9, compute_bm_4_9
from wrist import compute_cn_minus_ax
from Yaw import compute_yaw_angles    
from vertical import compute_vertical_angles
from elbow_wrist import compute_ax_minus_ar, compute_bm_minus_bg
from shoulder_elbow import compute_ar_minus_al, compute_bg_minus_ba
from shoulder_wrist import compute_ax_minus_al, compute_bm_minus_ba
from swing_plane import compute_bac_with_status, compute_selected_diffs
from bot import compute_diff1, compute_diff2, compute_diff3, compute_diff4, compute_diff5
from last import compute_midpoint_distances

def main():
    # ─── 1) 플레이어 파일 지정 ───
    players = {
        "Rory": Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx"),
        "hong": Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx"),
        # 나중에 다시 추가 가능
        # "kim":  Path("/Users/park_sh/Desktop/sim_pro/test/kys.xlsx"),
    }

    # ─── 2) 전체 매트릭스 정의 ───
    all_metrics = {
        "CN (4–9)":          compute_cn_4_9,
        "BM (4–9)":          compute_bm_4_9,
        "CN − AX (1–10)":    compute_cn_minus_ax,
        "Yaw (1–10)":        compute_yaw_angles,
        "Vertical (1–10)":   compute_vertical_angles,
        "AX − AR (1–9)":     compute_ax_minus_ar,
        "BM − BG (1–9)":     compute_bm_minus_bg,
        "AR − AL (1–9)":     compute_ar_minus_al,
        "BG − BA (1–9)":     compute_bg_minus_ba,
        "AX − AL (1–9)":     compute_ax_minus_al,
        "BM − BA (1–9)":     compute_bm_minus_ba,
        "Selected Diffs (1–10)": compute_selected_diffs,
        "BAC (1–10)":        compute_bac_with_status,
        "Diff 1 (1–9)":      compute_diff1,
        "Diff 2 (1–9)":      compute_diff2,
        "Diff 3 (1–9)":      compute_diff3,
        "Diff 4 (1–9)":      compute_diff4,
        "Diff 5 (1–9)":      compute_diff5,
        "Midpoint Distances (1–10)": compute_midpoint_distances,
    }

    # ─── 3) 엑셀에 담을 결과 계산 ───
    results = {
        metric: {name: func(fp) for name, fp in players.items()}
        for metric, func in all_metrics.items()
    }

    # ─── 4) Excel 저장 ───
    out_xlsx = Path("club_head_diffs.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        wb = writer.book
        ws = wb.add_worksheet("Metrics")
        writer.sheets["Metrics"] = ws

        title_fmt = wb.add_format({"bold": True,   "align": "left"})
        name_fmt  = wb.add_format({"italic": True, "align": "left"})
        val_fmt   = wb.add_format({"align": "center"})
        ws.set_column(0, 0, 25)
        ws.set_column(1, 12, 12)

        row = 0
        for metric, player_dict in results.items():
            ws.write(row, 0, metric, title_fmt)
            row += 1
            for pname, vals in player_dict.items():
                ws.write(row, 0, pname, name_fmt)
                for col, v in enumerate(vals, 1):
                    ws.write(row, col, v, val_fmt)
                row += 1
            row += 1

    print(f"▶️ Excel 저장 완료: {out_xlsx.resolve()}")

    # ─── 5) 플롯할 매트릭스만 골라서 그리기 ───
    plot_metrics = [
        "CN (4–9)",
        "BM (4–9)",
        "CN − AX (1–10)",
        "Yaw (1–10)",
        "Vertical (1–10)",
        # "AX − AR (1–9)",   # 얘네는 제외
        # "BM − BG (1–9)",
    ]

    n = len(plot_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)

    for ax, metric in zip(axes[0], plot_metrics):
        data = results[metric]
        # x축 프레임 범위 자동 설정
        if "(4–9)" in metric:
            x = list(range(4, 10))
        else:
            x = list(range(1, 11))
        for pname, vals in data.items():
            ax.plot(x, vals, marker='o', label=pname)
        ax.set_title(metric)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    img_path = Path("club_head_diffs.png")
    fig.savefig(img_path)
    print(f"▶️ 플롯 저장 완료: {img_path.resolve()}")
    plt.show()

if __name__ == "__main__":
    main()
