# main.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from Rolling import (
    compute_frame_thetas,
    compute_consecutive_diffs,
    compute_base_diffs,
)
from cocking import (
    cocking2d,
    cocking3d,
)
from Hinge import compute_arcsin_angles
from cuNbo import compute_cupping_bowing_angles
from clubface import compute_tilt_angles_excel

# 새로 만든 함수들을 불러옵니다
from club import (
    compute_cp_cs_cq_cn_diffs,
    compute_ay_bn_diffs,
)
from angle import (
    compute_segmented_signed_ABC
)

def main():
    # ─── 1) 비교할 플레이어 엑셀 파일 3개 지정 ───
    players = {
        "Rory": Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx"),
        "hong": Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx"),
        "kim":  Path("/Users/park_sh/Desktop/sim_pro/test/kys.xlsx"),
    }

    # ─── 2) 각 메트릭별 θ 리스트 계산 ───
    metrics = {
        "Wrist Rolling Angle":      compute_frame_thetas,
        "3D Cocking Angle": cocking3d,
        "Hinge Angle":    compute_arcsin_angles,
        "Cupping/Bowing Angle":     compute_cupping_bowing_angles,
        "clubface":           compute_tilt_angles_excel,
    }
    results = {
        metric: {name: func(fp) for name, fp in players.items()}
        for metric, func in metrics.items()
    }

    # ─── 3) Rolling diff 및 Cocking2D도 미리 계산 ───
    rolling_vals = results["Wrist Rolling Angle"]
    roll_consec  = {name: compute_consecutive_diffs(vals) for name, vals in rolling_vals.items()}
    roll_base    = {name: compute_base_diffs(vals)     for name, vals in rolling_vals.items()}
    

    # ─── 4) 엑셀로 저장 (단일 시트, 세로 블록) ───
    out_xlsx = Path("face_angle.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        workbook  = writer.book
        worksheet = workbook.add_worksheet("Metrics")
        writer.sheets["Metrics"] = worksheet

        title_fmt = workbook.add_format({"bold": True,  "align": "left"})
        name_fmt  = workbook.add_format({"italic": True,"align": "left"})
        val_fmt   = workbook.add_format({"align": "center"})

        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 20, 12)

        row = 0
        # 4.1) 메트릭별 θ 블록
        for metric_name, player_dict in results.items():
            worksheet.write(row, 0, metric_name, title_fmt)
            row += 1
            for name, vals in player_dict.items():
                worksheet.write(row, 0, name, name_fmt)
                for c, v in enumerate(vals, start=1):
                    worksheet.write(row, c, v, val_fmt)
                row += 1
            row += 1

        # 4.2) Rolling 연속 차이
        worksheet.write(row, 0, "Rolling Consecutive Diffs", title_fmt)
        row += 1
        for name, diffs in roll_consec.items():
            worksheet.write(row, 0, name, name_fmt)
            for c, v in enumerate(diffs, start=1):
                worksheet.write(row, c, v, val_fmt)
            row += 1
        row += 1

        # 4.3) Rolling 기준 차이
        worksheet.write(row, 0, "Rolling Base Diffs", title_fmt)
        row += 1
        for name, diffs in roll_base.items():
            worksheet.write(row, 0, name, name_fmt)
            for c, v in enumerate(diffs, start=1):
                worksheet.write(row, c, v, val_fmt)
            row += 1
        row += 1

        # 4.5) CP–CS / CQ–CN / CN–CQ diffs
        worksheet.write(row, 0, "club", title_fmt)
        row += 1
        for name, fp in players.items():
            diffs = compute_cp_cs_cq_cn_diffs(fp)
            worksheet.write(row, 0, name, name_fmt)
            for c, v in enumerate(diffs, start=1):
                worksheet.write(row, c, v, val_fmt)
            row += 1
        row += 1

        # 4.6) AY–BN diffs
        worksheet.write(row, 0, "Forearm Supination 1", title_fmt)
        row += 1
        for name, fp in players.items():
            diffs = compute_ay_bn_diffs(fp)
            worksheet.write(row, 0, name, name_fmt)
            for c, v in enumerate(diffs, start=1):
                worksheet.write(row, c, v, val_fmt)
            row += 1

        ### 4.1.2.2
        worksheet.write(row, 0, "Forearm Supination 2", title_fmt)
        row += 1
        for name, fp in players.items():
            vals = compute_segmented_signed_ABC(fp)
            worksheet.write(row, 0, name, name_fmt)
            for c, v in enumerate(vals, start=1):
                worksheet.write(row, c, v, val_fmt)
            row += 1

    print(f"▶️ 결과를 ‘Metrics’ 시트에 저장했습니다: {out_xlsx.resolve()}")

    # ─── 5) (옵션) 전체 비교 플롯 ───
    import math
    n_metrics = len(metrics)
    ncols = 3
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # 2D 배열을 1D로 펼쳐서 순서대로 매핑
    axes_flat = axes.flatten()

    for ax, (metric_name, player_dict) in zip(axes_flat, results.items()):
        for name, vals in player_dict.items():
            ax.plot(range(1, len(vals)+1), vals, marker='o', label=name)
        ax.set_title(metric_name)
        ax.set_xlabel('Frame n')
        ax.set_ylabel('θ (°)')
        ax.grid(True)
        ax.legend()

    # 사용하지 않은 축은 꺼 버리기
    for ax in axes_flat[n_metrics:]:
        ax.axis('off')

    plt.tight_layout()
    img_path = Path("all_metrics.png")
    fig.savefig(img_path)
    print(f"▶️ 플롯 이미지를 저장했습니다: {img_path.resolve()}")
    plt.show()

if __name__ == '__main__':
    main()
