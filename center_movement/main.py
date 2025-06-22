#!/usr/bin/env python3
# main.py (project_root)

import pandas as pd
from pathlib import Path

from center_gravity import compute_delta_x_table, compute_delta_y_table, compute_delta_z_table, compute_summary_table, compute_smdi_mrmi
from movement import compute_movement_table_Knee, compute_movement_table_hips, compute_movement_table_sho, compute_movement_table_head
from total_move import compute_x_report, compute_y_report, compute_z_report
from pelvis import compute_tilt_report
def main():
    # ─── 1) 입력 파일 경로 지정 ───
    # Pro(기준) / Golfer(실제) 엑셀 파일 경로
    pro_file = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golfer_file = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # ─── 2) ΔX, ΔY, ΔZ 테이블 계산 ───
    df_smdi_mrmi = compute_smdi_mrmi(pro_file, golfer_file)
    df_dx = compute_delta_x_table(pro_file,    golfer_file)
    df_dy = compute_delta_y_table(pro_file,    golfer_file)
    df_dz = compute_delta_z_table(pro_file,    golfer_file)
    df_mov_knee = compute_movement_table_Knee(pro_file, golfer_file)
    df_mov_hips = compute_movement_table_hips(pro_file, golfer_file)
    df_mov_sho = compute_movement_table_sho(pro_file, golfer_file)
    df_mov_head = compute_movement_table_head(pro_file, golfer_file)
    df_move_x = compute_x_report(pro_file, golfer_file)
    df_move_y = compute_y_report(pro_file, golfer_file)
    df_move_z = compute_z_report(pro_file, golfer_file)
    df_tilt = compute_tilt_report(pro_file, golfer_file, times=list(range(10)))

    out = Path("center_grab.xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        title_fmt = wb.add_format({"bold": True, "align": "left"})

        # write sections with spacing
        sections = [
            ("▶ SMDI & MRMI", df_smdi_mrmi),
            ("▶ delta_x", df_dx),
            ("▶ delta_y", df_dy),
            ("▶ delta_z", df_dz),
            ("▶ Summary Table", compute_summary_table(pro_file, golfer_file)),
            ("▶ Movement Table", df_mov_knee),
            ("▶ Movement Table (Hips)", df_mov_hips),
            ("▶ Movement Table (Shoulders)", df_mov_sho),
            ("▶ Movement Table (Head)", df_mov_head),
            ("▶ Movement X", df_move_x),
            ("▶ Movement Y", df_move_y),
            ("▶ Movement Z", df_move_z),
            ("▶ Pelvis Tilt Report", df_tilt),
        ]
        row = 0
        for title, df in sections:
            ws.write(row, 0, title, title_fmt)
            df.to_excel(writer, "Summary", startrow=row+1, startcol=0)
            row += len(df) + 3

    print(f"▶️ '{out.name}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
