#!/usr/bin/env python3
# main.py (project_root)

import pandas as pd
from pathlib import Path

from knee import (
    compute_knee_tdd_table,
    compute_knee_rotation_table,
)
from pelvis import (
    compute_pelvis_tdd_table,
    compute_pelvis_rotation_table,
)
from shoulder import (
    compute_shoulder_tdd_table,
    compute_shoulder_rotation_table,
)
from arm import (
    compute_arm_tdd_table,
    compute_arm_rotation_table,
)
from club import (
    compute_club_tdd_table,
    compute_club_rotation_table,
)
from center import (
    compute_pelvis_center_deviation,
    compute_knee_center_deviation,
    compute_shoulder_center_deviation,
    compute_xyz_diff_summary
)
def main():
    # ─── 1) 입력 파일 경로 지정 ────────────────────────────────
    pro_file    = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golfer_file = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # ─── 2) 테이블 계산 ────────────────────────────────
    df_tdd_knee = compute_knee_tdd_table(pro_file, golfer_file)
    df_rot_knee = compute_knee_rotation_table(pro_file, golfer_file)
    df_tdd_pelvis = compute_pelvis_tdd_table(pro_file, golfer_file)
    df_rot_pelvis = compute_pelvis_rotation_table(pro_file, golfer_file)
    df_tdd_shoulder = compute_shoulder_tdd_table(pro_file, golfer_file)
    df_rot_shoulder = compute_shoulder_rotation_table(pro_file, golfer_file)
    df_tdd_arm = compute_arm_tdd_table(pro_file, golfer_file)
    df_rot_arm = compute_arm_rotation_table(pro_file, golfer_file)
    df_tdd_club = compute_club_tdd_table(pro_file, golfer_file)
    df_rot_club = compute_club_rotation_table(pro_file, golfer_file)
    df_pelvis_center = compute_pelvis_center_deviation(pro_file, golfer_file)
    df_knee_center = compute_knee_center_deviation(pro_file, golfer_file)
    df_shoulder_center = compute_shoulder_center_deviation(pro_file, golfer_file)
    df_xyz_diff_summary = compute_xyz_diff_summary(pro_file, golfer_file)


    # ─── 3) 엑셀에 저장 ─────────────────────────────────────────
    out = Path("pelvis_analysis.xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        title_fmt = wb.add_format({"bold": True, "align": "left"})

        sections = [
            ("▶ Knee TDD Table",          df_tdd_knee),
            ("▶ Knee Rotation Table",     df_rot_knee),
            ("▶ Pelvis TDD Table",       df_tdd_pelvis),
            ("▶ Pelvis Rotation Table",  df_rot_pelvis),
            ("▶ Shoulder TDD Table",     df_tdd_shoulder),
            ("▶ Shoulder Rotation Table", df_rot_shoulder),
            ("▶ Arm TDD Table",          df_tdd_arm),
            ("▶ Arm Rotation Table",     df_rot_arm),
            ("▶ Club TDD Table",         df_tdd_club),
            ("▶ Club Rotation Table",    df_rot_club),
            ("▶ Pelvis Center Deviation", df_pelvis_center),
            ("▶ Knee Center Deviation",   df_knee_center),
            ("▶ Shoulder Center Deviation", df_shoulder_center),
            ("▶ XYZ Diff Summary",        df_xyz_diff_summary),
        ]

        row = 0
        for title, df in sections:
            ws.write(row, 0, title, title_fmt)
            df.to_excel(writer, "Summary", startrow=row+1, startcol=0, index=True)
            row += len(df) + 3

    print(f"▶️ 결과가 '{out.name}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
