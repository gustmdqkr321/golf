# #!/usr/bin/env python3
# # main.py (project_root)

# import pandas as pd
# from pathlib import Path

# from center_gravity import compute_delta_x_table, compute_delta_y_table, compute_delta_z_table, compute_summary_table, compute_smdi_mrmi
# from movement import compute_movement_table_Knee, compute_movement_table_hips, compute_movement_table_sho, compute_movement_table_head
# from total_move import compute_x_report, compute_y_report, compute_z_report
# from pelvis import compute_tilt_report
# def main():
#     # ─── 1) 입력 파일 경로 지정 ───
#     # Pro(기준) / Golfer(실제) 엑셀 파일 경로
#     pro_file = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
#     golfer_file = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

#     # ─── 2) ΔX, ΔY, ΔZ 테이블 계산 ───
#     df_smdi_mrmi = compute_smdi_mrmi(pro_file, golfer_file)
#     df_dx = compute_delta_x_table(pro_file,    golfer_file)
#     df_dy = compute_delta_y_table(pro_file,    golfer_file)
#     df_dz = compute_delta_z_table(pro_file,    golfer_file)
#     df_mov_knee = compute_movement_table_Knee(pro_file, golfer_file)
#     df_mov_hips = compute_movement_table_hips(pro_file, golfer_file)
#     df_mov_sho = compute_movement_table_sho(pro_file, golfer_file)
#     df_mov_head = compute_movement_table_head(pro_file, golfer_file)
#     df_move_x = compute_x_report(pro_file, golfer_file)
#     df_move_y = compute_y_report(pro_file, golfer_file)
#     df_move_z = compute_z_report(pro_file, golfer_file)
#     df_tilt = compute_tilt_report(pro_file, golfer_file, times=list(range(10)))

#     out = Path("center_grab.xlsx")
#     with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
#         wb = writer.book
#         ws = wb.add_worksheet("Summary")
#         writer.sheets["Summary"] = ws
#         title_fmt = wb.add_format({"bold": True, "align": "left"})

#         # write sections with spacing
#         sections = [
#             ("▶ SMDI & MRMI", df_smdi_mrmi),
#             ("▶ delta_x", df_dx),
#             ("▶ delta_y", df_dy),
#             ("▶ delta_z", df_dz),
#             ("▶ Summary Table", compute_summary_table(pro_file, golfer_file)),
#             ("▶ Movement Table", df_mov_knee),
#             ("▶ Movement Table (Hips)", df_mov_hips),
#             ("▶ Movement Table (Shoulders)", df_mov_sho),
#             ("▶ Movement Table (Head)", df_mov_head),
#             ("▶ Movement X", df_move_x),
#             ("▶ Movement Y", df_move_y),
#             ("▶ Movement Z", df_move_z),
#             ("▶ Pelvis Tilt Report", df_tilt),
#         ]
#         row = 0
#         for title, df in sections:
#             ws.write(row, 0, title, title_fmt)
#             df.to_excel(writer, "Summary", startrow=row+1, startcol=0)
#             row += len(df) + 3

#     print(f"▶️ '{out.name}'에 저장되었습니다.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# main.py (project_root)

import pandas as pd
from pathlib import Path

from .center_gravity import (
    compute_delta_x_table, compute_delta_y_table, compute_delta_z_table,
    compute_summary_table, compute_smdi_mrmi
)
from .movement import (
    compute_movement_table_Knee, compute_movement_table_hips,
    compute_movement_table_sho, compute_movement_table_head
)
from .total_move import compute_x_report, compute_y_report, compute_z_report
from .pelvis import compute_tilt_report

def main(
    pro_file: Path,
    golfer_file: Path,
    times: list[int],
    out_path: Path | None = None
) -> dict[str, pd.DataFrame]:
    """
    -- pro_file: 기준(프로) 엑셀 파일 경로
    -- golfer_file: 실제(골퍼) 엑셀 파일 경로
    -- times: Pelvis tilt 계산용 frame 리스트
    -- out_path: Excel로 저장할 경로 (None이면 저장 안 함)
    """
    # 1) 계산할 DataFrame 생성
    dfs = {
        "▶ SMDI & MRMI":           compute_smdi_mrmi(pro_file, golfer_file),
        "▶ ΔX Table":              compute_delta_x_table(pro_file, golfer_file),
        "▶ ΔY Table":              compute_delta_y_table(pro_file, golfer_file),
        "▶ ΔZ Table":              compute_delta_z_table(pro_file, golfer_file),
        "▶ Summary Table":         compute_summary_table(pro_file, golfer_file),
        "▶ Movement (Knee)":       compute_movement_table_Knee(pro_file, golfer_file),
        "▶ Movement (Hips)":       compute_movement_table_hips(pro_file, golfer_file),
        "▶ Movement (Shoulder)":   compute_movement_table_sho(pro_file, golfer_file),
        "▶ Movement (Head)":       compute_movement_table_head(pro_file, golfer_file),
        "▶ Total Move X Report":   compute_x_report(pro_file, golfer_file),
        "▶ Total Move Y Report":   compute_y_report(pro_file, golfer_file),
        "▶ Total Move Z Report":   compute_z_report(pro_file, golfer_file),
        "▶ Pelvis Tilt Report":    compute_tilt_report(pro_file, golfer_file, times),
    }

    # 2) out_path이 주어지면 Excel로 쓰기
    if out_path:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            wb = writer.book
            ws = wb.add_worksheet("Summary")
            writer.sheets["Summary"] = ws
            fmt = wb.add_format({"bold": True, "align": "left"})
            row = 0
            for title, df in dfs.items():
                ws.write(row, 0, title, fmt)
                df.to_excel(writer, sheet_name="Summary",
                            startrow=row+1, startcol=0, index=False)
                row += len(df) + 3
        print(f"▶️ Excel saved to '{out_path}'")

    return dfs

if __name__ == "__main__":
    # CLI 모드: 원래 쓰던 대로 Excel도 쓰고 싶다면 out_path를 넘겨주세요.
    pro = Path("/Users/park_sh/.../first_data_transition.xlsx")
    gol = Path("/Users/park_sh/.../sample_first.xlsx")
    dfs = main(pro, gol, times=list(range(10)), out_path=Path("center_grab.xlsx"))
