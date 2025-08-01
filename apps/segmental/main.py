#!/usr/bin/env python3
# main.py (project_root/apps/pelvis_analysis/main.py)

from pathlib import Path
import pandas as pd

from .knee    import compute_knee_tdd_table, compute_knee_rotation_table
from .pelvis  import compute_pelvis_tdd_table, compute_pelvis_rotation_table
from .shoulder import compute_shoulder_tdd_table, compute_shoulder_rotation_table
from .arm     import compute_arm_tdd_table, compute_arm_rotation_table
from .club    import compute_club_tdd_table, compute_club_rotation_table
from .center  import (
    compute_pelvis_center_deviation,
    compute_knee_center_deviation,
    compute_shoulder_center_deviation,
    compute_xyz_diff_summary
)

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
    -- out_path:    Excel로 저장할 경로 (None이면 저장 안 함)
    → 반환값: { "섹션명": DataFrame, … }
    """

    # 1) 각 테이블 계산
    df_tdd_knee      = compute_knee_tdd_table(pro_file, golfer_file)
    df_rot_knee      = compute_knee_rotation_table(pro_file, golfer_file)
    df_tdd_pelvis    = compute_pelvis_tdd_table(pro_file, golfer_file)
    df_rot_pelvis    = compute_pelvis_rotation_table(pro_file, golfer_file)
    df_tdd_shoulder  = compute_shoulder_tdd_table(pro_file, golfer_file)
    df_rot_shoulder  = compute_shoulder_rotation_table(pro_file, golfer_file)
    df_tdd_arm       = compute_arm_tdd_table(pro_file, golfer_file)
    df_rot_arm       = compute_arm_rotation_table(pro_file, golfer_file)
    df_tdd_club      = compute_club_tdd_table(pro_file, golfer_file)
    df_rot_club      = compute_club_rotation_table(pro_file, golfer_file)
    df_pelvis_center = compute_pelvis_center_deviation(pro_file, golfer_file)
    df_knee_center   = compute_knee_center_deviation(pro_file, golfer_file)
    df_shoulder_center = compute_shoulder_center_deviation(pro_file, golfer_file)
    df_xyz_diff_summary = compute_xyz_diff_summary(pro_file, golfer_file)

    # 2) 결과 모아서 dict으로
    results: dict[str, pd.DataFrame] = {
        "▶ Knee TDD Table":           df_tdd_knee,
        "▶ Knee Rotation Table":      df_rot_knee,
        "▶ Pelvis TDD Table":         df_tdd_pelvis,
        "▶ Pelvis Rotation Table":    df_rot_pelvis,
        "▶ Shoulder TDD Table":       df_tdd_shoulder,
        "▶ Shoulder Rotation Table":  df_rot_shoulder,
        "▶ Arm TDD Table":            df_tdd_arm,
        "▶ Arm Rotation Table":       df_rot_arm,
        "▶ Club TDD Table":           df_tdd_club,
        "▶ Club Rotation Table":      df_rot_club,
        "▶ Pelvis Center Deviation":  df_pelvis_center,
        "▶ Knee Center Deviation":    df_knee_center,
        "▶ Shoulder Center Deviation":df_shoulder_center,
        "▶ XYZ Diff Summary":         df_xyz_diff_summary,
    }

    # 3) Excel 저장 (out_path이 주어진 경우에만)
    if out_path:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            for title, df in results.items():
                sheet = title[:31]  # Excel 시트명 최대 31자
                df.to_excel(writer, sheet_name=sheet, index=True)
        print(f"▶️ Excel saved to {out_path.resolve()}")

    return results


if __name__ == "__main__":
    # 로컬 CLI 테스트
    base = Path("/Users/park_sh/Desktop/sim_pro")
    pro    = base / "driver/Rory McIlroy/first_data_transition.xlsx"
    golfer = base / "test/sample_first.xlsx"

    dfs = main(pro, golfer, times=[], out_path=Path("pelvis_analysis.xlsx"))
    for section, df in dfs.items():
        print(f"\n== {section} ==")
        print(df.head())
