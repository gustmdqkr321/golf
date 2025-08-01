#!/usr/bin/env python3
# main.py (project_root/apps/club_path/main.py)

from pathlib import Path
import pandas as pd

# — 기존 모듈 임포트 —
from .CHD            import compute_cn_4_9, compute_bm_4_9
from .wrist          import compute_cn_minus_ax
from .Yaw            import compute_yaw_angles    
from .vertical       import compute_vertical_angles
from .elbow_wrist    import compute_ax_minus_ar, compute_bm_minus_bg
from .shoulder_elbow import compute_ar_minus_al, compute_bg_minus_ba
from .shoulder_wrist import compute_ax_minus_al, compute_bm_minus_ba
from .swing_plane    import compute_bac_with_status, compute_selected_diffs
from .bot            import compute_diff1, compute_diff2, compute_diff3, compute_diff4, compute_diff5
from .last           import compute_midpoint_distances

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
    -- out_path:    저장할 Excel 경로 (None이면 저장 안 함)
    → 반환값: { "메트릭명": DataFrame, … }
    """

    # 1) 플레이어 맵핑
    players = {
        "Pro":    pro_file,
        "Golfer": golfer_file,
    }

    # 2) 메트릭 함수 정의
    all_metrics = {
        "CN (4–9)"                 : compute_cn_4_9,
        "BM (4–9)"                 : compute_bm_4_9,
        "CN − AX (1–10)"           : compute_cn_minus_ax,
        "Yaw (1–10)"               : compute_yaw_angles,
        "Vertical (1–10)"          : compute_vertical_angles,
        "AX − AR (1–9)"            : compute_ax_minus_ar,
        "BM − BG (1–9)"            : compute_bm_minus_bg,
        "AR − AL (1–9)"            : compute_ar_minus_al,
        "BG − BA (1–9)"            : compute_bg_minus_ba,
        "AX − AL (1–9)"            : compute_ax_minus_al,
        "BM − BA (1–9)"            : compute_bm_minus_ba,
        "Selected Diffs (1–10)"    : compute_selected_diffs,
        "BAC (1–10)"               : compute_bac_with_status,
        "Diff 1 (1–9)"             : compute_diff1,
        "Diff 2 (1–9)"             : compute_diff2,
        "Diff 3 (1–9)"             : compute_diff3,
        "Diff 4 (1–9)"             : compute_diff4,
        "Diff 5 (1–9)"             : compute_diff5,
        "Midpoint Distances (1–10)": compute_midpoint_distances,
    }

    # 3) DataFrame으로 변환
    dfs: dict[str, pd.DataFrame] = {}
    for metric, func in all_metrics.items():
        # player별 값 계산
        player_vals = {name: func(path) for name, path in players.items()}

        # 리턴된 리스트 길이에 맞춰 인덱스 생성
        sample = next(iter(player_vals.values()))
        idx = list(range(1, len(sample) + 1))

        # DataFrame 생성 (rows=Frame, cols=플레이어명)
        df = pd.DataFrame(player_vals, index=idx)
        df.index.name = "Frame"
        dfs[metric] = df

    # 4) out_path이 지정되면 Excel로 저장
    if out_path:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            for metric, df in dfs.items():
                sheet_name = metric[:31]  # 시트명 길이 제한
                df.to_excel(writer, sheet_name=sheet_name)
        print(f"▶️ Excel saved to {out_path.resolve()}")

    return dfs


if __name__ == "__main__":
    # 로컬에서 직접 테스트할 때 사용
    base = Path("/Users/park_sh/Desktop/sim_pro")
    pro = base / "driver/Rory McIlroy/first_data_transition.xlsx"
    gol = base / "test/sample_first.xlsx"

    tables = main(pro, gol, times=list(range(10)), out_path=Path("club_head_diffs.xlsx"))

    # 콘솔 요약 출력
    for metric, df in tables.items():
        print(f"\n== {metric} ==")
        print(df.head())
