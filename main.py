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

def main():
    # ─── 1) 비교할 플레이어 엑셀 파일 3개 지정 ───
    players = {
        "Rory":    Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx"),
        "Player2": Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx"),
        "Player3": Path("/Users/park_sh/Desktop/sim_pro/test/kys.xlsx"),
    }

    # ─── 2) θ 리스트 계산 ───
    rolling = { name: compute_frame_thetas(fp)           for name, fp in players.items() }
    cock3d  = { name: cocking3d(fp)                      for name, fp in players.items() }
    hinge   = { name: compute_arcsin_angles(fp)         for name, fp in players.items() }
    bowing  = { name: compute_cupping_bowing_angles(fp)  for name, fp in players.items() }
    tilt    = { name: compute_tilt_angles_excel(fp)     for name, fp in players.items() }

    frames      = list(range(1, 11))
    tilt_frames = list(range(1, len(next(iter(tilt.values()))) + 1))

    # ─── 3) DataFrame 준비 ───
    df_rolling = pd.DataFrame({"Frame": frames, **rolling})
    df_cock3d  = pd.DataFrame({"Frame": frames, **cock3d})
    df_hinge   = pd.DataFrame({"Frame": frames, **hinge})
    df_bowing  = pd.DataFrame({"Frame": frames, **bowing})
    df_tilt    = pd.DataFrame({"Frame": tilt_frames, **tilt})

    # ─── 4) 엑셀로 저장 ───
    out_xlsx = Path("results_comparison.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_rolling.to_excel(writer, sheet_name="Rolling", index=False)
        df_cock3d .to_excel(writer, sheet_name="Cocking3D", index=False)
        df_hinge  .to_excel(writer, sheet_name="Hinge",    index=False)
        df_bowing .to_excel(writer, sheet_name="Bowing",   index=False)
        df_tilt   .to_excel(writer, sheet_name="Tilt",     index=False)
    print(f"▶️ 결과를 엑셀로 저장했습니다: {out_xlsx.resolve()}")

    # ─── 5) 전체 비교 플롯 (5행 1열) ───
    fig, axes = plt.subplots(5, 1, figsize=(8, 20))

    # Rolling
    ax = axes[0]
    for name, vals in rolling.items():
        ax.plot(frames, vals, marker='o', label=name)
    ax.set(title="Wrist Rolling Angle", xlabel="Frame n", ylabel="θ (°)")
    ax.grid(); ax.legend()

    # Cocking 3D
    ax = axes[1]
    for name, vals in cock3d.items():
        ax.plot(frames, vals, marker='s', label=name)
    ax.set(title="Cocking Angle (3D atan2)", xlabel="Frame n", ylabel="θ (°)")
    ax.grid(); ax.legend()

    # Hinge
    ax = axes[2]
    for name, vals in hinge.items():
        ax.plot(frames, vals, marker='^', label=name)
    ax.set(title="Hinge Angle (−arcsin)", xlabel="Frame n", ylabel="θ (°)")
    ax.grid(); ax.legend()

    # Bowing
    ax = axes[3]
    for name, vals in bowing.items():
        ax.plot(frames, vals, marker='d', label=name)
    ax.set(title="Cupping/Bowing Angle", xlabel="Frame n", ylabel="θ (°)")
    ax.grid(); ax.legend()

    # Tilt
    ax = axes[4]
    for name, vals in tilt.items():
        ax.plot(tilt_frames, vals, marker='x', label=name)
    ax.set(title="Tilt Angle", xlabel="Frame n", ylabel="θ (°)")
    ax.grid(); ax.legend()

    plt.tight_layout()
    img_path = Path("all_metrics.png")
    fig.savefig(img_path)
    print(f"▶️ 플롯 이미지를 저장했습니다: {img_path.resolve()}")
    plt.show()

    # ─── 6) Rolling 구간별·기준 프레임별 차이 출력 ───
    print("\n=== Wrist Rolling 구간별 차이 (n → n+1) ===")
    for name, vals in rolling.items():
        diffs = compute_consecutive_diffs(vals)
        print(f"\n[{name}]")
        for i, d in enumerate(diffs, start=1):
            print(f"  {i}→{i+1}: {d:.2f}°")

    print("\n=== Wrist Rolling 1→n 차이 (1프레임 기준) ===")
    for name, vals in rolling.items():
        diffs = compute_base_diffs(vals)
        print(f"\n[{name}]")
        for i, d in enumerate(diffs, start=2):
            print(f"  1→{i}: {d:.2f}°")

    # ─── 7) Cocking 2D (cos⁻¹) 결과 프린트 ───
    print("\n=== 2D Cocking (cos⁻¹) 공식 결과 ===")
    for name, fp in players.items():
        print(f"\n[{name}]")
        cocking2d(fp)

if __name__ == '__main__':
    main()
