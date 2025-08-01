import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀을 numpy 2D 배열로 읽어들임"""
    # pandas 로 전체 시트를 header=None 으로 읽고 .values 반환
    import pandas as _pd
    return _pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """
    arr, 'AX1' → arr[row=0, col=col_letters_to_index('AX')]
    숫자는 1-based frame index, 문자는 컬럼 레이블
    """
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])
# 프레임 이름(1~10)
_frames = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","Finish"]

def compute_tilt_report(
    pro_path: Path,
    golfer_path: Path,
    times: list[float],
    pro_label: str = "Rory",
    golfer_label: str = "Hong"
) -> pd.DataFrame:
    """
    • 골반 tilt: θ_t = I_t - L_t
    • 어깨 tilt: θ_t = AM_t - BB_t
    • Δθ_t = |θ_t - θ_{t-1}| (절댓값)
    • v_t = Δθ_t / Δt
    • 구간(1-4, 4-7, 7-10) 총 Δθ, avg 속도 추가
    """

    # 1) raw tilt 계산
    arr_p = load_sheet(pro_path)
    arr_h = load_sheet(golfer_path)
    N = len(_frames)

    θp_pelvis   = np.array([ g(arr_p, f"I{i}") - g(arr_p, f"L{i}") for i in range(1, N+1) ])
    θh_pelvis   = np.array([ g(arr_h, f"I{i}") - g(arr_h, f"L{i}") for i in range(1, N+1) ])
    θp_shoulder = np.array([ g(arr_p, f"AM{i}") - g(arr_p, f"BB{i}") for i in range(1, N+1) ])
    θh_shoulder = np.array([ g(arr_h, f"AM{i}") - g(arr_h, f"BB{i}") for i in range(1, N+1) ])

    # 2) 기본 DataFrame
    df = pd.DataFrame({
        "Frame":  _frames,
        f"{pro_label} Pelvis θ":    θp_pelvis.round(2),
        f"{golfer_label} Pelvis θ": θh_pelvis.round(2),
        f"{pro_label} Shoulder θ":    θp_shoulder.round(2),
        f"{golfer_label} Shoulder θ": θh_shoulder.round(2),
        "Time": times,
    })

    # 3) Δθ (절댓값) & 속도 계산
    for part in ["Pelvis", "Shoulder"]:
        col_p = f"{pro_label} {part} θ"
        col_h = f"{golfer_label} {part} θ"

        # 절댓값 이동량
        df[f"Δ {col_p}"] = df[col_p].diff().abs().round(2)
        df[f"Δ {col_h}"] = df[col_h].diff().abs().round(2)

        # 속도 = Δθ / Δt
        dt = df["Time"].diff()
        df[f"{col_p} speed"] = (df[f"Δ {col_p}"] / dt).round(2)
        df[f"{col_h} speed"] = (df[f"Δ {col_h}"] / dt).round(2)

    # 4) Hong vs Rory raw-차이 컬럼
    df["Pelvis Δ(Hong–Rory)"]   = (df[f"{golfer_label} Pelvis θ"]   - df[f"{pro_label} Pelvis θ"]).round(2)
    df["Shoulder Δ(Hong–Rory)"] = (df[f"{golfer_label} Shoulder θ"] - df[f"{pro_label} Shoulder θ"]).round(2)

    # 5) 구간별 합계 및 평균 속도
    segs = {"1-4": (0, 3), "4-7": (3, 6), "7-10": (6, 9)}
    rows = []
    for seg, (i0, i1) in segs.items():
        d = {"Frame": seg,
             "Time": round(df.at[i1, "Time"] - df.at[i0, "Time"], 3)}
        for part in ["Pelvis", "Shoulder"]:
            for label in [pro_label, golfer_label]:
                # 총 변화량
                Δ = df.at[i1, f"Δ {label} {part} θ"].sum()
                d[f"Σ Δ {label} {part}"] = round(Δ, 2)
                # 구간 평균 속도
                vavg = Δ / (df.at[i1, "Time"] - df.at[i0, "Time"])
                d[f"{label} {part} avg speed"] = round(vavg, 2)
        # Hong–Rory diff의 구간 합계
        d["Σ Pelvis Δ(H–R)"]   = (df.loc[i0:i1, "Pelvis Δ(Hong–Rory)"].sum()).round(2)
        d["Σ Shoulder Δ(H–R)"] = (df.loc[i0:i1, "Shoulder Δ(Hong–Rory)"].sum()).round(2)
        rows.append(d)

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return df

def main():
    # ─── 1) 엑셀 파일 경로 설정 ───
    pro_file    = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golfer_file = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # ─── 2) 시간 벡터 생성 (1초 간격)
    times = [i for i in range(10)]

    # ─── 3) 골반 tilt 보고서 생성
    report = compute_tilt_report(pro_file, golfer_file, times)
    print(report)
if __name__ == "__main__":
    main()