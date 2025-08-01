# zreport.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Excel 로드 & g() ─────────────────────────────────────────────────────────
def col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    import pandas as _pd
    return _pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ─── Z 시리즈 생성 ─────────────────────────────────────────────────────────────
_marker_map_z = {
    "Ankle":    ("CA","CM"),
    "Knee":     ("BR","CD"),
    "Waist":    ("J","M"),
    "Shoulder": ("AN","BC"),
    "Head":     ("AE", None),
}
_frames = ["ADD","BH","BH2","TOP","TR","DH","IMP","FH1","FH2","Finish"]

def _make_z_df(path: Path, label: str) -> pd.DataFrame:
    arr = load_sheet(path)
    data = {"Frame": _frames}
    for part, (L, R) in _marker_map_z.items():
        vals = []
        for i in range(1, len(_frames)+1):
            if R is None:
                z = g(arr, f"{L}{i}")
            else:
                z = (g(arr, f"{L}{i}") + g(arr, f"{R}{i}")) / 2
            vals.append(round(z, 2))
        data[f"{label} {part} Z"] = vals
    return pd.DataFrame(data)

# ─── 전체 리포트 계산 함수 ─────────────────────────────────────────────────────
def compute_z_report(
    pro_path: Path,
    golfer_path: Path,
    pro_label: str = "Rory",
    golfer_label: str = "Hong"
) -> pd.DataFrame:
    # 1) 기본 프레임별 Z값
    df_pro = _make_z_df(pro_path, pro_label)
    df_go  = _make_z_df(golfer_path, golfer_label)
    df = pd.merge(df_pro, df_go, on="Frame")

    # 2) 부위 간 상대 위치
    # for player in [pro_label, golfer_label]:
    #     for a,b in [("Ankle","Knee"), ("Knee","Waist"), ("Waist","Shoulder")]:
    #         df[f"{player} {b} vs {a}"] = (
    #             df[f"{player} {b} Z"] - df[f"{player} {a} Z"]
    #         ).round(2)

    # 3) 1-4, 4-7, 7-10 스윙 구간 변화량 계산
    numeric = [c for c in df.columns if c!="Frame"]
    d1 = (df.loc[3, numeric] - df.loc[0, numeric]).round(2)  # ADD→TOP
    d2 = (df.loc[6, numeric] - df.loc[3, numeric]).round(2)  # TOP→IMP
    d3 = (df.loc[9, numeric] - df.loc[6, numeric]).round(2)  # IMP→Finish

    # 4) 각 구간 행 생성
    row1 = pd.concat([pd.Series({"Frame":"1-4"}),   d1])
    row2 = pd.concat([pd.Series({"Frame":"4-7"}),   d2])
    row3 = pd.concat([pd.Series({"Frame":"7-10"}),  d3])
    # 5) Total = d1 + d2 + d3
    d_total = (abs(d1) + abs(d2) + abs(d3)).round(2)
    row4    = pd.concat([pd.Series({"Frame":"Total"}), d_total])

    # 6) 데이터프레임에 붙이기
    df = pd.concat([df,
                    pd.DataFrame([row1, row2, row3, row4])],
                   ignore_index=True)

    # 7) 부호 불일치 '!' 표시 (1-4,4-7,7-10 세 구간만)
    n = df.shape[0]
    seg_idxs = [n-4, n-3, n-2]   # 이 인덱스들이 1-4,4-7,7-10
    for col in df.columns:
        if col.startswith(golfer_label) and col!="Frame":
            pro_col = col.replace(golfer_label, pro_label)
            ci = df.columns.get_loc(col)
            pi = df.columns.get_loc(pro_col)
            for ix in seg_idxs:
                p = float(df.iat[ix, pi])
                h = float(df.iat[ix, ci])
                df.iat[ix, ci] = f"{h:+.2f}!" if p*h < 0 else f"{h:+.2f}"
                df.iat[ix, pi] = f"{p:+.2f}"

    return df

def _make_z_df(path: Path, label: str) -> pd.DataFrame:
    arr = load_sheet(path)
    data = {"Frame": _frames}
    for part, (L, R) in _marker_map_z.items():
        vals = []
        for i in range(1, len(_frames)+1):
            if R is None:
                z = g(arr, f"{L}{i}")
            else:
                z = (g(arr, f"{L}{i}") + g(arr, f"{R}{i}")) / 2
            vals.append(round(z, 2))
        data[f"{label} {part} Z"] = vals
    return pd.DataFrame(data)

# ─── 전체 리포트 계산 함수 ─────────────────────────────────────────────────────
_marker_map_x = {
    "Knee":     ("BP","CB"),
    "Waist":    ("H","K"),
    "Shoulder": ("AL","BA"),
    "Head":     ("AC", None),
}
def _make_x_df(path: Path, label: str) -> pd.DataFrame:
    arr = load_sheet(path)
    data = {"Frame": _frames}
    for part, (L, R) in _marker_map_x.items():
        vals = []
        for i in range(1, len(_frames)+1):
            if R is None:
                z = g(arr, f"{L}{i}")
            else:
                z = (g(arr, f"{L}{i}") + g(arr, f"{R}{i}")) / 2
            vals.append(round(z, 2))
        data[f"{label} {part} X"] = vals
    return pd.DataFrame(data)
def compute_x_report(
    pro_path: Path,
    golfer_path: Path,
    pro_label: str = "Rory",
    golfer_label: str = "Hong"
) -> pd.DataFrame:
    # 1) 기본 프레임별 Z값
    df_pro = _make_x_df(pro_path, pro_label)
    df_go  = _make_x_df(golfer_path, golfer_label)
    df = pd.merge(df_pro, df_go, on="Frame")

    # 2) 부위 간 상대 위치
    # for player in [pro_label, golfer_label]:
    #     for a,b in [("Ankle","Knee"), ("Knee","Waist"), ("Waist","Shoulder")]:
    #         df[f"{player} {b} vs {a}"] = (
    #             df[f"{player} {b} Z"] - df[f"{player} {a} Z"]
    #         ).round(2)

    # 3) 1-4, 4-7, 7-10 스윙 구간 변화량 계산
    numeric = [c for c in df.columns if c!="Frame"]
    d1 = (df.loc[3, numeric] - df.loc[0, numeric]).round(2)  # ADD→TOP
    d2 = (df.loc[6, numeric] - df.loc[3, numeric]).round(2)  # TOP→IMP
    d3 = (df.loc[9, numeric] - df.loc[6, numeric]).round(2)  # IMP→Finish

    # 4) 각 구간 행 생성
    row1 = pd.concat([pd.Series({"Frame":"1-4"}),   d1])
    row2 = pd.concat([pd.Series({"Frame":"4-7"}),   d2])
    row3 = pd.concat([pd.Series({"Frame":"7-10"}),  d3])
    # 5) Total = d1 + d2 + d3
    d_total = (abs(d1) + abs(d2) + abs(d3)).round(2)
    row4    = pd.concat([pd.Series({"Frame":"Total"}), d_total])

    # 6) 데이터프레임에 붙이기
    df = pd.concat([df,
                    pd.DataFrame([row1, row2, row3, row4])],
                   ignore_index=True)

    # 7) 부호 불일치 '!' 표시 (1-4,4-7,7-10 세 구간만)
    n = df.shape[0]
    seg_idxs = [n-4, n-3, n-2]   # 이 인덱스들이 1-4,4-7,7-10
    for col in df.columns:
        if col.startswith(golfer_label) and col!="Frame":
            pro_col = col.replace(golfer_label, pro_label)
            ci = df.columns.get_loc(col)
            pi = df.columns.get_loc(pro_col)
            for ix in seg_idxs:
                p = float(df.iat[ix, pi])
                h = float(df.iat[ix, ci])
                df.iat[ix, ci] = f"{h:+.2f}!" if p*h < 0 else f"{h:+.2f}"
                df.iat[ix, pi] = f"{p:+.2f}"

    return df

_marker_map_y = {
    "Knee":     ("BQ","CC"),
    "Waist":    ("I","L"),
    "Shoulder": ("AM","BB"),
    "Head":     ("AD", None),
}
def _make_y_df(path: Path, label: str) -> pd.DataFrame:
    arr = load_sheet(path)
    data = {"Frame": _frames}
    for part, (L, R) in _marker_map_y.items():
        vals = []
        for i in range(1, len(_frames)+1):
            if R is None:
                z = g(arr, f"{L}{i}")
            else:
                z = (g(arr, f"{L}{i}") + g(arr, f"{R}{i}")) / 2
            vals.append(round(z, 2))
        data[f"{label} {part} Y"] = vals
    return pd.DataFrame(data)
def compute_y_report(
    pro_path: Path,
    golfer_path: Path,
    pro_label: str = "Rory",
    golfer_label: str = "Hong"
) -> pd.DataFrame:
    # 1) 기본 프레임별 Z값
    df_pro = _make_y_df(pro_path, pro_label)
    df_go  = _make_y_df(golfer_path, golfer_label)
    df = pd.merge(df_pro, df_go, on="Frame")

    # 2) 부위 간 상대 위치
    # for player in [pro_label, golfer_label]:
    #     for a,b in [("Ankle","Knee"), ("Knee","Waist"), ("Waist","Shoulder")]:
    #         df[f"{player} {b} vs {a}"] = (
    #             df[f"{player} {b} Z"] - df[f"{player} {a} Z"]
    #         ).round(2)

    # 3) 1-4, 4-7, 7-10 스윙 구간 변화량 계산
    numeric = [c for c in df.columns if c!="Frame"]
    d1 = (df.loc[3, numeric] - df.loc[0, numeric]).round(2)  # ADD→TOP
    d2 = (df.loc[6, numeric] - df.loc[3, numeric]).round(2)  # TOP→IMP
    d3 = (df.loc[9, numeric] - df.loc[6, numeric]).round(2)  # IMP→Finish

    # 4) 각 구간 행 생성
    row1 = pd.concat([pd.Series({"Frame":"1-4"}),   d1])
    row2 = pd.concat([pd.Series({"Frame":"4-7"}),   d2])
    row3 = pd.concat([pd.Series({"Frame":"7-10"}),  d3])
    # 5) Total = d1 + d2 + d3
    d_total = (abs(d1) + abs(d2) + abs(d3)).round(2)
    row4    = pd.concat([pd.Series({"Frame":"Total"}), d_total])

    # 6) 데이터프레임에 붙이기
    df = pd.concat([df,
                    pd.DataFrame([row1, row2, row3, row4])],
                   ignore_index=True)

    # 7) 부호 불일치 '!' 표시 (1-4,4-7,7-10 세 구간만)
    n = df.shape[0]
    seg_idxs = [n-4, n-3, n-2]   # 이 인덱스들이 1-4,4-7,7-10
    for col in df.columns:
        if col.startswith(golfer_label) and col!="Frame":
            pro_col = col.replace(golfer_label, pro_label)
            ci = df.columns.get_loc(col)
            pi = df.columns.get_loc(pro_col)
            for ix in seg_idxs:
                p = float(df.iat[ix, pi])
                h = float(df.iat[ix, ci])
                df.iat[ix, ci] = f"{h:+.2f}!" if p*h < 0 else f"{h:+.2f}"
                df.iat[ix, pi] = f"{p:+.2f}"

    return df

# ─── 시각화 함수 ───────────────────────────────────────────────────────────────
def plot_swings(
    df: pd.DataFrame,
    pro_label: str = "Rory",
    golfer_label: str = "Hong"
) -> None:
    parts = ["Ankle","Knee","Waist","Shoulder","Head"]
    last2 = df.iloc[-2:]
    r1 = last2.iloc[0][[f"{pro_label} {p} Z" for p in parts]].astype(float)
    r2 = last2.iloc[1][[f"{pro_label} {p} Z" for p in parts]].astype(float)
    h1 = last2.iloc[0][[f"{golfer_label} {p} Z" for p in parts]].astype(float)
    h2 = last2.iloc[1][[f"{golfer_label} {p} Z" for p in parts]].astype(float)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.plot(parts, r1, marker="o", label=f"{pro_label} 1-4")
    ax1.plot(parts, h1, marker="s", label=f"{golfer_label} 1-4")
    ax1.set_title("Backswing (1→4)"); ax1.grid(True); ax1.legend()

    ax2.plot(parts, r2, marker="o", linestyle="--", label=f"{pro_label} 4-7")
    ax2.plot(parts, h2, marker="s", linestyle="--", label=f"{golfer_label} 4-7")
    ax2.set_title("Downswing (4→7)"); ax2.grid(True); ax2.legend()

    plt.tight_layout()
    plt.show()

# ─── 직접 실행 시 ───────────────────────────────────────────────────────────
def main():
    pro_file    = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golfer_file = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    df = compute_y_report(pro_file, golfer_file)
    print(df)
    # plot_swings(df)

if __name__ == "__main__":
    main()
