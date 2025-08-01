# center_deviation.py

import numpy as np
import pandas as pd
from pathlib import Path

def col_letters_to_index(letters: str) -> int:
    """엑셀 컬럼 문자(A, B, ..., Z, AA, AB, ...)를 0-based 인덱스로 변환"""
    idx = 0
    for ch in letters.upper():
        idx = idx*26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def load_sheet(xlsx_path: Path) -> np.ndarray:
    """헤더 없는 엑셀 시트를 numpy 2D 배열로 읽어들임"""
    import pandas as _pd
    return _pd.read_excel(xlsx_path, header=None).values

def g(arr: np.ndarray, code: str) -> float:
    """코드 like 'H1', 'BP1', 'AL1' → 해당 (frame, col) 값을 float으로"""
    letters = ''.join(filter(str.isalpha, code))
    num     = int(''.join(filter(str.isdigit, code)))
    return float(arr[num-1, col_letters_to_index(letters)])

# ─────────────────────────────────────────────────────────────────────────────
# 1) 골반 중심점
# ─────────────────────────────────────────────────────────────────────────────
def _pelvis_center(arr: np.ndarray, i: int) -> np.ndarray:
    """왼쪽(H,I,J), 오른쪽(K,L,M)"""
    L = np.array([g(arr,f"H{i}"), g(arr,f"I{i}"), g(arr,f"J{i}")])
    R = np.array([g(arr,f"K{i}"), g(arr,f"L{i}"), g(arr,f"M{i}")])
    return (L+R)/2

# ─────────────────────────────────────────────────────────────────────────────
# 2) 무릎 중심점
# ─────────────────────────────────────────────────────────────────────────────
def _knee_center(arr: np.ndarray, i: int) -> np.ndarray:
    """왼쪽(BP,BQ,BR), 오른쪽(CB,CC,CD)"""
    L = np.array([g(arr,f"BP{i}"), g(arr,f"BQ{i}"), g(arr,f"BR{i}")])
    R = np.array([g(arr,f"CB{i}"), g(arr,f"CC{i}"), g(arr,f"CD{i}")])
    return (L+R)/2

# ─────────────────────────────────────────────────────────────────────────────
# 3) 어깨 중심점
# ─────────────────────────────────────────────────────────────────────────────
def _shoulder_center(arr: np.ndarray, i: int) -> np.ndarray:
    """왼쪽(AL,AM,AN), 오른쪽(BA,BB,BC)"""
    L = np.array([g(arr,f"AL{i}"), g(arr,f"AM{i}"), g(arr,f"AN{i}")])
    R = np.array([g(arr,f"BA{i}"), g(arr,f"BB{i}"), g(arr,f"BC{i}")])
    return (L+R)/2

# ─────────────────────────────────────────────────────────────────────────────
# 공통 로직: center_deviation_table
# ─────────────────────────────────────────────────────────────────────────────
def _compute_center_deviation(
    pro_path: Path,
    golfer_path: Path,
    center_fn: callable
) -> pd.DataFrame:
    """
    • index: ['1-4','4-7','7-10','Total']
    • columns: Rory X,Y,Z | Hong X,Y,Z | ΔX,ΔY,ΔZ
    """
    arr_p  = load_sheet(pro_path)
    arr_h  = load_sheet(golfer_path)
    # 10프레임이라 가정
    C_p = np.vstack([center_fn(arr_p, i) for i in range(1,11)])
    C_h = np.vstack([center_fn(arr_h, i) for i in range(1,11)])
    Bp, Bh = C_p[0], C_h[0]   # ADD=0번째 기준점

    seg_defs = {"1-4":(0,3),"4-7":(3,6),"7-10":(6,9)}
    rows = []
    for seg, (s,e) in seg_defs.items():
        Cp_avg = C_p[s:e].mean(axis=0)
        Ch_avg = C_h[s:e].mean(axis=0)
        dCp = Cp_avg - Bp
        dCh = Ch_avg - Bh
        dDf = dCh - dCp

        # 부호 반전 체크
        marks = ["!" if dCp[k]*dCh[k]<0 else "" for k in range(3)]
        row = {
            "Frame":   seg,
            "Rory X":  f"{dCp[0]:+.2f}",
            "Rory Y":  f"{dCp[1]:+.2f}",
            "Rory Z":  f"{dCp[2]:+.2f}",
            "Hong X":  f"{dCh[0]:+.2f}{marks[0]}",
            "Hong Y":  f"{dCh[1]:+.2f}{marks[1]}",
            "Hong Z":  f"{dCh[2]:+.2f}{marks[2]}",
            "ΔX (H-R)":f"{dDf[0]:+.2f}",
            "ΔY (H-R)":f"{dDf[1]:+.2f}",
            "ΔZ (H-R)":f"{dDf[2]:+.2f}",
        }
        rows.append(row)

    # Total 행: 세 구간 절대값 합계
    abs_sums = {}
    for col in rows[0].keys():
        if col=="Frame": continue
        # Hong 쪽 느낌표 제거
        vals = [r[col].rstrip("!") for r in rows]
        abs_sums[col] = sum(abs(float(v)) for v in vals)

    total = {"Frame":"Total"}
    total.update({c:f"{abs_sums[c]:.2f}" for c in abs_sums})
    rows.append(total)

    return pd.DataFrame(rows).set_index("Frame")

# ─────────────────────────────────────────────────────────────────────────────
# 외부 호출용 래퍼
# ─────────────────────────────────────────────────────────────────────────────
def compute_pelvis_center_deviation(pro_path: Path, golfer_path: Path) -> pd.DataFrame:
    return _compute_center_deviation(pro_path, golfer_path, _pelvis_center)

def compute_knee_center_deviation(pro_path: Path, golfer_path: Path) -> pd.DataFrame:
    return _compute_center_deviation(pro_path, golfer_path, _knee_center)

def compute_shoulder_center_deviation(pro_path: Path, golfer_path: Path) -> pd.DataFrame:
    return _compute_center_deviation(pro_path, golfer_path, _shoulder_center)

def compute_xyz_diff_summary(pro_path: Path, golfer_path: Path) -> pd.DataFrame:
    """
    XYZ 차이 통합표
    • index 없이 리셋된 DataFrame
    • columns: ['부위','구간','X 차이 (홍 - 로리)','Y 차이 (홍 - 로리)','Z 차이 (홍 - 로리)']
    • 구간 순서: 1-4, 4-7, 7-10, 부위 순서: 골반→어깨→무릎
    • Total 행은 제외
    """
    parts = [
        ("골반",   compute_pelvis_center_deviation),
        ("어깨",   compute_shoulder_center_deviation),
        ("무릎",   compute_knee_center_deviation),
    ]
    rows = []
    for 부위, func in parts:
        df = func(pro_path, golfer_path)
        for 구간 in ["1-4","4-7","7-10"]:
            dx = float(df.at[구간, "ΔX (H-R)"])
            dy = float(df.at[구간, "ΔY (H-R)"])
            dz = float(df.at[구간, "ΔZ (H-R)"])
            rows.append({
                "부위": 부위,
                "구간": 구간,
                "X 차이 (홍 - 로리)": round(dx, 2),
                "Y 차이 (홍 - 로리)": round(dy, 2),
                "Z 차이 (홍 - 로리)": round(dz, 2),
            })
    return pd.DataFrame(rows)


if __name__=="__main__":
    pro = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    hong = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    print("▶ Pelvis Deviation")
    print(compute_pelvis_center_deviation(pro, hong), "\n")

    print("▶ Knee Deviation")
    print(compute_knee_center_deviation(pro, hong), "\n")

    print("▶ Shoulder Deviation")
    print(compute_shoulder_center_deviation(pro, hong))

    print("▶ XYZ Diff Summary")
    print(compute_xyz_diff_summary(pro, hong).to_string(index=False))