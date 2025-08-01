# knee_movement.py

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

def compute_movement_table_Knee(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • 프레임 i→i+1 간 무릎 중심 좌표 이동량 ΔX, ΔY, ΔZ 계산
    • Rory vs Hong 부호 불일치 시 '!' 표시
    • 구간(1-4,4-7,7-10)의 합산 및 절대합, Total, TotalAbs, TotalXYZ 절대합 계산
    • index: ['1-2', …, f'{N-1}-{N}', '1-4', '4-7', '7-10', 'Total', 'TotalAbs', 'TotalXYZ']
    • columns: ['ΔX_Rory','ΔY_Rory','ΔZ_Rory','ΔX_Hong','ΔY_Hong','ΔZ_Hong']
    """
    # 1) 무릎 중심 좌표 함수
    def com(arr, n):
        x = 0.5 * (g(arr, f"BP{n}") + g(arr, f"CB{n}"))
        y = 0.5 * (g(arr, f"BQ{n}") + g(arr, f"CC{n}"))
        z = 0.5 * (g(arr, f"BR{n}") + g(arr, f"CD{n}"))
        return np.array([x, y, z], dtype=float)

    arr_r = load_sheet(path_pro)
    arr_h = load_sheet(path_golfer)
    N = arr_r.shape[0]

    # 2) 프레임 간 Δ 계산
    delta_r = [com(arr_r, i+1) - com(arr_r, i) for i in range(1, 10)]
    delta_h = [com(arr_h, i+1) - com(arr_h, i) for i in range(1, 10)]
    seg_keys = [f"{i}-{i+1}" for i in range(1, 10)]

    # 3) DataFrame 초기화
    mov = pd.DataFrame(index=seg_keys)
    for comp, label in zip([delta_r, delta_h], ['Rory', 'Hong']):
        dfp = pd.DataFrame(comp, index=seg_keys, columns=['ΔX','ΔY','ΔZ'])
        # 소수점 2자리 포맷
        for ax in ['X','Y','Z']:
            mov[f'Δ{ax}_{label}'] = dfp[f'Δ{ax}'].round(2)

    # 4) 부호 불일치 '!' 표시 (Hong만 표시)
    for ax in ['X','Y','Z']:
        for idx in seg_keys:
            xr = mov.at[idx, f'Δ{ax}_Rory']
            xh = mov.at[idx, f'Δ{ax}_Hong']
            if xr * xh < 0:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}!"
            else:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}"
            # Rory도 포맷 문자열로
            mov.at[idx, f'Δ{ax}_Rory'] = f"{xr:.2f}"

    # 5) 구간 합산 및 절대합 계산
    sections = {'1-4':(1,4), '4-7':(4,7), '7-10':(7,10)}
    for sec, (a,b) in sections.items():
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        for label in ['Rory','Hong']:
            for ax in ['X','Y','Z']:
                col = f'Δ{ax}_{label}'
                # 숫자만 추출
                vals = mov.loc[keys, col].astype(str).str.rstrip('!').astype(float)
                mov.at[sec, col] = round(vals.sum(), 2)
                # mov.at[sec, f'{col}_abs'] = round(vals.abs().sum(), 2)

    # 6) Total, TotalAbs, TotalXYZ 절대합
    # Total: signed sum
    for label in ['Rory','Hong']:
        for ax in ['X','Y','Z']:
            col = f'Δ{ax}_{label}'
            vals = mov.loc[seg_keys, col].astype(str).str.rstrip('!').astype(float)
            mov.at['Total', col] = round(vals.sum(), 2)
            mov.at['TotalAbs', col] = round(vals.abs().sum(), 2)
        # TotalXYZ 절대합
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        # mov.at['TotalXYZ', f'TotalAbsXYZ_{label}'] = mov.loc['TotalAbs', abs_cols].astype(float).sum()

    for label in ['Rory','Hong']:
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        total_xyz = mov.loc['TotalAbs', abs_cols].astype(float).sum()
        # ΔX 컬럼에만 값을 넣고, 나머지 축은 빈칸으로
        mov.at['TotalXYZ', f'ΔX_{label}'] = round(total_xyz, 2)

    return mov

def compute_movement_table_hips(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • 프레임 i→i+1 간 무릎 중심 좌표 이동량 ΔX, ΔY, ΔZ 계산
    • Rory vs Hong 부호 불일치 시 '!' 표시
    • 구간(1-4,4-7,7-10)의 합산 및 절대합, Total, TotalAbs, TotalXYZ 절대합 계산
    • index: ['1-2', …, f'{N-1}-{N}', '1-4', '4-7', '7-10', 'Total', 'TotalAbs', 'TotalXYZ']
    • columns: ['ΔX_Rory','ΔY_Rory','ΔZ_Rory','ΔX_Hong','ΔY_Hong','ΔZ_Hong']
    """
    # 1) 무릎 중심 좌표 함수
    def com(arr, n):
        x = 0.5 * (g(arr, f"H{n}") + g(arr, f"K{n}"))
        y = 0.5 * (g(arr, f"I{n}") + g(arr, f"L{n}"))
        z = 0.5 * (g(arr, f"J{n}") + g(arr, f"M{n}"))
        return np.array([x, y, z], dtype=float)

    arr_r = load_sheet(path_pro)
    arr_h = load_sheet(path_golfer)
    N = arr_r.shape[0]

    # 2) 프레임 간 Δ 계산
    delta_r = [com(arr_r, i+1) - com(arr_r, i) for i in range(1, 10)]
    delta_h = [com(arr_h, i+1) - com(arr_h, i) for i in range(1, 10)]
    seg_keys = [f"{i}-{i+1}" for i in range(1, 10)]

    # 3) DataFrame 초기화
    mov = pd.DataFrame(index=seg_keys)
    for comp, label in zip([delta_r, delta_h], ['Rory', 'Hong']):
        dfp = pd.DataFrame(comp, index=seg_keys, columns=['ΔX','ΔY','ΔZ'])
        # 소수점 2자리 포맷
        for ax in ['X','Y','Z']:
            mov[f'Δ{ax}_{label}'] = dfp[f'Δ{ax}'].round(2)

    # 4) 부호 불일치 '!' 표시 (Hong만 표시)
    for ax in ['X','Y','Z']:
        for idx in seg_keys:
            xr = mov.at[idx, f'Δ{ax}_Rory']
            xh = mov.at[idx, f'Δ{ax}_Hong']
            if xr * xh < 0:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}!"
            else:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}"
            # Rory도 포맷 문자열로
            mov.at[idx, f'Δ{ax}_Rory'] = f"{xr:.2f}"

    # 5) 구간 합산 및 절대합 계산
    sections = {'1-4':(1,4), '4-7':(4,7), '7-10':(7,10)}
    for sec, (a,b) in sections.items():
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        for label in ['Rory','Hong']:
            for ax in ['X','Y','Z']:
                col = f'Δ{ax}_{label}'
                # 숫자만 추출
                vals = mov.loc[keys, col].astype(str).str.rstrip('!').astype(float)
                mov.at[sec, col] = round(vals.sum(), 2)
                # mov.at[sec, f'{col}_abs'] = round(vals.abs().sum(), 2)

    # 6) Total, TotalAbs, TotalXYZ 절대합
    # Total: signed sum
    for label in ['Rory','Hong']:
        for ax in ['X','Y','Z']:
            col = f'Δ{ax}_{label}'
            vals = mov.loc[seg_keys, col].astype(str).str.rstrip('!').astype(float)
            mov.at['Total', col] = round(vals.sum(), 2)
            mov.at['TotalAbs', col] = round(vals.abs().sum(), 2)
        # TotalXYZ 절대합
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        # mov.at['TotalXYZ', f'TotalAbsXYZ_{label}'] = mov.loc['TotalAbs', abs_cols].astype(float).sum()

    for label in ['Rory','Hong']:
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        total_xyz = mov.loc['TotalAbs', abs_cols].astype(float).sum()
        # ΔX 컬럼에만 값을 넣고, 나머지 축은 빈칸으로
        mov.at['TotalXYZ', f'ΔX_{label}'] = round(total_xyz, 2)

    return mov

def compute_movement_table_sho(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • 프레임 i→i+1 간 무릎 중심 좌표 이동량 ΔX, ΔY, ΔZ 계산
    • Rory vs Hong 부호 불일치 시 '!' 표시
    • 구간(1-4,4-7,7-10)의 합산 및 절대합, Total, TotalAbs, TotalXYZ 절대합 계산
    • index: ['1-2', …, f'{N-1}-{N}', '1-4', '4-7', '7-10', 'Total', 'TotalAbs', 'TotalXYZ']
    • columns: ['ΔX_Rory','ΔY_Rory','ΔZ_Rory','ΔX_Hong','ΔY_Hong','ΔZ_Hong']
    """
    # 1) 무릎 중심 좌표 함수
    def com(arr, n):
        x = 0.5 * (g(arr, f"AL{n}") + g(arr, f"BA{n}"))
        y = 0.5 * (g(arr, f"AM{n}") + g(arr, f"BB{n}"))
        z = 0.5 * (g(arr, f"AN{n}") + g(arr, f"BC{n}"))
        return np.array([x, y, z], dtype=float)

    arr_r = load_sheet(path_pro)
    arr_h = load_sheet(path_golfer)
    N = arr_r.shape[0]

    # 2) 프레임 간 Δ 계산
    delta_r = [com(arr_r, i+1) - com(arr_r, i) for i in range(1, 10)]
    delta_h = [com(arr_h, i+1) - com(arr_h, i) for i in range(1, 10)]
    seg_keys = [f"{i}-{i+1}" for i in range(1, 10)]

    # 3) DataFrame 초기화
    mov = pd.DataFrame(index=seg_keys)
    for comp, label in zip([delta_r, delta_h], ['Rory', 'Hong']):
        dfp = pd.DataFrame(comp, index=seg_keys, columns=['ΔX','ΔY','ΔZ'])
        # 소수점 2자리 포맷
        for ax in ['X','Y','Z']:
            mov[f'Δ{ax}_{label}'] = dfp[f'Δ{ax}'].round(2)

    # 4) 부호 불일치 '!' 표시 (Hong만 표시)
    for ax in ['X','Y','Z']:
        for idx in seg_keys:
            xr = mov.at[idx, f'Δ{ax}_Rory']
            xh = mov.at[idx, f'Δ{ax}_Hong']
            if xr * xh < 0:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}!"
            else:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}"
            # Rory도 포맷 문자열로
            mov.at[idx, f'Δ{ax}_Rory'] = f"{xr:.2f}"

    # 5) 구간 합산 및 절대합 계산
    sections = {'1-4':(1,4), '4-7':(4,7), '7-10':(7,10)}
    for sec, (a,b) in sections.items():
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        for label in ['Rory','Hong']:
            for ax in ['X','Y','Z']:
                col = f'Δ{ax}_{label}'
                # 숫자만 추출
                vals = mov.loc[keys, col].astype(str).str.rstrip('!').astype(float)
                mov.at[sec, col] = round(vals.sum(), 2)
                # mov.at[sec, f'{col}_abs'] = round(vals.abs().sum(), 2)

    # 6) Total, TotalAbs, TotalXYZ 절대합
    # Total: signed sum
    for label in ['Rory','Hong']:
        for ax in ['X','Y','Z']:
            col = f'Δ{ax}_{label}'
            vals = mov.loc[seg_keys, col].astype(str).str.rstrip('!').astype(float)
            mov.at['Total', col] = round(vals.sum(), 2)
            mov.at['TotalAbs', col] = round(vals.abs().sum(), 2)
        # TotalXYZ 절대합
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        # mov.at['TotalXYZ', f'TotalAbsXYZ_{label}'] = mov.loc['TotalAbs', abs_cols].astype(float).sum()

    for label in ['Rory','Hong']:
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        total_xyz = mov.loc['TotalAbs', abs_cols].astype(float).sum()
        # ΔX 컬럼에만 값을 넣고, 나머지 축은 빈칸으로
        mov.at['TotalXYZ', f'ΔX_{label}'] = round(total_xyz, 2)

    return mov

def compute_movement_table_head(path_pro: Path, path_golfer: Path) -> pd.DataFrame:
    """
    • 프레임 i→i+1 간 무릎 중심 좌표 이동량 ΔX, ΔY, ΔZ 계산
    • Rory vs Hong 부호 불일치 시 '!' 표시
    • 구간(1-4,4-7,7-10)의 합산 및 절대합, Total, TotalAbs, TotalXYZ 절대합 계산
    • index: ['1-2', …, f'{N-1}-{N}', '1-4', '4-7', '7-10', 'Total', 'TotalAbs', 'TotalXYZ']
    • columns: ['ΔX_Rory','ΔY_Rory','ΔZ_Rory','ΔX_Hong','ΔY_Hong','ΔZ_Hong']
    """
    # 1) 무릎 중심 좌표 함수
    def com(arr, n):
        x = 0.5 * (g(arr, f"AC{n}") + g(arr, f"AC{n}") )
        y = 0.5 * (g(arr, f"AD{n}") + g(arr, f"AD{n}") )
        z = 0.5 * (g(arr, f"AE{n}") + g(arr, f"AE{n}") )
        return np.array([x, y, z], dtype=float)

    arr_r = load_sheet(path_pro)
    arr_h = load_sheet(path_golfer)
    N = arr_r.shape[0]

    # 2) 프레임 간 Δ 계산
    delta_r = [com(arr_r, i+1) - com(arr_r, i) for i in range(1, 10)]
    delta_h = [com(arr_h, i+1) - com(arr_h, i) for i in range(1, 10)]
    seg_keys = [f"{i}-{i+1}" for i in range(1, 10)]

    # 3) DataFrame 초기화
    mov = pd.DataFrame(index=seg_keys)
    for comp, label in zip([delta_r, delta_h], ['Rory', 'Hong']):
        dfp = pd.DataFrame(comp, index=seg_keys, columns=['ΔX','ΔY','ΔZ'])
        # 소수점 2자리 포맷
        for ax in ['X','Y','Z']:
            mov[f'Δ{ax}_{label}'] = dfp[f'Δ{ax}'].round(2)

    # 4) 부호 불일치 '!' 표시 (Hong만 표시)
    for ax in ['X','Y','Z']:
        for idx in seg_keys:
            xr = mov.at[idx, f'Δ{ax}_Rory']
            xh = mov.at[idx, f'Δ{ax}_Hong']
            if xr * xh < 0:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}!"
            else:
                mov.at[idx, f'Δ{ax}_Hong'] = f"{xh:.2f}"
            # Rory도 포맷 문자열로
            mov.at[idx, f'Δ{ax}_Rory'] = f"{xr:.2f}"

    # 5) 구간 합산 및 절대합 계산
    sections = {'1-4':(1,4), '4-7':(4,7), '7-10':(7,10)}
    for sec, (a,b) in sections.items():
        keys = [f"{i}-{i+1}" for i in range(a, b)]
        for label in ['Rory','Hong']:
            for ax in ['X','Y','Z']:
                col = f'Δ{ax}_{label}'
                # 숫자만 추출
                vals = mov.loc[keys, col].astype(str).str.rstrip('!').astype(float)
                mov.at[sec, col] = round(vals.sum(), 2)
                # mov.at[sec, f'{col}_abs'] = round(vals.abs().sum(), 2)

    # 6) Total, TotalAbs, TotalXYZ 절대합
    # Total: signed sum
    for label in ['Rory','Hong']:
        for ax in ['X','Y','Z']:
            col = f'Δ{ax}_{label}'
            vals = mov.loc[seg_keys, col].astype(str).str.rstrip('!').astype(float)
            mov.at['Total', col] = round(vals.sum(), 2)
            mov.at['TotalAbs', col] = round(vals.abs().sum(), 2)
        # TotalXYZ 절대합
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        # mov.at['TotalXYZ', f'TotalAbsXYZ_{label}'] = mov.loc['TotalAbs', abs_cols].astype(float).sum()

    for label in ['Rory','Hong']:
        abs_cols = [f'Δ{ax}_{label}' for ax in ['X','Y','Z']]
        total_xyz = mov.loc['TotalAbs', abs_cols].astype(float).sum()
        # ΔX 컬럼에만 값을 넣고, 나머지 축은 빈칸으로
        mov.at['TotalXYZ', f'ΔX_{label}'] = round(total_xyz, 2)

    return mov

def total_move(
    path_pro: Path,
    path_golfer: Path,
    pro_label: str = "Rory",
    golfer_label: str = "Hong",
) -> pd.DataFrame:
    """
    네 부위별로 1-4, 4-7, 7-10, Total 구간의
    절대 이동량(ΔX,ΔY,ΔZ 절댓값 합)을
    '구간','무릎 총 이동(Rory)','무릎 총 이동(Hong)',…,'머리 총 이동(Hong)' 형태로 반환
    """
    # 1) 네 부위에 대해 compute_movement_table_* 호출
    tables = {
        "무릎":    compute_movement_table_Knee(path_pro, path_golfer),
        "골반":    compute_movement_table_hips(path_pro, path_golfer),
        "어깨":    compute_movement_table_sho(path_pro, path_golfer),
        "머리":    compute_movement_table_head(path_pro, path_golfer),
    }

    segments = ["1-4","4-7","7-10","Total"]
    out = []
    for seg in segments:
        row = {"구간": seg}
        for part, df in tables.items():
            for label in [pro_label, golfer_label]:
                # 절대 이동량 = |ΔX|+|ΔY|+|ΔZ| 의 합
                if seg == "Total":
                    # TotalAbs 행에 이미 계산된 절대합이 들어있다고 가정
                    abs_val = df.at["TotalAbs", f"ΔX_{label}"] + \
                              df.at["TotalAbs", f"ΔY_{label}"] + \
                              df.at["TotalAbs", f"ΔZ_{label}"]
                else:
                    # 구간 프레임 키 리스트 (e.g. '1-4' → ['1-2','2-3','3-4'])
                    a,b = map(int, seg.split("-"))
                    keys = [f"{i}-{i+1}" for i in range(a, b)]
                    vals = []
                    for ax in ["X","Y","Z"]:
                        ser = df.loc[keys, f"Δ{ax}_{label}"] \
                               .astype(str).str.rstrip("!").astype(float)
                        vals.append(ser.abs().sum())
                    abs_val = sum(vals)
                # 소수 첫째자리까지 반올림
                row[f"{part} 총 이동({label}, cm)"] = round(abs_val, 2)
        out.append(row)

    summary_df = pd.DataFrame(out)
    # 원하는 컬럼 순서로 재배치
    cols = ["구간"]
    for part in ["무릎","골반","어깨","머리"]:
        for label in [pro_label, golfer_label]:
            cols.append(f"{part} 총 이동({label}, cm)")
    return summary_df[cols]

def total_move_ratio(
    path_pro: Path,
    path_golfer: Path,
    pro_label: str = "Rory",
    golfer_label: str = "Hong",
) -> pd.DataFrame:
    """
    네 부위별로 1-4, 4-7, 7-10, Total 구간의
    절대 이동량(ΔX,ΔY,ΔZ 절댓값 합)과
    구간별 절대 이동량 / Total 절대 이동량 *100 (%)
    를 계산해 반환합니다.
    """
    tables = {
        "무릎": compute_movement_table_Knee(path_pro, path_golfer),
        "골반": compute_movement_table_hips(path_pro, path_golfer),
        "어깨": compute_movement_table_sho(path_pro, path_golfer),
        "머리": compute_movement_table_head(path_pro, path_golfer),
    }
    segments = ["1-4","4-7","7-10","Total"]
    out = []
    # 1) 절대 이동량 수집
    for seg in segments:
        row = {"구간": seg}
        for part, df in tables.items():
            for label in [pro_label, golfer_label]:
                if seg == "Total":
                    abs_val = (
                        df.at["TotalAbs", f"ΔX_{label}"]
                      + df.at["TotalAbs", f"ΔY_{label}"]
                      + df.at["TotalAbs", f"ΔZ_{label}"]
                    )
                else:
                    a,b = map(int, seg.split("-"))
                    keys = [f"{i}-{i+1}" for i in range(a, b)]
                    vals = []
                    for ax in ["X","Y","Z"]:
                        ser = df.loc[keys, f"Δ{ax}_{label}"] \
                               .astype(str).str.rstrip("!").astype(float)
                        vals.append(ser.abs().sum())
                    abs_val = sum(vals)
                row[f"{part} 절대이동({label},cm)"] = round(abs_val, 2)
        out.append(row)

    summary = pd.DataFrame(out)
    # 2) 비율(%) 계산
    for part in ["무릎","골반","어깨","머리"]:
        for label in [pro_label, golfer_label]:
            abs_col = f"{part} 절대이동({label},cm)"
            pct_col = f"{part} 이동비율({label},%)"
            # Total 절대 이동량
            total = summary.loc[summary['구간']=="Total", abs_col].iloc[0]
            summary[pct_col] = (summary[abs_col] / total * 100).round(2)

    # 3) 컬럼 순서 재배치 (절대이동, 이동비율 순서로)
    cols = ["구간"]
    for part in ["무릎","골반","어깨","머리"]:
        for label in [pro_label, golfer_label]:
            cols += [f"{part} 이동비율({label},%)"]
    summary = summary[cols]

    return summary

if __name__ == "__main__":
    # 테스트용 예시 경로
    pro = Path("/Users/park_sh/Desktop/sim_pro/driver/Rory McIlroy/first_data_transition.xlsx")
    golf = Path("/Users/park_sh/Desktop/sim_pro/test/sample_first.xlsx")

    # df_movement = compute_movement_table_head(pro, golf)
    # print("=== Movement Table ===")
    df_movement = total_move_ratio(pro, golf)
    print("=== Movement Summary Table ===")
    print(df_movement)
