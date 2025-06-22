#!/usr/bin/env python3
# run_all.py (상위 폴더에 저장)

import sys
import subprocess
from pathlib import Path
import pandas as pd

def main():
    base_dir = Path(__file__).resolve().parent

    # 1) 하위 폴더 중에 main.py가 있는 폴더를 모두 찾기
    child_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "main.py").exists()]

    excel_paths = []

    # 2) 각 폴더에서 main.py 실행 (cwd를 해당 폴더로 지정)
    for d in child_dirs:
        script = d / "main.py"
        print(f"▶️  실행 중: {script}")
        subprocess.run([sys.executable, script.name], cwd=d, check=True)
        # main.py가 출력한 xlsx 파일(들)을 모아서 리스트에 추가
        excels = list(d.glob("*.xlsx"))
        if not excels:
            print(f"⚠️  엑셀 파일이 발견되지 않음: {d}")
        excel_paths.extend(excels)

    # 3) 한 개의 워크북으로 합치기
    out_file = base_dir / "combined_summary.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for excel in excel_paths:
            xls = pd.ExcelFile(excel)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                # 시트 이름은 "<폴더명>_<원본시트명>" (31자 제한 고려)
                sheet_name = f"{excel.parent.name}_{sheet}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"▶️ '{out_file.name}' 생성 완료 ({len(excel_paths)}개의 파일을 합침)")

if __name__ == "__main__":
    main()
