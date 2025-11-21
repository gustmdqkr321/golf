# app.py
from pathlib import Path
import io
import pandas as pd
import streamlit as st
from core.loader import discover_sections

# app.py (상단 임포트 밑)
import io, re
import pandas as pd


# === 자동 탐색용 고정 파일명 ===
RAINBOW_FILENAME = "first_data_transition.xlsx"  # 무지개(기존) 엑셀
GS_FILENAME      = "CsvExport.csv"               # GS CSV

from pathlib import Path

def _find_file(root: str | Path, filename: str, recursive: bool = True) -> Path | None:
    """root 폴더에서 filename을 찾는다. recursive=True면 하위 폴더도 탐색."""
    try:
        base = Path(root).expanduser()
        if not base.exists():
            return None
        direct = base / filename
        if direct.exists():
            return direct
        if recursive:
            for p in base.rglob(filename):
                return p
    except Exception:
        pass
    return None


# 세션 저장소 초기화
if "section_tables" not in st.session_state:
    st.session_state["section_tables"] = {}   # {section_id: {"title": str, "tables": dict[str, DataFrame]}}

# 시트명 안전화
def _safe_sheet(name: str, used: set[str]) -> str:
    s = re.sub(r'[\\/\?\*\[\]\:\'"]', '', str(name)).strip()
    s = (s or "Sheet").replace(' ', '_')[:31]
    base, i = s, 1
    while s in used:
        suf = f"_{i}"
        s = (base[:31-len(suf)] if len(base) > 31-len(suf) else base) + suf
        i += 1
    used.add(s); 
    return s
   
def _write_section_sheet(writer: pd.ExcelWriter, sheet_name: str, tables: dict[str, pd.DataFrame]):
    wb = writer.book
    base_fmt     = wb.add_format({'border': 1, 'border_color': '#000000'})                 # ★ 모든 셀 기본
    num_fmt    = wb.add_format({'num_format': '0.00'})
    title_fmt  = wb.add_format({'bold': True, 'font_size': 12})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
    red_fill   = wb.add_format({'bg_color': '#FFC7CE'})
    yellow_fill= wb.add_format({'bg_color': '#FFEB9C'})  # ← 추가
    DIFF_THRESH = 0.30  # 30% 임계치 (원하면 사이드바 옵션으로 빼도 됨)

    def _col_letter(idx: int) -> str:
        s = ""; idx0 = idx
        while True:
            s = chr(idx0 % 26 + 65) + s
            idx0 = idx0 // 26 - 1
            if idx0 < 0:
                break
        return s

    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    cur_row = 0
    for name, df in tables.items():
        ws.write(cur_row, 0, str(name), title_fmt)
        cur_row += 1

        df.to_excel(writer, sheet_name=sheet_name, startrow=cur_row, startcol=0, index=False, header=True)

        n_rows, n_cols = df.shape
        for c in range(n_cols):
            ws.write(cur_row, c, df.columns[c], header_fmt)

        ws.set_column(0, max(0, n_cols-1), 14, num_fmt)
        # ── 가로형 하이라이트 (Ama 행만) ─────────────────────────────────
        # label_col: '구분' 또는 '항목' 중 하나를 자동 사용
        # ── (우선) 가로형: '구분' 또는 '항목' + 숫자 프레임 컬럼 ─────────────
        handled_horizontal = False
        label_col = next((c for c in ["구분", "항목"] if c in df.columns), None)
        if label_col is not None:
            frame_cols = [c for c in df.columns if c != label_col and str(c).isdigit()]
            if frame_cols:
                handled_horizontal = True

                # (NEW) 라벨 정규화: 프로/일반(=Ama) 대소문자/영문/한글/구분자 처리
                import re
                def _norm_role(x: object) -> str | None:
                    s = "" if x is None else str(x).strip()
                    # "L · Pro", "R.Pro", "프로", "Ama" 등 → 마지막 토큰을 역할 후보로
                    parts = re.split(r"\s*[·\.\|\-:]\s*", s.replace(" ", ""))
                    cand = parts[-1].lower() if parts else s.lower()
                    if cand.startswith("pro") or cand == "프로":
                        return "프로"
                    if cand.startswith("ama") or cand == "일반":
                        return "일반"
                    return None

                # 행 라벨을 정규화해서 r_pro / r_ama 찾기
                r_pro = r_ama = None
                for ridx, v in df[label_col].items():
                    role = _norm_role(v)
                    if role == "프로" and r_pro is None:
                        r_pro = int(ridx)
                    elif role == "일반" and r_ama is None:
                        r_ama = int(ridx)

                # 단순 2행 비교(프로/일반)가 잡히면 Ama 행만 색칠
                if r_pro is not None and r_ama is not None:
                    data_start = cur_row + 1  # 헤더 바로 아래
                    excel_row_pro = data_start + r_pro
                    excel_row_ama = data_start + r_ama

                    for col_name in frame_cols:
                        c_idx = df.columns.get_loc(col_name)
                        col_letter = _col_letter(c_idx)

                        # 빨강: 부호 반대 (Ama 셀만)
                        formula_red = f'=${col_letter}{excel_row_pro+1}*${col_letter}{excel_row_ama+1}<0'
                        ws.conditional_format(excel_row_ama, c_idx, excel_row_ama, c_idx, {
                            'type': 'formula', 'criteria': formula_red, 'format': red_fill
                        })

                        # 노랑: 부호 같고 상대차이 ≥ 임계치 (Ama 셀만)
                        formula_yellow = (
                            f'=AND('
                            f'${col_letter}{excel_row_pro+1}*${col_letter}{excel_row_ama+1}>=0,'
                            f'IF(MAX(ABS(${col_letter}{excel_row_pro+1}),ABS(${col_letter}{excel_row_ama+1}))=0,'
                            f'FALSE,'
                            f'ABS(${col_letter}{excel_row_pro+1}-${col_letter}{excel_row_ama+1})/'
                            f'MAX(ABS(${col_letter}{excel_row_pro+1}),ABS(${col_letter}{excel_row_ama+1}))>={DIFF_THRESH}'
                            f'))'
                        )
                        ws.conditional_format(excel_row_ama, c_idx, excel_row_ama, c_idx, {
                            'type': 'formula', 'criteria': formula_yellow, 'format': yellow_fill
                        })

                    cur_row += n_rows + 1 + 2
                    continue  # 다음 테이블로

                # (측면 L/R + 역할 혼합 라벨 케이스는 기존 블록이 처리)


                # (기존) 측면(L/R)+역할 라벨 조합 처리 블록이 있다면 여기서 그대로 유지

        # ── 가로형 끝 ─────────────────────────────────────────────────────


        # ── (기존) 세로형: Pro↔Ama 컬럼 쌍 찾고 Ama 컬럼만 칠함 ────────────
        headers = list(map(str, df.columns))
        col_index = {h: i for i, h in enumerate(headers)}
        pairs = []
        for i, h in enumerate(headers):
            if "프로" in h:
                h_ama = h.replace("프로", "일반")
                if h_ama in col_index:
                    pairs.append((i, col_index[h_ama]))
            if "Pro" in h and "프로" not in h:
                h_ama2 = h.replace("Pro", "Ama")
                if h_ama2 in col_index:
                    pairs.append((i, col_index[h_ama2]))
        seen, unique_pairs = set(), []
        for p, a in pairs:
            key = tuple(sorted((p, a)))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((p, a))

        data_start = cur_row + 1
        data_end   = cur_row + n_rows
        for p_idx, a_idx in unique_pairs:
            p_col = _col_letter(p_idx)
            a_col = _col_letter(a_idx)

            for r in range(data_start, data_end + 1):
                excel_r = r + 1
                # 빨강: 부호 반대 → Ama만 칠함
                formula_red = f'=${p_col}{excel_r}*${a_col}{excel_r}<0'
                ws.conditional_format(r, a_idx, r, a_idx, {
                    'type': 'formula', 'criteria': formula_red, 'format': red_fill
                })
                # 노랑: 부호 같고, 상대차이 ≥ 임계치 → Ama만 칠함
                formula_yellow = (
                    f'=AND('
                    f'${p_col}{excel_r}*${a_col}{excel_r}>=0,'
                    f'IF(MAX(ABS(${p_col}{excel_r}),ABS(${a_col}{excel_r}))=0,'
                    f'FALSE,'
                    f'ABS(${p_col}{excel_r}-${a_col}{excel_r})/MAX(ABS(${p_col}{excel_r}),ABS(${a_col}{excel_r}))>={DIFF_THRESH}'
                    f'))'
                )
                ws.conditional_format(r, a_idx, r, a_idx, {
                    'type': 'formula', 'criteria': formula_yellow, 'format': yellow_fill
                })

        cur_row += n_rows + 1 + 2




# 섹션에서 만든 표 dict를 마스터에 등록
def register_section(section_id: str, section_title: str, tables: dict[str, pd.DataFrame]):
    st.session_state["section_tables"][section_id] = {
        "title": section_title,
        "tables": tables,
    }

st.set_page_config(page_title="Modular Streamlit App", layout="wide")

# ── 프로젝트 경로 ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SECTIONS_DIR = BASE_DIR / "sections"

# ── 코드 내 디폴트 파일 경로(여기만 바꾸면 됨) ───────────────────────────────
USE_CODE_DEFAULTS = True  # 업로드 없을 때 코드 디폴트 사용 여부

# 무지개(기존) 엑셀
DEFAULT_PRO_PATH = "/Users/park_sh/Desktop/sim_pro/레퍼/test/rory.xlsx"
DEFAULT_AMA_PATH = "/Users/park_sh/Desktop/sim_pro/레퍼/test/hong.xlsx"

# GS CSV (프로/일반)
DEFAULT_GS_PRO_PATH = "/Users/park_sh/Desktop/sim_pro/레퍼/test/CsvExport_rory.csv"
DEFAULT_GS_AMA_PATH = "/Users/park_sh/Desktop/sim_pro/레퍼/test/CsvExport_hong.csv"

# ── 파일 로더 (xlsx/csv/xls/xlsb) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_xlsx_to_array(file_or_path):
    """
    - UploadedFile/BytesIO/Path 모두 지원
    - 반환: numpy.ndarray (header=None)
    """
    name = getattr(file_or_path, "name", str(file_or_path))
    suffix = Path(name).suffix.lower()

    fobj = file_or_path
    if hasattr(file_or_path, "getvalue"):  # UploadedFile
        fobj = io.BytesIO(file_or_path.getvalue())

    try:
        if suffix in (".xlsx", ".xlsm", ".xltx", ".xltm"):
            try:
                import openpyxl  # noqa: F401
            except ImportError:
                st.error("`.xlsx`를 읽으려면 `openpyxl>=3.1.5`가 필요합니다.")
                return None
            df = pd.read_excel(fobj, header=None, engine="openpyxl")

        elif suffix == ".csv":
            # CSV는 구분자 자동 감지 + python 엔진 + 깨진 줄은 건너뛰기
            # (C 엔진은 엄격해서 "Expected 1 fields..." 같은 오류가 잘 납니다)
            try:
                df = pd.read_csv(
                    fobj,
                    header=None,
                    sep=None,                # 구분자 자동 감지
                    engine="python",         # 유연한 파서
                    on_bad_lines="skip",     # 비정상 라인은 건너뛰기
                    skipinitialspace=True,   # 구분자 뒤 공백 무시
                    encoding_errors="ignore" # 깨진 인코딩은 무시
                )
            except Exception:
                # 재시도: 흔한 구분자들을 순차적으로 시도
                if hasattr(fobj, "seek"):
                    fobj.seek(0)
                for sep_try in [",", ";", "\t", "|"]:
                    try:
                        df = pd.read_csv(
                            fobj,
                            header=None,
                            sep=sep_try,
                            engine="python",
                            on_bad_lines="skip",
                            skipinitialspace=True,
                            encoding_errors="ignore",
                        )
                        break
                    except Exception:
                        if hasattr(fobj, "seek"):
                            fobj.seek(0)
                else:
                    raise  # 모두 실패하면 원래 예외 올림


        elif suffix == ".xls":
            try:
                import xlrd  # noqa: F401
            except ImportError:
                st.error("`.xls`를 읽으려면 `xlrd<2.0`가 필요합니다.")
                return None
            df = pd.read_excel(fobj, header=None, engine="xlrd")

        elif suffix == ".xlsb":
            try:
                import pyxlsb  # noqa: F401
            except ImportError:
                st.error("`.xlsb`를 읽으려면 `pyxlsb`가 필요합니다.")
                return None
            df = pd.read_excel(fobj, header=None, engine="pyxlsb")

        else:
            st.error(f"지원하지 않는 파일 형식입니다: {suffix}")
            return None

        return df.values
    except Exception as e:
        st.exception(e)
        return None

def try_read_default(p: str | Path | None):
    if not p:
        return None, None
    p = Path(p).expanduser()
    if not p.exists():
        st.sidebar.warning(f"디폴트 파일이 없습니다: {p}")
        return None, None
    try:
        return read_xlsx_to_array(p), p.name
    except Exception as e:
        st.sidebar.error(f"디폴트 파일 읽기 실패: {p} ({e})")
        return None, None

# ── CSV 로더(DF 반환) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_df_robust(file_or_path, header=None, **kwargs):
    fobj = file_or_path
    if hasattr(file_or_path, "getvalue"):
        fobj = io.BytesIO(file_or_path.getvalue())
    try:
        df = pd.read_csv(
            fobj,
            header=header,
            sep=None,
            engine="python",
            on_bad_lines="skip",
            skipinitialspace=True,
            encoding_errors="ignore",
            **kwargs,             # ← 추가
        )
        return df
    except Exception:
        if hasattr(fobj, "seek"): fobj.seek(0)
        for sep_try in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(
                    fobj,
                    header=header,
                    sep=sep_try,
                    engine="python",
                    on_bad_lines="skip",
                    skipinitialspace=True,
                    encoding_errors="ignore",
                    **kwargs,         # ← 추가
                )
                return df
            except Exception:
                if hasattr(fobj, "seek"): fobj.seek(0)
        raise

def try_read_csv_default(p: str | Path | None):
    if not p:
        return None, None
    p = Path(p).expanduser()
    if not p.exists():
        st.sidebar.warning(f"GS 디폴트 CSV가 없습니다: {p}")
        return None, None
    try:
        return read_csv_df_robust(p, header=0), p.name
    except Exception as e:
        st.sidebar.error(f"GS CSV 읽기 실패: {p} ({e})")
        return None, None
import io
import pandas as pd
import streamlit as st
# app.py 등 공용 로더 파일에 넣으세요
import io, csv, pandas as pd

def _sniff_csv(text: str):
    lines = text.splitlines()
    # 1) 'sep=,' 같은 엑셀 헤더 처리
    for i, ln in enumerate(lines[:5]):
        low = ln.strip().lower()
        if low.startswith("sep=") and len(low) >= 5:
            sep = ln.strip()[4:5]
            return sep, i + 1  # 다음 줄부터 데이터
    # 2) 가장 안정적인 구분자 추정
    candidates = [",", ";", "\t", "|"]
    best_sep, best_score, start_row = ",", -1, 0
    for sep in candidates:
        counts = []
        for ln in lines:
            if not ln.strip():
                counts.append(0)
                continue
            counts.append(ln.count(sep))
        pos = [i for i, c in enumerate(counts) if c > 0]
        if not pos:
            continue
        sr = pos[0]
        avg = sum(counts[i] for i in pos) / len(pos)
        score = avg - 0.1 * sr
        if score > best_score:
            best_sep, best_score, start_row = sep, score, sr
    return best_sep, start_row

@st.cache_data(show_spinner=False)
def read_gs_csv_raw(file_or_path, sep: str | None = None) -> pd.DataFrame:
    """
    GS CSV → DataFrame(열 절대 삭제 X, 행 길이 패딩으로 균일화)
    - 구분자 sep이 없으면 자동 스니핑
    - 'sep=,' 라인 자동 무시
    - 모든 행을 '최대 열 수'로 맞추어 우측을 "" 로 패딩
    """
    # 1) 바이트 → 텍스트
    if hasattr(file_or_path, "getvalue"):  # UploadedFile
        raw = file_or_path.getvalue()
    else:
        with open(file_or_path, "rb") as f:
            raw = f.read()
    try:
        text = raw.decode("utf-8-sig", errors="ignore")
    except Exception:
        text = raw.decode("utf-8", errors="ignore")

    # 2) 구분자/시작행 추정
    sniffed_sep, start_row = _sniff_csv(text)
    use_sep = sep if sep else sniffed_sep

    # 3) csv.reader로 직접 읽어서 모든 행 길이를 동일화
    sio = io.StringIO(text)
    r = csv.reader(sio, delimiter=use_sep)
    all_rows = list(r)

    # header 없는 raw라 가정하고 start_row부터 데이터
    data_rows = all_rows[start_row:]

    # 최대 열 수
    max_len = max((len(row) for row in data_rows), default=0)

    # 우측 패딩(빈 셀 보존)
    for row in data_rows:
        if len(row) < max_len:
            row += [""] * (max_len - len(row))

    # 4) DataFrame 화 (절대 dropna로 열 삭제하지 말 것!)
    df = pd.DataFrame(data_rows, dtype=str)
    # 필요하다면 트리밍만
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df




def try_read_gs_default(p: str | Path | None, sep=","):
    if not p:
        return None, None
    p = Path(p).expanduser()
    if not p.exists():
        st.sidebar.warning(f"GS 디폴트 CSV가 없습니다: {p}")
        return None, None
    try:
        return read_gs_csv_raw(p, sep=sep), p.name
    except Exception as e:
        st.sidebar.error(f"GS CSV 읽기 실패: {p} ({e})")
        return None, None


# ── 헤더 ────────────────────────────────────────────────────────────────────
st.title("🧩 Modular Streamlit App")
st.caption("메인앱에서 파일 업로드 → 섹션에 컨텍스트 전달 → 섹션이 로직을 호출해 UI 렌더")

# ── 사이드바 업로드 (여러 파일 드래그&드롭) ──────────────────────────────────
with st.sidebar:
    st.header("업로드 (드래그&드롭, 여러 파일)")
    st.caption(
        f"각 드롭존에 '{RAINBOW_FILENAME}'(엑셀)와 '{GS_FILENAME}'(CSV)를 함께 올리세요.\n"
        "이름으로 자동 식별합니다."
    )
    pro_files = st.file_uploader("프로 파일 묶음 (.xlsx .csv)", type=["xlsx", "csv"],
                                 accept_multiple_files=True, key="multi_pro")
    ama_files = st.file_uploader("일반 파일 묶음 (.xlsx .csv)", type=["xlsx", "csv"],
                                 accept_multiple_files=True, key="multi_ama")

def _pick_by_name(files, rb_name: str, gs_name: str):
    """업로드된 파일들 중 무지개/GS를 파일명으로 골라 반환."""
    rb, gs = None, None
    if files:
        for f in files:
            name = f.name.strip()
            low = name.lower()
            # 무지개 파일: 정확매칭 우선, 느슨한 매칭 보조
            if low == rb_name.lower() or "first_data_transition" in low:
                rb = f
            # GS 파일: 정확매칭 우선, 느슨한 매칭 보조
            if low == gs_name.lower() or "csvexport" in low:
                gs = f
    return rb, gs

# ── 파일 선택: 업로드(멀티) > 디폴트 ────────────────────────────────────────
pro_arr = None; pro_name = None
ama_arr = None; ama_name = None
gs_pro_arr = None; gs_pro_name = None
gs_ama_arr = None; gs_ama_name = None

# 프로 묶음
if pro_files:
    rb, gs = _pick_by_name(pro_files, RAINBOW_FILENAME, GS_FILENAME)
    if rb is not None:
        pro_arr = read_xlsx_to_array(rb); pro_name = rb.name
    if gs is not None:
        gs_pro_arr = read_gs_csv_raw(gs, sep=","); gs_pro_name = gs.name
elif USE_CODE_DEFAULTS:
    pro_arr, pro_name = try_read_default(DEFAULT_PRO_PATH)
    gs_pro_arr, gs_pro_name = try_read_gs_default(DEFAULT_GS_PRO_PATH, sep=",")

# 일반 묶음
if ama_files:
    rb, gs = _pick_by_name(ama_files, RAINBOW_FILENAME, GS_FILENAME)
    if rb is not None:
        ama_arr = read_xlsx_to_array(rb); ama_name = rb.name
    if gs is not None:
        gs_ama_arr = read_gs_csv_raw(gs, sep=","); gs_ama_name = gs.name
elif USE_CODE_DEFAULTS:
    ama_arr, ama_name = try_read_default(DEFAULT_AMA_PATH)
    gs_ama_arr, gs_ama_name = try_read_gs_default(DEFAULT_GS_AMA_PATH, sep=",")

# 업로드 상태 표시
with st.sidebar:
    def _ok(x): return "✅" if x is not None else "⚠️"
    st.write(f"프로: 무지개 {_ok(pro_arr)} / GS {_ok(gs_pro_arr)}")
    st.write(f"일반: 무지개 {_ok(ama_arr)} / GS {_ok(gs_ama_arr)}")




# ── 컨텍스트 ────────────────────────────────────────────────────────────────
ctx = {
    "pro_arr": pro_arr,
    "ama_arr": ama_arr,
    "gs_pro_arr": gs_pro_arr,
    "gs_ama_arr": gs_ama_arr,
    "files": {
        "pro_name": pro_name,
        "ama_name": ama_name,
        "gs_pro_name": gs_pro_name,
        "gs_ama_name": gs_ama_name,
    },
}

# ── 섹션 검색/선택 ───────────────────────────────────────────────────────────
sections = discover_sections(SECTIONS_DIR)
if not sections:
    st.warning("섹션이 없습니다. sections/ 아래에 폴더와 main.py를 추가하세요.")
    st.stop()

sections_sorted = sorted(
    sections,
    key=lambda s: s["meta"].get("title", s["id"])   # 타이틀 기준 정렬
)

choices = {
    f"{s['meta'].get('title', s['id'])}": s
    for i, s in enumerate(sections_sorted)
}

labels = list(choices.keys())

# 쿼리파라미터 유지
qp = st.query_params
current_id = qp.get("section")
default_label = next((lbl for lbl, sec in choices.items() if sec["id"] == current_id), labels[0])

with st.sidebar:
    st.header("섹션")
    picked_label = st.selectbox("이동", options=labels, index=labels.index(default_label), key="section_select")

selected = choices[picked_label]
st.query_params["section"] = selected["id"]  # URL 동기화




# ─────────────────────────────────────────────────────────────
#  🌈 전역 dataframe 하이라이트 오버라이드 (웹 표시용)
#    - 빨강: 프로×일반 < 0 (부호 반대)
#    - 노랑: 위가 아니고, 상대차이 ≥ percent_threshold
#      상대차이 = |p-a| / max(|p|, |a|)  (0-division 방지)
# ─────────────────────────────────────────────────────────────
from pandas.io.formats.style import Styler
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as _html
import numpy as np

_orig_dataframe = st.dataframe  # 백업

def _build_sign_and_diff_styles(
    df: pd.DataFrame,
    pair_rules=(("프로", "일반"), ("Pro", "Ama"), ("pro_", "ama_"), ("Pro_", "Ama_")),
    red="#FFC7CE",
    yellow="#FFEB9C",
    percent_threshold: float = 0.30,
) -> pd.DataFrame:
    import re
    headers = list(df.columns)                 # 원본 라벨 유지
    headers_str = list(map(str, headers))      # 문자열 버전
    col_index = {h: i for i, h in enumerate(headers)}

    # ─────────────────────────────────────────────────────────
    # ① 가로형(행 기반) 표 감지: '구분' 또는 '항목' 라벨 열이 있고
    #    나머지가 프레임(숫자) 컬럼인 경우
    # ─────────────────────────────────────────────────────────
    label_col = next((c for c in ["구분", "항목"] if c in df.columns), None)
    if label_col is not None:
        frame_cols = [c for c in df.columns if c != label_col and str(c).isdigit()]
        if frame_cols:
            styles = pd.DataFrame("", index=df.index, columns=df.columns)

            # 1) (NEW) 단순 2행 비교: '프로' / '일반' 라벨만 있는 경우
            label_map = {str(v).strip(): idx for idx, v in df[label_col].items()}
            if "프로" in label_map and "일반" in label_map:
                r_pro = label_map["프로"]
                r_ama = label_map["일반"]

                p = pd.to_numeric(df.loc[r_pro, frame_cols], errors="coerce")
                a = pd.to_numeric(df.loc[r_ama, frame_cols], errors="coerce")

                red_mask = (p * a) < 0
                denom = np.maximum(np.abs(p), np.abs(a))
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.where(denom > 0, np.abs(p - a) / denom, np.nan)
                yellow_mask = (~red_mask) & (pd.Series(rel, index=frame_cols) >= percent_threshold)

                # ✅ '일반' 행만 칠하기
                for c in frame_cols:
                    if bool(red_mask.get(c, False)):
                        styles.at[r_ama, c] = f"background-color: {red}"
                    elif bool(yellow_mask.get(c, False)):
                        styles.at[r_ama, c] = f"background-color: {yellow}"

                return styles  # 가로형 처리 끝

            # 2) (기존) 측면(L/R) + 역할(프로/일반) 라벨 조합인 경우
            import re
            def _parse_side_role(s: str):
                s = ("" if s is None else str(s)).strip()
                parts = re.split(r"\s*[·\.]\s*", s.replace(" ", ""))
                if len(parts) >= 2:
                    side, role = parts[0], parts[1]
                else:
                    return None, None
                if role in ("프로", "Pro", "pro", "PRO"):
                    role = "프로"
                elif role in ("일반", "Ama", "ama", "AMA"):
                    role = "일반"
                else:
                    return None, None
                return side, role

            side_rows: dict[str, dict[str, object]] = {}
            for ridx, label in df[label_col].items():
                side, role = _parse_side_role(label)
                if side and role:
                    side_rows.setdefault(side, {})[role] = ridx

            for side, roles in side_rows.items():
                if not ("프로" in roles and "일반" in roles):
                    continue
                r_pro = roles["프로"]
                r_ama = roles["일반"]

                p = pd.to_numeric(df.loc[r_pro, frame_cols], errors="coerce")
                a = pd.to_numeric(df.loc[r_ama, frame_cols], errors="coerce")

                red_mask = (p * a) < 0
                denom = np.maximum(np.abs(p), np.abs(a))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(denom > 0, np.abs(p - a) / denom, np.nan)
                yellow_mask = (~red_mask) & (pd.Series(rel, index=frame_cols) >= percent_threshold)

                # ✅ Ama(일반) 행만 색칠
                for c in frame_cols:
                    if bool(red_mask.get(c, False)):
                        styles.at[r_ama, c] = f"background-color: {red}"
                    elif bool(yellow_mask.get(c, False)):
                        styles.at[r_ama, c] = f"background-color: {yellow}"

            return styles


    # ─────────────────────────────────────────────────────────
    # ② (기존) 세로형(열 기반) 비교: Pro↔Ama 쌍 찾기 → Ama만 칠함
    # ─────────────────────────────────────────────────────────
    pairs = []
    for h in headers_str:
        for a, b in pair_rules:
            if a in h:
                cand = h.replace(a, b)
                # headers_str 기준으로 존재 확인
                if cand in headers_str:
                    # 원본 라벨로 치환
                    orig_h = headers[headers_str.index(h)]
                    orig_cand = headers[headers_str.index(cand)]
                    pairs.append((orig_h, orig_cand))
    # 중복 제거
    seen, uniq_pairs = set(), []
    for p_col, a_col in pairs:
        key = tuple(sorted((str(p_col), str(a_col))))
        if key not in seen:
            seen.add(key)
            uniq_pairs.append((p_col, a_col))

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for p_col, a_col in uniq_pairs:
        p = pd.to_numeric(df[p_col], errors="coerce")
        a = pd.to_numeric(df[a_col], errors="coerce")

        red_mask = (p * a) < 0
        denom = np.maximum(np.abs(p), np.abs(a))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, np.abs(p - a) / denom, np.nan)
        yellow_mask = (~red_mask) & (pd.Series(rel, index=df.index) >= percent_threshold)

        # ✅ Ama 열만 칠한다 (Pro 열은 칠하지 않음)
        styles.loc[red_mask,    a_col] = f"background-color: {red}"
        styles.loc[yellow_mask, a_col] = f"background-color: {yellow}"

    return styles


def _apply_highlight_to_styler(styler: Styler, **opts) -> Styler:
    df = styler.data if hasattr(styler, "data") else None
    if isinstance(df, pd.DataFrame):
        styles = _build_sign_and_diff_styles(df, **opts)
        styler = styler.apply(lambda _df: styles, axis=None)

    # ① 모든 셀 2자리 포맷 (숫자/숫자문자열/끝이 '!'인 문자열 모두 대응)
    def _fmt2_all(x):
        # 빈값/NaN은 공백
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        # 순수 숫자
        if isinstance(x, (int, float, np.integer, np.floating)):
            return f"{float(x):.2f}"
        # 문자열 처리: '12.3!' 또는 '12.3' 등
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("!"):
                core = s[:-1]
                try:
                    v = float(core)
                    return f"{v:.2f}!"
                except Exception:
                    return x  # 숫자 파싱 실패 시 원본 유지
            # 숫자 문자열이면 2자리
            try:
                v = float(s)
                return f"{v:.2f}"
            except Exception:
                return x  # 텍스트는 그대로
        # 그 외 타입은 그대로
        return x

    styler = styler.format(_fmt2_all, na_rep="")

    # ② 인덱스 숨김
    styler = styler.hide(axis="index")

    # ③ 테이블 외형
    styler = styler.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
        {'selector': 'th, td', 'props': [('border', '1px solid #DDD'), ('padding', '6px 8px')]},
        {'selector': 'thead th', 'props': [('background', '#F7F7F7')]}
    ])
    return styler


def _render_styler(styler: Styler, height: int | None = None, scrolling: bool = True):
    # 간단한 높이 추정
    try:
        n_rows = getattr(styler, "data", None).shape[0]
    except Exception:
        n_rows = 20
    base, per = 42, 28
    h = height or min(900, base + per * (n_rows + 1))
    _html(styler.to_html(), height=h, scrolling=scrolling)

def _auto_highlight_dataframe(data=None, *args, **kwargs):
    try:
        if isinstance(data, Styler):
            styled = _apply_highlight_to_styler(data, percent_threshold=0.30)
            return _render_styler(styled)
        if isinstance(data, pd.DataFrame):
            styled = pd.io.formats.style.Styler(data)
            styled = _apply_highlight_to_styler(styled, percent_threshold=0.30)
            return _render_styler(styled)
    except Exception as e:
        st.warning(f"자동 하이라이트 적용 실패: {e}")
    return _orig_dataframe(data, *args, **kwargs)

# 섹션 실행 전에 반드시 패치!
st.dataframe = _auto_highlight_dataframe





# ── 섹션 실행 ────────────────────────────────────────────────────────────────
import inspect

run_fn = selected.get("run")
if callable(run_fn):
    sig = inspect.signature(run_fn)
    try:
        # run(ctx) 지원이면 ctx 전달, 아니면 인자 없이
        if len(sig.parameters) >= 1:
            run_fn(ctx)
        else:
            run_fn()
    except TypeError as e:
        # 정말로 인자 불일치로 실패했고, 파라미터가 0개일 때만 run() 재시도
        if len(sig.parameters) == 0:
            run_fn()
        else:
            st.error("섹션 실행 중 TypeError가 발생했습니다. 아래 상세를 확인하세요.")
            st.exception(e)
            st.stop()
    except Exception as e:
        st.error("섹션 실행 중 오류가 발생했습니다. 아래 상세를 확인하세요.")
        st.exception(e)
        st.stop()
else:
    st.error("선택한 섹션에 run 함수가 없습니다.")

# ── 상태 안내 ────────────────────────────────────────────────────────────────
with st.sidebar:
    if pro_arr is None or ama_arr is None:
        st.info("무지개(기존) 엑셀: 업로드 또는 디폴트 중 하나가 비어 있습니다.")
    else:
        st.success(f"사용 파일: 프로 `{pro_name}` · 일반 `{ama_name}`")

    if gs_pro_arr is None or gs_ama_arr is None:
        st.info("GS CSV: 업로드하거나 디폴트 경로를 설정하세요.")
    else:
        st.success(f"GS 파일: 프로 `{gs_pro_name}` · 일반 `{gs_ama_name}`")


# app.py (사이드바나 페이지 하단 아무 곳)
from datetime import datetime

st.markdown("---")
st.subheader("📦 마스터 엑셀 내보내기")
if st.button("모든 섹션을 하나의 엑셀로 다운로드"):
    sections = st.session_state.get("section_tables", {})
    if not sections:
        st.warning("먼저 각 섹션 페이지를 열어 표를 생성해 주세요.")
    else:
        used = set()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            for sec_id, payload in sections.items():
                title  = payload.get("title", sec_id)
                tables = payload.get("tables", {})
                sheet  = _safe_sheet(title, used)
                _write_section_sheet(writer, sheet_name=sheet, tables=tables)
        buf.seek(0)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📥 마스터 엑셀 받기",
            data=buf.getvalue(),
            file_name=f"master_sections_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

