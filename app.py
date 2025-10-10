# app.py
from pathlib import Path
import io
import pandas as pd
import streamlit as st
from core.loader import discover_sections

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

# ── 사이드바 업로드 ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("업로드")
    pro_file = st.file_uploader("프로 엑셀(.xlsx)", type=["xlsx"], key="pro_file")
    ama_file = st.file_uploader("일반 엑셀(.xlsx)", type=["xlsx"], key="ama_file")
    st.divider()
    gs_pro_file = st.file_uploader("프로 GS(.csv)", type=["csv"], key="gs_pro_file")
    gs_ama_file = st.file_uploader("일반 GS(.csv)", type=["csv"], key="gs_ama_file")

# ── 파일 선택: 업로드 > 디폴트 ──────────────────────────────────────────────
if pro_file:
    pro_arr = read_xlsx_to_array(pro_file)
    pro_name = pro_file.name
elif USE_CODE_DEFAULTS:
    pro_arr, pro_name = try_read_default(DEFAULT_PRO_PATH)
else:
    pro_arr, pro_name = None, None

if ama_file:
    ama_arr = read_xlsx_to_array(ama_file)
    ama_name = ama_file.name
elif USE_CODE_DEFAULTS:
    ama_arr, ama_name = try_read_default(DEFAULT_AMA_PATH)
else:
    ama_arr, ama_name = None, None

# GS (csv) — DataFrame으로, header=None
if gs_pro_file:
    gs_pro_arr = read_gs_csv_raw(gs_pro_file, sep=",")   # 필요하면 sep=";"로
    gs_pro_name = gs_pro_file.name
elif USE_CODE_DEFAULTS:
    gs_pro_arr, gs_pro_name = try_read_gs_default(DEFAULT_GS_PRO_PATH, sep=",")
else:
    gs_pro_arr, gs_pro_name = None, None

if gs_ama_file:
    gs_ama_arr = read_gs_csv_raw(gs_ama_file, sep=",")
    gs_ama_name = gs_ama_file.name
elif USE_CODE_DEFAULTS:
    gs_ama_arr, gs_ama_name = try_read_gs_default(DEFAULT_GS_AMA_PATH, sep=",")
else:
    gs_ama_arr, gs_ama_name = None, None



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