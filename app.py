# app.py
from pathlib import Path
import streamlit as st
import pandas as pd
from core.loader import discover_sections
import io

st.set_page_config(page_title="Modular Streamlit App", layout="wide")

# ── 프로젝트 경로 ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SECTIONS_DIR = BASE_DIR / "sections"

# ── 코드 내 디폴트 파일 경로(여기만 바꿔주세요) ──────────────────────────────
USE_CODE_DEFAULTS = True  # 업로드 없을 때 코드 디폴트 사용 여부
DEFAULT_PRO_PATH = "/Users/park_sh/Desktop/sim_pro/레퍼/test/rory.xlsx"
DEFAULT_AMA_PATH = "/Users/park_sh/Desktop/sim_pro/레퍼/test/hong.xlsx"
# 절대경로를 쓰고 싶으면: Path("/Users/park_sh/Desktop/sim_pro/data/pro.xlsx")

# ── 캐시된 로더 ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_xlsx_to_array(file_or_path):
    name = getattr(file_or_path, "name", str(file_or_path))
    suffix = Path(name).suffix.lower()

    fobj = file_or_path
    if hasattr(file_or_path, "getvalue"):  # UploadedFile
        fobj = io.BytesIO(file_or_path.getvalue())

    try:
        if suffix in (".xlsx", ".xlsm", ".xltx", ".xltm"):
            try:
                import openpyxl  # ensure installed
            except ImportError:
                st.error("`.xlsx`를 읽으려면 `openpyxl`이 필요합니다. requirements.txt에 `openpyxl>=3.1.5`를 추가하세요.")
                return None
            df = pd.read_excel(fobj, header=None, engine="openpyxl")
        elif suffix == ".csv":
            df = pd.read_csv(fobj, header=None, encoding_errors="ignore")
        elif suffix == ".xls":
            try:
                import xlrd
            except ImportError:
                st.error("`.xls`를 읽으려면 `xlrd<2.0`이 필요합니다. requirements.txt에 추가하세요.")
                return None
            df = pd.read_excel(fobj, header=None, engine="xlrd")
        elif suffix == ".xlsb":
            try:
                import pyxlsb
            except ImportError:
                st.error("`.xlsb`를 읽으려면 `pyxlsb`가 필요합니다. requirements.txt에 추가하세요.")
                return None
            df = pd.read_excel(fobj, header=None, engine="pyxlsb")
        else:
            st.error(f"지원하지 않는 파일 형식입니다: {suffix}")
            return None

        return df.values
    except Exception as e:
        st.exception(e)
        return None
        
def try_read_default(p: Path | None):
    if not p:
        return None, None
    p = Path(p).expanduser()
    if p.exists():
        try:
            return read_xlsx_to_array(p), p.name
        except Exception as e:
            st.sidebar.error(f"디폴트 파일 읽기 실패: {p} ({e})")
            return None, None
    else:
        st.sidebar.warning(f"디폴트 파일이 없습니다: {p}")
        return None, None

st.title("🧩 Modular Streamlit App")
st.caption("메인앱에서 파일 업로드 → 섹션에 컨텍스트 전달 → 섹션이 로직을 호출해 UI 렌더")

# ── 사이드바: 업로드 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("업로드")
    pro_file = st.file_uploader("프로 엑셀(.xlsx)", type=["xlsx"], key="pro_file")
    ama_file = st.file_uploader("일반 엑셀(.xlsx)", type=["xlsx"], key="ama_file")

# ── 파일 선택 로직: 업로드 > 코드 디폴트 ─────────────────────────────────────
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

ctx = {
    "pro_arr": pro_arr,
    "ama_arr": ama_arr,
    "files": {"pro_name": pro_name, "ama_name": ama_name},
}

# ── 섹션 검색/선택 ────────────────────────────────────────────────────────────
sections = discover_sections(SECTIONS_DIR)
if not sections:
    st.warning("섹션이 없습니다. sections/ 아래에 폴더와 main.py를 추가하세요.")
    st.stop()

sections_sorted = sorted(
    sections, key=lambda s: (s["meta"].get("order", 1000), s["meta"].get("title", s["id"]))
)

# 라벨 ↔ 섹션 매핑 (인덱스 None 문제 회피)
choices = {
    f"{s['meta'].get('icon','📁')} {s['meta'].get('title', s['id'])}": s
    for s in sections_sorted
}
labels = list(choices.keys())

# 쿼리파라미터로 섹션 유지
qp = st.query_params
current_id = qp.get("section")
default_label = next((lbl for lbl, sec in choices.items() if sec["id"] == current_id), labels[0])

with st.sidebar:
    st.header("섹션")
    picked_label = st.selectbox("이동", options=labels, index=labels.index(default_label), key="section_select")

selected = choices[picked_label]
st.query_params["section"] = selected["id"]  # URL 동기화

# 섹션 실행 (run(ctx) / run() 모두 지원)
run_fn = selected.get("run")
if callable(run_fn):
    try:
        run_fn(ctx)
    except TypeError:
        run_fn()
else:
    st.error("선택한 섹션에 run 함수가 없습니다.")

# ── 상태 안내 ────────────────────────────────────────────────────────────────
with st.sidebar:
    if pro_arr is None or ama_arr is None:
        st.info("업로드 또는 코드 디폴트 중 하나가 비어 있습니다.")
    else:
        st.success(f"사용 파일: 프로 `{pro_name}` · 일반 `{ama_name}`")
