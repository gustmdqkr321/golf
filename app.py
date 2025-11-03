# app.py
from pathlib import Path
import io
import pandas as pd
import streamlit as st
from core.loader import discover_sections

# app.py (상단 임포트 밑)
import io, re
import pandas as pd


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

                # 빨강: 부호 반대
                formula_red = f'=${p_col}{excel_r}*${a_col}{excel_r}<0'
                ws.conditional_format(r, p_idx, r, p_idx, {
                    'type': 'formula', 'criteria': formula_red, 'format': red_fill
                })
                ws.conditional_format(r, a_idx, r, a_idx, {
                    'type': 'formula', 'criteria': formula_red, 'format': red_fill
                })

                # 노랑: 부호 같고, 상대차이 ≥ 임계치
                # =AND($p*$a>=0, IF(MAX(ABS($p),ABS($a))=0, FALSE, ABS($p-$a)/MAX(ABS($p),ABS($a))>=0.3))
                formula_yellow = (
                    f'=AND('
                    f'${p_col}{excel_r}*${a_col}{excel_r}>=0,'
                    f'IF(MAX(ABS(${p_col}{excel_r}),ABS(${a_col}{excel_r}))=0,'
                    f'FALSE,'
                    f'ABS(${p_col}{excel_r}-${a_col}{excel_r})/MAX(ABS(${p_col}{excel_r}),ABS(${a_col}{excel_r}))>={DIFF_THRESH}'
                    f'))'
                )
                ws.conditional_format(r, p_idx, r, p_idx, {
                    'type': 'formula', 'criteria': formula_yellow, 'format': yellow_fill
                })
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
    pair_rules=(("프로", "일반"), ("Pro", "Ama"),("pro_", "ama_"),
        ("Pro_", "Ama_"),),
    red="#FFC7CE",
    yellow="#FFEB9C",
    percent_threshold: float = 0.30,
) -> pd.DataFrame:
    headers = list(map(str, df.columns))
    col_index = {h: i for i, h in enumerate(headers)}

    # 후보쌍(프로↔일반 / Pro↔Ama) 자동 탐지
    pairs = []
    for h in headers:
        for a, b in pair_rules:
            if a in h:
                cand = h.replace(a, b)
                if cand in col_index:
                    pairs.append((h, cand))
    # 중복 제거
    seen, uniq_pairs = set(), []
    for p, a in pairs:
        key = tuple(sorted((p, a)))
        if key not in seen:
            seen.add(key)
            uniq_pairs.append((p, a))

    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for p_col, a_col in uniq_pairs:
        p = pd.to_numeric(df[p_col], errors="coerce")
        a = pd.to_numeric(df[a_col], errors="coerce")

        # 1) 빨강: 부호 반대
        red_mask = (p * a) < 0

        # 2) 노랑: 빨강이 아닌 것 중 상대차이 ≥ 임계치
        denom = np.maximum(np.abs(p), np.abs(a))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(denom > 0, np.abs(p - a) / denom, np.nan)
        yellow_mask = (~red_mask) & (pd.Series(rel, index=df.index) >= percent_threshold)

        styles.loc[red_mask,   p_col] = f"background-color: {red}"
        styles.loc[red_mask,   a_col] = f"background-color: {red}"
        styles.loc[yellow_mask, p_col] = f"background-color: {yellow}"
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

