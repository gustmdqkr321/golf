# sections/sim_pro/features/vectorize.py
from __future__ import annotations
import io, math, json
import numpy as np
import pandas as pd

# 이미 만든 피처들 불러오기 (단일표가 반드시 build_single_table)
from . import _1body   as f1
from . import _2t      as f2   # BN1-AY1
from . import _3t      as f3   # |CA1-CM1|
from . import _4t      as f4   # A4 vs (AC4,AD4,AE4)
from . import _5t      as f5   # A(AX1..)->B(AX4..)
from . import _6t      as f6   # A4 vs B 평균
from . import _7t      as f7   # AR/BG 거리 1·4·Δ
from . import _8t      as f8   # CN/CO/CP 차
from . import _9t      as f9   # B10, B4/(B7-B4)

# --------- 벡터 스키마 정의 ---------
# 각 피처의 "단일표"에서 어떤 행들을 벡터에 쓸지 인덱스로 정의 (None이면 전체 사용)
# 인덱스는 0-based
VECTOR_SCHEMA = [
    ("body",  f1.build_single_table,  None),                 # 11개 전체
    ("bn_ay", f2.build_single_table,  [-1]),                 # 마지막(BN1-AY1)만
    ("abs_ca_cm", f3.build_single_table, [-1]),              # 마지막(|CA1-CM1|)
    ("ab_row4", f4.build_single_table, None),                # 5개: X,Y,각도,Z,거리
    ("ab_1_to_4", f5.build_single_table, None),              # 5개
    ("ab4_midavg", f6.build_single_table, None),             # 5개
    ("ar_bg", f7.build_single_table, None),                  # 3개: 1,4,Δ
    ("cnco_cp", f8.build_single_table, None),                # 3개
    ("b_metrics", f9.build_single_table, None),              # 2개
]


# sections/sim_pro/features/vectorize.py 내 맨 아래에 추가
from pathlib import Path
import pandas as pd
import numpy as np

def _load_sheet_to_array(path: Path) -> np.ndarray:
    """엑셀/CSV → 넘파이 2D 배열. 숫자 이외는 NaN으로 강제."""
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, header=None, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None)
    else:
        raise ValueError(f"지원하지 않는 포맷: {path.suffix}")
    return df.apply(pd.to_numeric, errors="coerce").values

def build_db_from_fs(root_dir: str,
                     pattern: str = "**/first_data_transi*.xlsx") -> dict:
    """
    예) root_dir='driver'
    - driver/BOMEE LEE/first_data_transi_1.xlsx
    - driver/Brook Pancake/first_data_transi_A.xlsx
    처럼 저장된 모든 파일을 스윙 '하나'로 보고 벡터화.
    DB 레코드 이름은 '프로명/파일명' 형태.
    """
    root = Path(root_dir)
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"'{root_dir}' 아래에 '{pattern}' 파일을 못 찾았어요.")
    items: list[tuple[str, np.ndarray]] = []
    for p in paths:
        arr = _load_sheet_to_array(p)
        name = f"{p.parent.name}/{p.stem}"  # 예: 'BOMEE LEE/first_data_transi_1'
        items.append((name, arr))
    db = fit_db(items)  # 표준화(Z-score) 포함
    return db

def export_db_from_fs(root_dir: str, out_path: str,
                      pattern: str = "**/first_data_transi*.xlsx") -> dict:
    """파일트리 스캔→DB 생성→.npz 저장하고 DB를 반환."""
    db = build_db_from_fs(root_dir, pattern=pattern)
    save_db(db, out_path)
    return db

def topk_per_pro(result_df: pd.DataFrame, per_pro: int = 1) -> pd.DataFrame:
    """
    search_similar() 결과에서 프로당 상위 N개만 남기기.
    이름이 '프로/파일' 형식이라고 가정.
    """
    tmp = result_df.copy()
    tmp["프로_이름"] = tmp["프로"].astype(str).str.split("/").str[0]
    return (tmp.sort_values("cosine_sim", ascending=False)
               .groupby("프로_이름", as_index=False)
               .head(per_pro)
               .reset_index(drop=True))


def _flatten_values(df: pd.DataFrame, keep_idx: list[int] | None) -> tuple[list[str], list[float]]:
    vals = pd.to_numeric(df["값"], errors="coerce").tolist()
    names = df["항목"].astype(str).tolist()
    if keep_idx is not None:
        names = [names[i] for i in keep_idx]
        vals  = [vals[i]  for i in keep_idx]
    # NaN → 0.0 (안정적 비교용)
    vals = [0.0 if (v is None or (isinstance(v, float) and not math.isfinite(v))) else float(v) for v in vals]
    return names, vals

def compute_feature_vector(arr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """단일 스윙 배열 → 고정 길이 벡터(40차원)와 feature 이름 목록"""
    feat_names: list[str] = []
    feat_vals:  list[float] = []
    for prefix, builder, keep in VECTOR_SCHEMA:
        df = builder(arr)
        names, vals = _flatten_values(df, keep)
        # 이름에 prefix 붙여 충돌 방지
        names = [f"{prefix}:{n}" for n in names]
        feat_names.extend(names)
        feat_vals.extend(vals)
    v = np.asarray(feat_vals, dtype=float)
    return v, feat_names

# --------- DB 빌드/저장/로드 ---------
def fit_db(pro_items: list[tuple[str, np.ndarray]]) -> dict:
    """
    pro_items: [(이름, 배열), ...]
    반환: {'names','Z','mu','sigma','feature_names'}
      - Z: 표준화된 벡터 행렬 (n_pro x d)
      - mu,sigma: 표준화 파라미터 (d,)
    """
    X_list, names = [], []
    feat_names_ref: list[str] | None = None
    for name, arr in pro_items:
        v, feat_names = compute_feature_vector(arr)
        if feat_names_ref is None:
            feat_names_ref = feat_names
        else:
            # 스키마 불일치 방지
            assert feat_names == feat_names_ref, "Feature schema mismatch"
        X_list.append(v)
        names.append(name)
    X = np.vstack(X_list)  # (n, d)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # 0으로 나눔 방지
    Z = (X - mu) / sigma
    return {"names": np.array(names, dtype=object),
            "Z": Z, "mu": mu, "sigma": sigma,
            "feature_names": np.array(feat_names_ref, dtype=object)}

def save_db(db: dict, path: str) -> None:
    np.savez(path,
             names=db["names"], Z=db["Z"],
             mu=db["mu"], sigma=db["sigma"],
             feature_names=db["feature_names"])

def load_db(file_bytes_or_path) -> dict:
    """file-like(bytes) 또는 경로 문자열 모두 지원"""
    if isinstance(file_bytes_or_path, (bytes, bytearray, io.BytesIO)):
        npz = np.load(io.BytesIO(file_bytes_or_path if isinstance(file_bytes_or_path, (bytes, bytearray)) else file_bytes_or_path.read()), allow_pickle=True)
    else:
        npz = np.load(file_bytes_or_path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}

# --------- 검색 ---------
def cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    B = B.astype(float)
    a_norm = np.linalg.norm(a) or 1.0
    B_norm = np.linalg.norm(B, axis=1)
    B_norm[B_norm == 0] = 1.0
    return (B @ a) / (B_norm * a_norm)

def search_similar(user_arr: np.ndarray, db: dict, top_k: int = 5) -> pd.DataFrame:
    v, feat_names = compute_feature_vector(user_arr)
    # 표준화(프로 DB의 파라미터 사용)
    z = (v - db["mu"]) / db["sigma"]
    sims = cosine_sim(z, db["Z"])
    order = np.argsort(-sims)[:top_k]
    rows = []
    for idx in order:
        rows.append([int(idx), str(db["names"][idx]), float(sims[idx])])
    return pd.DataFrame(rows, columns=["rank_idx", "프로", "cosine_sim"]).sort_values("cosine_sim", ascending=False).reset_index(drop=True)
