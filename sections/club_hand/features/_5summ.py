from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# A1 → 값 유틸 (무지개 ndarray 전용)
# ─────────────────────────────────────────────
_CELL = re.compile(r"^([A-Za-z]+)(\d+)$")

def _col_idx(letters: str) -> int:
    idx = 0
    for ch in letters.upper():
        idx = idx*26 + (ord(ch) - ord("A") + 1)
    return idx - 1

def g(arr: np.ndarray, addr: str) -> float:
    m = _CELL.match(addr.strip())
    if not m:
        return float("nan")
    c = _col_idx(m.group(1))
    r = int(m.group(2)) - 1
    try:
        return float(arr[r, c])
    except Exception:
        return float("nan")

# ─────────────────────────────────────────────
# 회전중심(좌우 평균) 구간 평균 - ADD 기준
# 구간: 1-4( idx 0..2 ), 4-7( 3..5 ), 7-10( 6..8 )
# ─────────────────────────────────────────────
_SEG_RANGES = {
    "1-4":  (0, 3),  # [0,1,2]
    "4-7":  (3, 6),  # [3,4,5]
    "7-10": (6, 9),  # [6,7,8]
}

def _center_series(arr: np.ndarray,
                   Lx: str, Ly: str, Lz: str,
                   Rx: str, Ry: str, Rz: str) -> np.ndarray:
    """10프레임(1..10)의 좌우 평균 중심 좌표 (10,3)"""
    out = np.zeros((10, 3), dtype=float)
    for i in range(10):
        n = i+1
        L = np.array([g(arr, f"{Lx}{n}"), g(arr, f"{Ly}{n}"), g(arr, f"{Lz}{n}")], dtype=float)
        R = np.array([g(arr, f"{Rx}{n}"), g(arr, f"{Ry}{n}"), g(arr, f"{Rz}{n}")], dtype=float)
        out[i] = (L + R) / 2.0
    return out

def _rotation_center_rows(arr: np.ndarray,
                          Lx: str, Ly: str, Lz: str,
                          Rx: str, Ry: str, Rz: str) -> list[tuple[str, float, float, float]]:
    """
    각 구간의 (구간평균 중심 - ADD 중심) [X,Y,Z] 반환
    """
    C = _center_series(arr, Lx, Ly, Lz, Rx, Ry, Rz)  # (10,3)
    base = C[0]  # ADD
    rows: list[tuple[str, float, float, float]] = []
    for label, (s, e) in _SEG_RANGES.items():
        seg_mean = C[s:e].mean(axis=0)  # e 미포함 → s,s+1,s+2
        diff = np.round(seg_mean - base, 3)
        rows.append((label, float(diff[0]), float(diff[1]), float(diff[2])))
    return rows

def _build_diff_rows(name: str,
                     pro_arr: np.ndarray, ama_arr: np.ndarray,
                     Lx: str, Ly: str, Lz: str, Rx: str, Ry: str, Rz: str) -> list[dict]:
    """
    한 신체부위(name)에 대해 Ama−Pro 차이를 구간별로 생성
    """
    pro_rows = _rotation_center_rows(pro_arr, Lx, Ly, Lz, Rx, Ry, Rz)
    ama_rows = _rotation_center_rows(ama_arr, Lx, Ly, Lz, Rx, Ry, Rz)

    out: list[dict] = []
    for (g, px, py, pz), (_, ax, ay, az) in zip(pro_rows, ama_rows):
        out.append({
            "부위": name,
            "구간": g,
            "X 차이 (Ama - Pro)": round(ax - px, 2),
            "Y 차이 (Ama - Pro)": round(ay - py, 2),
            "Z 차이 (Ama - Pro)": round(az - pz, 2),
        })
    return out

# ─────────────────────────────────────────────
# 공개 API: 골반/어깨/무릎 한 방에 합치기
# ─────────────────────────────────────────────
def build_rotation_center_diff_all(base_pro: np.ndarray, base_ama: np.ndarray) -> pd.DataFrame:
    """
    골반(HIJ/KLM), 어깨(ALM/ABB), 무릎(BPQ/CCD)의
    회전중심 구간 차이(Ama−Pro)를 모아 반환.
    행: 각 부위 × (1-4, 4-7, 7-10)
    """
    rows: list[dict] = []
    # 골반: 왼(H,I,J) / 오른(K,L,M)
    rows += _build_diff_rows("골반", base_pro, base_ama, "H","I","J", "K","L","M")
    # 어깨: 왼(AL,AM,AN) / 오른(BA,BB,BC)
    rows += _build_diff_rows("어깨", base_pro, base_ama, "AL","AM","AN", "BA","BB","BC")
    # 무릎: 왼(BP,BQ,BR) / 오른(CB,CC,CD)
    rows += _build_diff_rows("무릎", base_pro, base_ama, "BP","BQ","BR", "CB","CC","CD")

    df = pd.DataFrame(rows, columns=[
        "부위","구간",
        "X 차이 (Ama - Pro)","Y 차이 (Ama - Pro)","Z 차이 (Ama - Pro)"
    ])
    return df
