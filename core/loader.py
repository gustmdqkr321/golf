# core/loader.py
from pathlib import Path
import importlib.util
from typing import List, Dict, Any

def _import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def discover_sections(sections_dir: Path) -> List[Dict[str, Any]]:
    """
    sections_dir 하위의 각 폴더에서 main.py를 찾아 로드한다.
    각 main.py는 아래 2개를 반드시 제공해야 한다:
      - get_metadata() -> dict (id, title, icon, order 등)
      - run() -> None
    """
    results = []
    if not sections_dir.exists():
        return results

    for sub in sorted(sections_dir.iterdir()):
        if not sub.is_dir():
            continue
        main_py = sub / "main.py"
        if not main_py.exists():
            continue
        module_name = f"sections.{sub.name}.main"
        try:
            mod = _import_from_path(module_name, main_py)
            if not hasattr(mod, "get_metadata") or not hasattr(mod, "run"):
                # main.py 계약 위반 시 스킵
                continue
            meta = mod.get_metadata()
            _id = meta.get("id", sub.name)
            results.append({
                "id": _id,
                "path": sub,
                "meta": meta,
                "run": getattr(mod, "run"),
                "module": mod,
            })
        except Exception as e:
            # 로딩 실패 시 그냥 건너뛴다(필요시 로그 출력)
            print(f"[WARN] Failed to load section {sub.name}: {e}")
            continue

    return results
