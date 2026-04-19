from __future__ import annotations

import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_DIRS = [
    REPO_ROOT / "hfl_v7-RN",
    REPO_ROOT / "hfl_v7-no-ascon",
    REPO_ROOT / "ml_classic",
    REPO_ROOT / "Analisis de Modelos" / "RN",
]

SKIP_PARTS = {
    "__pycache__",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def main() -> None:
    checked = 0
    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            continue
        for path in sorted(target_dir.rglob("*.py")):
            if should_skip(path):
                continue
            py_compile.compile(str(path), doraise=True)
            checked += 1
            print(f"[OK] {path.relative_to(REPO_ROOT)}")

    if checked == 0:
        raise SystemExit("No se encontraron archivos Python para validar.")

    print(f"Validacion Python completada. Archivos compilados: {checked}")


if __name__ == "__main__":
    main()
