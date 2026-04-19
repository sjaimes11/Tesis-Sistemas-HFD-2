from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOTS = [
    REPO_ROOT / "Analisis de Modelos" / "RN",
]


def main() -> None:
    checked = 0
    for root in NOTEBOOK_ROOTS:
        if not root.exists():
            continue
        for notebook_path in sorted(root.rglob("*.ipynb")):
            with notebook_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            checked += 1
            print(f"[OK] {notebook_path.relative_to(REPO_ROOT)}")

    if checked == 0:
        raise SystemExit("No se encontraron notebooks para validar.")

    print(f"Validacion de notebooks completada. Notebooks revisados: {checked}")


if __name__ == "__main__":
    main()
