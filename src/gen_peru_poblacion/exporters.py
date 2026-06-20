from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_csv(perfiles: list[dict[str, Any]], path: str | Path) -> Path:
    """Exporta perfiles a CSV. Devuelve el Path del archivo creado."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perfiles).to_csv(dest, index=False, encoding="utf-8")
    return dest


def export_json(perfiles: list[dict[str, Any]], path: str | Path) -> Path:
    """Exporta perfiles a JSON array. Devuelve el Path del archivo creado."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(perfiles, f, ensure_ascii=False, indent=2, default=str)
    return dest


def export_jsonl(perfiles: list[dict[str, Any]], path: str | Path) -> Path:
    """Exporta perfiles a JSONL (una línea por perfil). Devuelve el Path del archivo creado."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for perfil in perfiles:
            f.write(json.dumps(perfil, ensure_ascii=False, default=str) + "\n")
    return dest
