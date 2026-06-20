"""
verify.py — Verificación KS entre perfiles generados y distribuciones fuente.

Uso como módulo:
    from gen_peru_poblacion.verify import run_ks_check
    report = run_ks_check(data_dir="./output", fuente_dir="data/fuentes")

Uso como script independiente:
    python -m gen_peru_poblacion.verify --data-dir ./output --fuente-dir data/fuentes
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Variables principales evaluadas en el KS check
VARIABLES_PRINCIPALES = ["region", "rubro", "tamaño", "adopcion_digital", "nivel_educativo"]

# Mapeos ordinales para variables categóricas evaluadas con KS continuo
_ADOPCION_ORD = {"nula": 0, "baja": 1, "media": 2, "alta": 3}
_TAMAÑO_ORD = {"unipersonal": 0, "familiar_2_a_4": 1, "pequena_5_a_10": 2, "mas_de_10": 3}
_EDUCACION_ORD = {
    "sin_nivel": 0, "primaria_incompleta": 1, "primaria_completa": 2,
    "secundaria_incompleta": 3, "secundaria_completa": 4,
    "tecnica_incompleta": 5, "tecnica_completa": 6,
    "universitaria_incompleta": 7, "universitaria_completa": 8,
}
_REGION_ORD = {
    "selva": 0, "sierra_norte": 1, "sierra_sur": 2, "sierra_centro": 3,
    "costa_sur": 4, "costa_norte": 5, "lima_metropolitana": 6,
}
_RUBRO_ORD = {
    "agricultura_familiar": 0, "manufactura_artesanal": 1, "construccion": 2,
    "restaurantes_y_food": 3, "transporte": 4, "servicios_personales": 5,
    "comercio_minorista": 6, "otro": 7,
}

_ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "adopcion_digital": _ADOPCION_ORD,
    "tamaño": _TAMAÑO_ORD,
    "nivel_educativo": _EDUCACION_ORD,
    "region": _REGION_ORD,
    "rubro": _RUBRO_ORD,
}


@dataclass
class VariableScore:
    variable: str
    ks_statistic: float
    similarity: float  # 1 - ks_statistic
    passed: bool       # similarity >= threshold


@dataclass
class KSReport:
    scores: list[VariableScore] = field(default_factory=list)
    global_score: float = 0.0
    passed: bool = False
    threshold: float = 0.70
    n_generated: int = 0
    n_fuente: int = 0

    def __str__(self) -> str:
        lines = [
            f"KS Report — {self.n_generated} perfiles generados",
            f"Umbral: similarity ≥ {self.threshold:.2f}",
            f"Score global: {self.global_score:.3f}  {'✓ PASS' if self.passed else '✗ FAIL'}",
            "",
            f"{'Variable':<25} {'KS stat':>10} {'Similarity':>12} {'Estado':>8}",
            "-" * 60,
        ]
        for s in self.scores:
            estado = "PASS" if s.passed else "FAIL"
            lines.append(
                f"{s.variable:<25} {s.ks_statistic:>10.4f} {s.similarity:>12.4f} {estado:>8}"
            )
        return "\n".join(lines)


def _load_generated(data_dir: str | Path) -> pd.DataFrame:
    """Carga perfiles desde CSV o JSONL en data_dir."""
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        frames.append(pd.read_csv(csv_file))

    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records:
            frames.append(pd.DataFrame(records))

    for json_file in sorted(data_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            frames.append(pd.DataFrame(data))

    if not frames:
        raise FileNotFoundError(f"No se encontraron archivos CSV/JSONL/JSON en: {data_dir}")

    return pd.concat(frames, ignore_index=True)


def _fuente_distribution(fuente: dict, variable: str) -> np.ndarray:
    """
    Convierte la distribución fuente de una variable a un array de valores numéricos
    (expandido por proporciones) para poder compararlo con KS de dos muestras.

    Soporta dos formatos del JSON fuente:
      - Con clave "valores": {"valores": {"cat_a": 0.3, "cat_b": 0.7}}
      - Sin "valores": {"cat_a": 0.3, "cat_b": 0.7, "_descripcion": ...}
    """
    n_expand = 10_000
    ord_map = _ORDINAL_MAPS.get(variable, {})

    if variable not in fuente:
        raise KeyError(f"Variable '{variable}' no encontrada en la fuente.")

    bloque = fuente[variable]

    # Formato con clave explícita "valores"
    if "valores" in bloque:
        valores = bloque["valores"]
    else:
        # Formato directo: filtrar claves de metadatos (empiezan con "_")
        valores = {k: v for k, v in bloque.items() if not k.startswith("_") and isinstance(v, float)}

    if not valores:
        raise KeyError(f"No se encontraron probabilidades para '{variable}'.")

    arr = []
    for cat, prob in valores.items():
        if cat in ord_map:
            count = max(1, int(round(prob * n_expand)))
            arr.extend([ord_map[cat]] * count)
    return np.array(arr, dtype=float)


def _to_numeric(series: pd.Series, variable: str) -> np.ndarray:
    """Convierte una columna categórica del DataFrame generado a numérico ordinal."""
    ord_map = _ORDINAL_MAPS.get(variable)
    if ord_map is None:
        raise KeyError(f"Sin mapa ordinal para '{variable}'.")
    return series.map(ord_map).dropna().to_numpy(dtype=float)


def run_ks_check(
    data_dir: str | Path,
    fuente_dir: str | Path = "data/fuentes",
    segmento: str = "mype",
    threshold: float = 0.70,
    variables: list[str] | None = None,
) -> KSReport:
    """
    Calcula KS similarity entre los perfiles generados en data_dir
    y las distribuciones fuente de fuente_dir.

    Parámetros
    ----------
    data_dir   : directorio con CSV/JSONL/JSON de perfiles generados
    fuente_dir : directorio con JSONs de distribuciones fuente
    segmento   : nombre del segmento (determina qué JSON cargar)
    threshold  : umbral de similarity para PASS/FAIL (default 0.70)
    variables  : lista de variables a evaluar; None = VARIABLES_PRINCIPALES

    Devuelve
    --------
    KSReport con score por variable y score global.
    """
    fuente_path = Path(fuente_dir) / f"{segmento}_distribucion_2023.json"
    with open(fuente_path, encoding="utf-8") as f:
        fuente = json.load(f)

    df_gen = _load_generated(data_dir)
    vars_to_check = variables or VARIABLES_PRINCIPALES

    scores: list[VariableScore] = []
    for var in vars_to_check:
        if var not in df_gen.columns:
            continue
        try:
            ref = _fuente_distribution(fuente, var)
            gen_numeric = _to_numeric(df_gen[var], var)
            ks_stat, _ = ks_2samp(gen_numeric, ref)
            similarity = 1.0 - ks_stat
            scores.append(VariableScore(
                variable=var,
                ks_statistic=round(ks_stat, 6),
                similarity=round(similarity, 6),
                passed=similarity >= threshold,
            ))
        except (KeyError, ValueError):
            continue

    global_score = float(np.mean([s.similarity for s in scores])) if scores else 0.0
    return KSReport(
        scores=scores,
        global_score=round(global_score, 6),
        passed=global_score >= threshold,
        threshold=threshold,
        n_generated=len(df_gen),
        n_fuente=10_000,
    )


# ── Punto de entrada como script independiente ───────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verifica KS similarity entre perfiles generados y distribuciones fuente INEI/PRODUCE/SBS."
    )
    parser.add_argument("--data-dir", required=True, help="Directorio con perfiles generados (CSV/JSONL/JSON)")
    parser.add_argument("--fuente-dir", default="data/fuentes", help="Directorio con JSONs fuente (default: data/fuentes)")
    parser.add_argument("--segmento", default="mype", help="Segmento a verificar (default: mype)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Umbral de similarity (default: 0.70)")
    parser.add_argument("--verbose", action="store_true", help="Mostrar detalle por variable")
    args = parser.parse_args()

    report = run_ks_check(
        data_dir=args.data_dir,
        fuente_dir=args.fuente_dir,
        segmento=args.segmento,
        threshold=args.threshold,
    )
    print(report)
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    _main()
