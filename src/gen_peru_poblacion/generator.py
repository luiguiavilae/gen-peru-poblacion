from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from gen_peru_poblacion.calibrator import Calibrator
from gen_peru_poblacion.config import Config
from gen_peru_poblacion.exporters import export_csv, export_json, export_jsonl

DATA_SOURCES = ["INEI-ENAHO-2023", "PRODUCE-ENAMIN-2023", "SBS-2023"]

_DATA_SOURCES_BY_SEGMENT: dict[str, list[str]] = {
    "mype": DATA_SOURCES,
    "consumidores": ["INEI-ENAHO-2023", "OSIPTEL-ERESTEL-2023", "IPSOS-Peru-2023"],
    "financiero": ["SBS-IF-2023", "BCRP-Encuesta-SF-2023", "INEI-ENAHO-2023"],
    "salud": ["INEI-ENDES-2023", "MINSA-ASIS-2023", "SIS-2023", "SUSALUD-2023"],
}

_EXPORTERS = {
    "csv": export_csv,
    "json": export_json,
    "jsonl": export_jsonl,
}


class PopulationGenerator:
    """
    Orquesta el pipeline completo: calibrar → sintetizar → enriquecer → exportar.

    Uso:
        gen = PopulationGenerator(Config(segmento="mype", n=100))
        perfiles = gen.generate()
        gen.export(perfiles, formato="csv", path="./output/mype.csv")
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._calibrator = Calibrator(
            segmento=config.segmento,
            fuente_dir=config.fuente_dir,
        )
        self._fitted = False

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            self._calibrator.fit()
            self._fitted = True

    def generate(self) -> list[dict[str, Any]]:
        """
        Genera config.n perfiles sintéticos.

        Cada perfil incluye:
          - Todos los campos del segmento (region, rubro, tamaño, etc.)
          - synthetic: True
          - data_sources: lista de fuentes INEI/PRODUCE/SBS
          - perfil_id: UUID4 único
          - segmento: nombre del segmento
        """
        self._ensure_fitted()
        df = self._calibrator.sample(self.config.n)

        if self.config.region is not None:
            df = self._filter_or_resample_region(df, self.config.region)

        perfiles = df.to_dict(orient="records")
        sources = _DATA_SOURCES_BY_SEGMENT.get(self.config.segmento, DATA_SOURCES)
        return [self._enrich(p, self.config.segmento, sources) for p in perfiles]

    def export(
        self,
        perfiles: list[dict[str, Any]],
        formato: str | None = None,
        path: str | Path | None = None,
    ) -> Path:
        """
        Exporta perfiles al formato indicado.

        Si formato es None, usa config.formato.
        Si path es None, escribe en config.output_path con nombre automático.
        Devuelve el Path del archivo creado.
        """
        fmt = formato or self.config.formato
        if fmt not in _EXPORTERS:
            raise ValueError(f"formato '{fmt}' no válido. Opciones: {list(_EXPORTERS)}")

        if path is None:
            dest = self.config.output_path / f"{self.config.segmento}.{fmt}"
        else:
            dest = Path(path)

        return _EXPORTERS[fmt](perfiles, dest)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _enrich(perfil: dict[str, Any], segmento: str, sources: list[str]) -> dict[str, Any]:
        """Agrega campos obligatorios a cada perfil generado por SDV."""
        return {
            "perfil_id": str(uuid.uuid4()),
            "synthetic": True,
            "data_sources": sources,
            "segmento": segmento,
            **perfil,
        }

    def _filter_or_resample_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """
        Si se especificó una región, filtra y rellena hasta alcanzar config.n filas.
        Evita un loop infinito: si SDV no produce suficientes filas de esa región
        tras 10 intentos de oversample, usa las que haya con advertencia.
        """
        target = self.config.n
        filtered = df[df["region"] == region].copy()

        attempts = 0
        while len(filtered) < target and attempts < 10:
            extra = self._calibrator.sample(target * 3)
            filtered = pd.concat(
                [filtered, extra[extra["region"] == region]], ignore_index=True
            )
            attempts += 1

        if len(filtered) < target:
            import warnings
            warnings.warn(
                f"Solo se obtuvieron {len(filtered)} perfiles para región '{region}' "
                f"(pedido: {target}). Devolviendo los disponibles.",
                RuntimeWarning,
                stacklevel=3,
            )
            return filtered.reset_index(drop=True)

        return filtered.iloc[:target].reset_index(drop=True)
