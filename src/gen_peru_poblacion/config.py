from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SEGMENTOS_VALIDOS = {"mype", "consumidores", "financiero", "salud"}
REGIONES_VALIDAS = {
    "lima_metropolitana",
    "costa_norte",
    "costa_sur",
    "sierra_norte",
    "sierra_centro",
    "sierra_sur",
    "selva",
    "callao",
}
FORMATOS_VALIDOS = {"csv", "json", "jsonl"}
LLM_PROVIDERS_VALIDOS = {"deepseek", "openai", "callable", "none"}


@dataclass
class Config:
    segmento: str
    n: int
    output_dir: str = "./output"
    region: str | None = None
    llm_provider: str = "deepseek"
    formato: str = "jsonl"
    fuente_dir: str = "data/fuentes"

    def __post_init__(self) -> None:
        if self.segmento not in SEGMENTOS_VALIDOS:
            raise ValueError(
                f"segmento '{self.segmento}' no válido. Opciones: {sorted(SEGMENTOS_VALIDOS)}"
            )
        if self.n <= 0:
            raise ValueError(f"n debe ser mayor que 0, recibido: {self.n}")
        if self.region is not None and self.region not in REGIONES_VALIDAS:
            raise ValueError(
                f"region '{self.region}' no válida. Opciones: {sorted(REGIONES_VALIDAS)}"
            )
        if self.formato not in FORMATOS_VALIDOS:
            raise ValueError(
                f"formato '{self.formato}' no válido. Opciones: {sorted(FORMATOS_VALIDOS)}"
            )
        if self.llm_provider not in LLM_PROVIDERS_VALIDOS:
            raise ValueError(
                f"llm_provider '{self.llm_provider}' no válido. Opciones: {sorted(LLM_PROVIDERS_VALIDOS)}"
            )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def fuente_path(self) -> Path:
        return Path(self.fuente_dir)
