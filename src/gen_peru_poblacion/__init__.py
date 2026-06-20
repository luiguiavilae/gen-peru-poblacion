from __future__ import annotations

from gen_peru_poblacion.config import Config
from gen_peru_poblacion.generator import PopulationGenerator

__version__ = "0.1.0"
__all__ = ["Config", "PopulationGenerator", "AgentBuilder", "__version__"]


def __getattr__(name: str) -> object:
    if name == "AgentBuilder":
        try:
            from gen_peru_poblacion.agent_builder import AgentBuilder
            return AgentBuilder
        except ImportError as exc:
            raise ImportError(
                "AgentBuilder requiere dependencias opcionales. "
                "Instala con: pip install gen-peru-poblacion[agents]"
            ) from exc
    raise AttributeError(f"El módulo 'gen_peru_poblacion' no tiene atributo '{name}'")
