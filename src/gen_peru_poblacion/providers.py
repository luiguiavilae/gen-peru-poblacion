"""
providers.py — Interfaz LLMProvider y sus implementaciones.

Patrón: cualquier callable(messages, **kwargs) -> str puede actuar como LLM.
El core del paquete NUNCA importa este módulo.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

Messages = list[dict[str, str]]


class LLMProvider(ABC):
    """Interfaz base para cualquier proveedor de LLM."""

    @abstractmethod
    def complete(self, messages: Messages, **kwargs: Any) -> str:
        """Recibe una lista de mensajes OpenAI-format y devuelve la respuesta como string."""


class DeepSeekProvider(LLMProvider):
    """
    Proveedor DeepSeek usando el SDK de OpenAI apuntando a la API de DeepSeek.
    Compatible también con cualquier API OpenAI-compatible (Groq, Together, etc.)
    cambiando base_url y model.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "DeepSeekProvider requiere el paquete openai. "
                "Instala con: pip install gen-peru-poblacion[agents]"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, messages: Messages, **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.choices[0].message.content or ""


class CallableProvider(LLMProvider):
    """
    Envuelve cualquier callable con firma (messages, **kwargs) -> str como LLMProvider.

    Uso:
        def mi_llm(messages, **kwargs):
            return "respuesta fija"

        provider = CallableProvider(mi_llm)
    """

    def __init__(self, fn: Callable[..., str]) -> None:
        if not callable(fn):
            raise TypeError(f"fn debe ser callable, recibido: {type(fn)}")
        self._fn = fn

    def complete(self, messages: Messages, **kwargs: Any) -> str:
        return self._fn(messages, **kwargs)
