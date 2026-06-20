"""
Tests del módulo de agentes (T-010).
Todos los tests usan CallableProvider con mocks — no requieren LLM real ni API key.
Si langgraph no está instalado, todos los tests son marcados como skip.
"""
from __future__ import annotations

import pytest

try:
    from gen_peru_poblacion.agent_builder import Agent, AgentBuilder, _build_system_prompt
    from gen_peru_poblacion.providers import CallableProvider
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AGENTS_AVAILABLE,
    reason="Requiere pip install gen-peru-poblacion[agents]",
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def perfil_mype():
    return {
        "perfil_id": "test-agent-001",
        "synthetic": True,
        "data_sources": ["INEI-ENAHO-2023"],
        "segmento": "mype",
        "edad_dueño": 38,
        "region": "lima_metropolitana",
        "rubro": "comercio_minorista",
        "nivel_educativo": "secundaria_completa",
        "formalizado": "completamente_informal",
        "adopcion_digital": "media",
        "credito": "sin_credito",
        "ingreso_mensual_soles": 1800,
        "tamaño": "unipersonal",
        "canal_venta": "whatsapp_incluido",
        "lengua_materna": "castellano",
    }


@pytest.fixture
def simple_provider():
    """CallableProvider que devuelve una respuesta fija predecible."""
    return CallableProvider(lambda msgs, **kw: "Sí, tengo mi negocio de bodega en Lima.")


@pytest.fixture
def recording_provider():
    """CallableProvider que registra todos los mensajes que recibe."""
    received: list[list[dict]] = []

    def fn(messages, **kw):
        received.append(list(messages))
        return f"Respuesta al mensaje número {len(received)}."

    provider = CallableProvider(fn)
    provider._received = received
    return provider


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_agent_builds_from_profile(perfil_mype, simple_provider):
    """AgentBuilder.from_profile no falla con un perfil y CallableProvider válidos."""
    agent = AgentBuilder.from_profile(perfil_mype, simple_provider)
    assert isinstance(agent, Agent)


def test_agent_chat_single_turn(perfil_mype, simple_provider):
    """Un turno de chat devuelve un string no vacío coherente con la respuesta del provider."""
    agent = AgentBuilder.from_profile(perfil_mype, simple_provider)
    response = agent.chat("¿Usas aplicaciones bancarias para tu negocio?")

    assert isinstance(response, str)
    assert len(response) > 0
    # El provider siempre devuelve la misma cadena
    assert "bodega" in response or "Lima" in response or "negocio" in response


def test_agent_multi_turn_coherence(perfil_mype, recording_provider):
    """
    En el segundo turno, el provider recibe el contexto completo del turno anterior:
    - Mensaje del usuario del turno 1
    - Respuesta del agente del turno 1
    - Nuevo mensaje del usuario del turno 2
    """
    agent = AgentBuilder.from_profile(perfil_mype, recording_provider)

    agent.chat("¿Usas aplicaciones bancarias para tu negocio?")
    agent.chat("¿Cuánto ganas al mes aproximadamente?")

    assert len(recording_provider._received) == 2, (
        "El provider debería haber recibido exactamente 2 llamadas"
    )

    # El contexto del turno 2 debe incluir el intercambio del turno 1
    msgs_turno_2 = recording_provider._received[1]
    all_content = " ".join(m.get("content", "") for m in msgs_turno_2)

    assert "¿Usas aplicaciones bancarias" in all_content, (
        "El mensaje del turno 1 no aparece en el contexto del turno 2"
    )
    assert "Respuesta al mensaje número 1" in all_content, (
        "La respuesta del turno 1 no aparece en el contexto del turno 2"
    )
    assert "¿Cuánto ganas" in all_content, (
        "El mensaje del turno 2 debe estar presente en el contexto"
    )


def test_system_prompt_includes_profile_data(perfil_mype):
    """El system prompt generado contiene los datos clave del perfil."""
    prompt = _build_system_prompt(perfil_mype)

    assert "38" in prompt                        # edad
    assert "Lima" in prompt                      # región
    assert "comercio" in prompt.lower()          # rubro
    assert "1,800" in prompt or "1800" in prompt # ingreso
    assert "WhatsApp" in prompt                  # canal/adopción digital
    assert "secundaria" in prompt.lower()        # nivel educativo
    assert "primera persona" in prompt           # instrucción de registro
