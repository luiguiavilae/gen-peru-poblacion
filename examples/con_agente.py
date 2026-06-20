"""
Ejemplo con agente conversacional usando CallableProvider (mock sin credenciales).
Requiere: pip install gen-peru-poblacion[agents]

Para usar DeepSeek real, reemplaza el mock por DeepSeekProvider (ver comentario al final).

Ejecutar desde la raíz del proyecto:
    python examples/con_agente.py
"""
from gen_peru_poblacion import Config, PopulationGenerator
from gen_peru_poblacion.agent_builder import AgentBuilder
from gen_peru_poblacion.providers import CallableProvider

# 1. Generar un perfil sintético
print("Generando perfil de emprendedor MYPE...")
cfg = Config(segmento="mype", n=1)
gen = PopulationGenerator(cfg)
perfiles = gen.generate()
perfil = perfiles[0]

print(f"\nPerfil generado:")
print(f"  Región:         {perfil['region']}")
print(f"  Rubro:          {perfil['rubro']}")
print(f"  Edad:           {perfil['edad_dueño']}")
print(f"  Adopción digital: {perfil['adopcion_digital']}")
print(f"  Lengua materna: {perfil['lengua_materna']}")
print(f"  Ingreso:        S/ {perfil['ingreso_mensual_soles']:,}")

# 2. Definir el provider (mock local — responde siempre lo mismo)
# Para usar DeepSeek real, reemplaza este bloque por:
#
#   from gen_peru_poblacion.providers import DeepSeekProvider
#   import os
#   provider = DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"])
#
def mock_llm(messages, **kwargs):
    """Mock que simula una respuesta del LLM para demostración."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    return (
        f"[MOCK — en producción aquí respondería el LLM real]\n"
        f"Pregunta recibida: '{last_user}'\n"
        f"El agente respondería en primera persona desde el perfil configurado."
    )

provider = CallableProvider(mock_llm)

# 3. Construir el agente
print("\nConstruyendo agente conversacional...")
agent = AgentBuilder.from_profile(perfil, llm_provider=provider)
print("  → Agente listo")

# 4. Conversación de ejemplo
preguntas = [
    "¿Usas aplicaciones bancarias para tu negocio?",
    "¿Y cómo manejas los pagos de tus clientes?",
]

print("\n--- Conversación ---")
for pregunta in preguntas:
    print(f"\nUsuario: {pregunta}")
    respuesta = agent.chat(pregunta)
    print(f"Agente:  {respuesta}")

print("\n--- Fin del ejemplo ---")
print("Para habilitar respuestas reales, configura DeepSeekProvider con tu API key.")
