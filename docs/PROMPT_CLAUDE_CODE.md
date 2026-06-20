# Prompt para Claude Code — gen-peru-poblacion

Copia y pega esto en Claude Code (o Cursor / Aider) para arrancar la implementación.

---

```
Lee el archivo docs/SPEC.md y luego sigue exactamente este proceso:

## FASE 1 — Plan (no escribas código todavía)

Genera `docs/plan.md` con:
- Overview del objetivo en 3 oraciones
- 4 fases de implementación en orden (core → agentes → CLI → eval integration)
- Dependencias entre módulos (qué debe existir antes de qué)
- Riesgos identificados con mitigación propuesta
- Estimación de complejidad por módulo (S/M/L)

Espera mi aprobación antes de continuar.

## FASE 2 — Tareas atómicas

Genera `docs/tasks.md` con tareas ordenadas. Cada tarea debe tener:
- ID (T-001, T-002...)
- Descripción en una oración
- Archivos que crea o modifica
- Criterio de done verificable
- Dependencias (qué task debe estar done antes)

Las tareas deben poder ejecutarse de forma independiente.
Espera mi aprobación antes de continuar.

## FASE 3 — Implementación task a task

Por cada tarea:
1. Anuncia qué vas a hacer
2. Implementa
3. Describe los cambios
4. Espera mi confirmación antes de la siguiente

---

## Stack obligatorio

- Python 3.10+
- SDV (GaussianCopulaSynthesizer) para calibración
- LangGraph para orquestación de agentes
- DeepSeek como LLM default (API compatible con OpenAI SDK)
- Typer para CLI
- Pandas + scipy para exportación y verificación KS
- pyproject.toml con hatchling

## Estructura de módulos a crear

src/gen_peru_poblacion/
  __init__.py         → exports: Config, PopulationGenerator, AgentBuilder
  config.py           → dataclass Config con: segmento, n, region, output_dir, llm_provider
  calibrator.py       → class Calibrator: carga JSON de data/fuentes/, entrena GaussianCopula
  generator.py        → class PopulationGenerator: calibrar() → sintetizar() → exportar()
  agent_builder.py    → class AgentBuilder: from_profile(perfil, llm_provider, callable=None)
  providers.py        → LLMProvider interface + DeepSeekProvider + CallableProvider
  exporters.py        → export_csv(), export_json(), export_jsonl()
  verify.py           → run_ks_check(data_dir, fuente_dir) → KSReport
  cli.py              → app Typer con comandos: generate, verify

## Patrones a respetar

1. El core (generación) NUNCA importa nada de LLM — funciona offline
2. AgentBuilder es opcional — solo se importa con [agents] extra
3. Cada perfil generado incluye siempre: synthetic=True, data_sources=[...], perfil_id (uuid4)
4. verify.py es ejecutable como script independiente: python -m gen_peru_poblacion.verify
5. LLM provider sigue el patrón callable de llm_bridge: def my_llm(messages, **kwargs) → str

## Datos fuente

Los JSON en data/fuentes/mype_distribucion_2023.json ya existen.
El calibrator debe leerlos y mapear las distribuciones al GaussianCopulaSynthesizer.

## Tests mínimos requeridos

tests/test_generator.py:
- test_generate_profiles_basic: genera 10 perfiles, verifica schema
- test_profiles_have_required_fields: todos los campos del schema están presentes
- test_synthetic_flag: todos los perfiles tienen synthetic=True
- test_export_csv: archivo CSV es válido y parseable
- test_export_jsonl: archivo JSONL tiene N líneas válidas
- test_ks_basic: KS score se puede calcular sin errores

tests/test_agents.py (solo si [agents] está instalado):
- test_agent_builds_from_profile: AgentBuilder no falla con CallableProvider mock
- test_agent_chat_single_turn: responde algo sin errores
- test_agent_multi_turn_coherence: segundo turno recibe contexto del primero

## CLI esperado

gen-peru-poblacion generate --segmento mype --region lima --n 100 --formato csv
gen-peru-poblacion generate --segmento mype --n 50 --formato jsonl --output ./data
gen-peru-poblacion verify --data-dir ./output --verbose

## Archivos que YA EXISTEN (no recrear)

- docs/SPEC.md
- data/fuentes/mype_distribucion_2023.json
- pyproject.toml
- README.md

## Archivos que debes crear

- src/gen_peru_poblacion/__init__.py
- src/gen_peru_poblacion/config.py
- src/gen_peru_poblacion/calibrator.py
- src/gen_peru_poblacion/generator.py
- src/gen_peru_poblacion/agent_builder.py
- src/gen_peru_poblacion/providers.py
- src/gen_peru_poblacion/exporters.py
- src/gen_peru_poblacion/verify.py
- src/gen_peru_poblacion/cli.py
- tests/test_generator.py
- tests/test_agents.py
- examples/basico.py
- examples/con_agente.py
- LICENSE (Apache 2.0)
- CONTRIBUTING.md
- SECURITY.md
- .github/workflows/ci.yml
- .gitignore

Empieza con FASE 1. No escribas código todavía.
```
