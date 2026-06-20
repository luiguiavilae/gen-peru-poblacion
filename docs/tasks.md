# Tareas atómicas — gen-peru-poblacion

Orden de ejecución: T-001 → T-018. Cada tarea es independiente dentro de su bloque de dependencias.

---

## Fase 1 — Core (generación offline)

### T-001 — Dataclass Config
**Descripción:** Crear `config.py` con el dataclass `Config` que contiene los parámetros de generación.  
**Archivos:** crea `src/gen_peru_poblacion/config.py`  
**Criterio de done:** `from gen_peru_poblacion.config import Config; c = Config(segmento="mype", n=10, region="lima_metropolitana", output_dir="./output", llm_provider="deepseek")` no lanza excepción; `Config` valida que `n > 0` y que `segmento` sea un valor permitido.  
**Dependencias:** ninguna

---

### T-002 — Calibrator con SDV
**Descripción:** Crear `calibrator.py` con la clase `Calibrator` que carga el JSON fuente, construye un dataset de entrenamiento y entrena un `GaussianCopulaSynthesizer`.  
**Archivos:** crea `src/gen_peru_poblacion/calibrator.py`  
**Criterio de done:** `from gen_peru_poblacion.calibrator import Calibrator; cal = Calibrator("mype"); cal.fit()` no lanza excepción; `cal.sample(10)` devuelve un `pd.DataFrame` con 10 filas y columnas: `region`, `rubro`, `tamaño`, `formalizado`, `canal_venta`, `adopcion_digital`, `credito`, `edad_dueño`, `nivel_educativo`, `ingreso_mensual_soles`, `lengua_materna`.  
**Dependencias:** T-001

---

### T-003 — Exporters
**Descripción:** Crear `exporters.py` con las funciones `export_csv()`, `export_json()` y `export_jsonl()` que reciben una lista de dicts y un path de salida.  
**Archivos:** crea `src/gen_peru_poblacion/exporters.py`  
**Criterio de done:** Dado una lista de 5 dicts, cada función crea el archivo correspondiente; `pd.read_csv(path)` parsea el CSV sin errores; `json.load(open(path))` parsea el JSON; todas las líneas del JSONL son JSON válido individualmente.  
**Dependencias:** ninguna

---

### T-004 — PopulationGenerator
**Descripción:** Crear `generator.py` con la clase `PopulationGenerator` que orquesta calibrar → sintetizar → agregar campos obligatorios → exportar.  
**Archivos:** crea `src/gen_peru_poblacion/generator.py`  
**Criterio de done:** `gen = PopulationGenerator(Config(...)); perfiles = gen.generate()` devuelve una lista de dicts donde cada dict tiene `synthetic=True`, `data_sources=["INEI-ENAHO-2023", "PRODUCE-ENAMIN-2023", "SBS-2023"]` y `perfil_id` (string UUID4 único entre perfiles); `gen.export(perfiles, formato="csv", path="./out")` crea el archivo.  
**Dependencias:** T-001, T-002, T-003

---

### T-005 — verify.py (KS check standalone)
**Descripción:** Crear `verify.py` con la función `run_ks_check(data_dir, fuente_dir)` que carga perfiles generados y calcula KS similarity contra las distribuciones fuente, y un bloque `__main__` para ejecución directa.  
**Archivos:** crea `src/gen_peru_poblacion/verify.py`  
**Criterio de done:** `python -m gen_peru_poblacion.verify --data-dir ./output --fuente-dir ./data/fuentes` imprime un `KSReport` con score por variable y un score global; el módulo importa sin errores aunque `generator.py` no haya sido importado; devuelve `KSReport` con atributo `passed: bool` (True si score global ≥ 0.70).  
**Dependencias:** ninguna (usa solo `pandas`, `scipy`, `json`)

---

### T-006 — __init__.py con exports públicos
**Descripción:** Crear `__init__.py` que exporta `Config`, `PopulationGenerator` y `AgentBuilder` (este último con import lazy para no requerir `[agents]`).  
**Archivos:** crea `src/gen_peru_poblacion/__init__.py`  
**Criterio de done:** `from gen_peru_poblacion import Config, PopulationGenerator` funciona con solo las dependencias core; `from gen_peru_poblacion import AgentBuilder` lanza `ImportError` claro con mensaje "Instala gen-peru-poblacion[agents]" si `langgraph` no está disponible; `__version__ = "0.1.0"` está definido.  
**Dependencias:** T-001, T-004

---

### T-007 — tests/test_generator.py
**Descripción:** Crear el archivo de tests del core con los 6 casos requeridos por el SPEC.  
**Archivos:** crea `tests/test_generator.py`  
**Criterio de done:** `pytest tests/test_generator.py -v` pasa los 6 tests: `test_generate_profiles_basic` (N=10, schema completo), `test_profiles_have_required_fields` (todos los campos presentes), `test_synthetic_flag` (`synthetic=True` en todos), `test_unique_perfil_ids` (no hay UUIDs duplicados), `test_export_csv` (CSV parseable con pandas), `test_export_jsonl` (JSONL con N líneas válidas); `test_ks_basic` (KSReport calculable sin errores).  
**Dependencias:** T-004, T-005

---

## Fase 2 — Agentes (requiere `[agents]`)

### T-008 — providers.py (LLM providers)
**Descripción:** Crear `providers.py` con la interfaz `LLMProvider`, `DeepSeekProvider` (wrapper OpenAI SDK apuntando a DeepSeek) y `CallableProvider` (acepta cualquier `callable(messages, **kwargs) → str`).  
**Archivos:** crea `src/gen_peru_poblacion/providers.py`  
**Criterio de done:** `from gen_peru_poblacion.providers import CallableProvider; p = CallableProvider(lambda msgs, **kw: "ok"); p.complete([{"role":"user","content":"hola"}]) == "ok"`; `DeepSeekProvider` instancia sin error si `openai` está disponible; el módulo importa sin error si `openai` NO está instalado (el import de `openai` está dentro de `DeepSeekProvider.__init__`).  
**Dependencias:** ninguna

---

### T-009 — agent_builder.py (AgentBuilder con LangGraph)
**Descripción:** Crear `agent_builder.py` con la clase `AgentBuilder` que construye un agente LangGraph que simula al emprendedor del perfil dado, con soporte multi-turno.  
**Archivos:** crea `src/gen_peru_poblacion/agent_builder.py`  
**Criterio de done:** `AgentBuilder.from_profile(perfil, provider=CallableProvider(...))` devuelve un objeto con método `chat(mensaje: str) → str`; el system prompt incluye los valores del perfil (región, rubro, edad, nivel digital, etc.); dos llamadas consecutivas a `chat()` en la misma instancia mantienen el historial (el segundo turno puede referenciar el primero); el módulo lanza `ImportError` con mensaje claro si `langgraph` no está instalado.  
**Dependencias:** T-008

---

### T-010 — tests/test_agents.py
**Descripción:** Crear el archivo de tests de agentes con los 3 casos requeridos, usando `CallableProvider` como mock.  
**Archivos:** crea `tests/test_agents.py`  
**Criterio de done:** `pytest tests/test_agents.py -v` pasa los 3 tests cuando `[agents]` está instalado: `test_agent_builds_from_profile` (no lanza excepción), `test_agent_chat_single_turn` (devuelve string no vacío), `test_agent_multi_turn_coherence` (segundo turno recibe historial, verificado inspeccionando los mensajes que llegan al callable mock); si `langgraph` no está instalado, todos los tests son `skip` con mensaje claro.  
**Dependencias:** T-009

---

## Fase 3 — CLI

### T-011 — cli.py con Typer
**Descripción:** Crear `cli.py` con la app Typer y los dos comandos `generate` y `verify` con todos los flags documentados.  
**Archivos:** crea `src/gen_peru_poblacion/cli.py`  
**Criterio de done:** `gen-peru-poblacion --help` lista ambos comandos; `gen-peru-poblacion generate --help` muestra flags `--segmento`, `--region`, `--n`, `--formato`, `--output`; `gen-peru-poblacion generate --segmento mype --n 10 --formato csv` crea un archivo CSV en `./output/`; `gen-peru-poblacion verify --data-dir ./output` imprime el KS report; errores de argumento inválido muestran mensaje útil sin traceback.  
**Dependencias:** T-004, T-005

---

## Fase 4 — Empaquetado y comunidad

### T-012 — examples/basico.py
**Descripción:** Crear el ejemplo de uso básico que genera perfiles y los exporta en los 3 formatos sin necesidad de `[agents]`.  
**Archivos:** crea `examples/basico.py`  
**Criterio de done:** `python examples/basico.py` corre sin errores con solo las dependencias core instaladas y produce 3 archivos de salida; el script tiene comentarios que explican cada paso.  
**Dependencias:** T-004

---

### T-013 — examples/con_agente.py
**Descripción:** Crear el ejemplo de uso con agente conversacional usando `CallableProvider` con un mock simple para que sea ejecutable sin credenciales LLM reales.  
**Archivos:** crea `examples/con_agente.py`  
**Criterio de done:** `python examples/con_agente.py` corre sin errores con `[agents]` instalado; usa `CallableProvider` con una función mock que devuelve respuestas fijas; el script muestra cómo sustituir el mock por `DeepSeekProvider` con un comentario claro.  
**Dependencias:** T-009

---

### T-014 — LICENSE (Apache 2.0)
**Descripción:** Crear el archivo LICENSE con el texto completo de Apache 2.0 con el año y autor correctos.  
**Archivos:** crea `LICENSE`  
**Criterio de done:** El archivo contiene el texto oficial de Apache 2.0 con `Copyright 2025` en el encabezado; `pyproject.toml` ya referencia `{ file = "LICENSE" }` (ya configurado — solo verificar).  
**Dependencias:** ninguna

---

### T-015 — CONTRIBUTING.md
**Descripción:** Crear la guía de contribución con instrucciones de setup, convenciones de código y proceso de PR.  
**Archivos:** crea `CONTRIBUTING.md`  
**Criterio de done:** El archivo documenta: cómo instalar en modo dev (`pip install -e ".[dev,agents]"`), cómo correr tests (`pytest`), cómo correr el linter (`ruff check .`), la convención de commits (tipo: descripción), y cómo agregar nuevos segmentos de datos.  
**Dependencias:** ninguna

---

### T-016 — SECURITY.md
**Descripción:** Crear la política de seguridad con el proceso de reporte de vulnerabilidades y el compromiso de privacidad de datos.  
**Archivos:** crea `SECURITY.md`  
**Criterio de done:** El archivo documenta: cómo reportar vulnerabilidades (email o GitHub private advisory), el compromiso de que la librería nunca procesa datos reales de personas, y que los datos fuente son distribuciones agregadas sin microdatos.  
**Dependencias:** ninguna

---

### T-017 — .gitignore
**Descripción:** Crear el `.gitignore` adaptado al stack Python/hatchling del proyecto, incluyendo exclusiones específicas del proyecto.  
**Archivos:** crea `.gitignore`  
**Criterio de done:** El `.gitignore` excluye: `__pycache__/`, `*.pyc`, `.venv/`, `dist/`, `*.egg-info/`, `.coverage`, `htmlcov/`, `.sdv_model` (caché de modelo SDV), `output/` (archivos generados localmente), `.env`, `.mypy_cache/`, `.ruff_cache/`.  
**Dependencias:** ninguna

---

### T-018 — .github/workflows/ci.yml
**Descripción:** Crear el workflow de GitHub Actions que corre ruff, mypy y pytest en matrix Python 3.10/3.11/3.12 en cada push y PR.  
**Archivos:** crea `.github/workflows/ci.yml`  
**Criterio de done:** El archivo es YAML válido; define un job `test` con matrix `python-version: ["3.10", "3.11", "3.12"]`; los pasos son: checkout → setup-python → `pip install -e ".[dev,agents]"` → `ruff check .` → `pytest --cov` → subida de cobertura con `coverage xml`; se dispara en `push` y `pull_request` hacia `main`.  
**Dependencias:** ninguna

---

## Resumen de dependencias

```
T-001 ──► T-002 ──► T-004 ──► T-007
                  ┘         └──► T-011
T-003 ──────────► T-004
T-005 ──────────────────────► T-007, T-011
T-008 ──► T-009 ──► T-010
                 └──► T-013
T-004 ──► T-012

T-014, T-015, T-016, T-017, T-018, T-006 — sin dependencias internas
```

**Tareas que pueden ejecutarse en paralelo:**
- T-001, T-003, T-005, T-008, T-014, T-015, T-016, T-017, T-018 (ninguna depende de otra)
- T-002 y T-003 en paralelo (ambas dependen solo de T-001 para T-002, y ninguna para T-003)
- T-010 y T-013 en paralelo (ambas dependen de T-009)
