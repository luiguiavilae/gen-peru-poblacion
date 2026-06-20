# Plan de Implementación — gen-peru-poblacion

## Overview

`gen-peru-poblacion` es una librería Python que genera poblaciones sintéticas de emprendedores peruanos (segmento MYPE v1) calibradas con distribuciones reales de INEI/PRODUCE/SBS 2023, usando SDV como motor estadístico. El core funciona completamente offline; los agentes conversacionales son opcionales y se activan con el extra `[agents]`, siguiendo el patrón de separación limpia entre generación y LLM. La v1 entrega CLI, exportación CSV/JSON/JSONL, verificación KS y agentes conectables a cualquier LLM vía callable provider.

---

## Fases de implementación

### Fase 1 — Core (generación offline)
Implementa la cadena central: `config.py` → `calibrator.py` → `generator.py` → `exporters.py`.  
No importa nada relacionado con LLM. Debe funcionar con solo las dependencias base del `pyproject.toml`.  
Incluye `verify.py` como script independiente de verificación KS.

**Módulos:** `config.py`, `calibrator.py`, `generator.py`, `exporters.py`, `verify.py`, `__init__.py`  
**Tests asociados:** `tests/test_generator.py` (todos los casos excepto agent)

### Fase 2 — Agentes (opcional, requiere `[agents]`)
Implementa `providers.py` y `agent_builder.py`.  
Usa imports condicionales (`try/except ImportError`) para que el core no rompa si `langgraph` no está instalado.  
El `AgentBuilder` recibe un perfil sintético y construye un agente LangGraph que simula al emprendedor.

**Módulos:** `providers.py`, `agent_builder.py`  
**Tests asociados:** `tests/test_agents.py`

### Fase 3 — CLI
Implementa `cli.py` con Typer: comandos `generate` y `verify`.  
Depende de `generator.py` (Fase 1) y expone todos los flags documentados en el SPEC.  
No necesita `[agents]`; el CLI base opera solo con el core.

**Módulos:** `cli.py`  
**Tests asociados:** ninguno nuevo (CLI se verifica manualmente con `gen-peru-poblacion --help`)

### Fase 4 — Integración y empaquetado
Agrega los archivos de comunidad, CI y ejemplos: `examples/`, `.github/workflows/ci.yml`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `.gitignore`.  
Verifica que `pyproject.toml` tenga los entry points y extras correctos.  
Corre el test suite completo y confirma que la instalación limpia (`pip install -e .`) funciona.

**Archivos:** `examples/basico.py`, `examples/con_agente.py`, `.github/workflows/ci.yml`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `.gitignore`

---

## Dependencias entre módulos

```
data/fuentes/*.json
        │
        ▼
   calibrator.py ◄── config.py ──► exporters.py
        │                               │
        └──────────► generator.py ◄─────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           cli.py              verify.py (standalone)
              │
              │   (solo con [agents])
              ▼
         providers.py
              │
              ▼
        agent_builder.py
```

**Reglas de importación obligatorias:**
- `generator.py` NUNCA importa `providers.py` ni `agent_builder.py`
- `agent_builder.py` importa `generator.py` solo para tipado de perfiles (dict), no como dependencia de ejecución
- `verify.py` no importa nada del paquete — solo `pandas` y `scipy`
- `cli.py` importa `agent_builder` con `try/except ImportError` y avisa si `[agents]` no está instalado

**Orden mínimo de existencia antes de testear:**
1. `config.py` (ninguna dependencia interna)
2. `calibrator.py` (requiere `config.py`)
3. `exporters.py` (ninguna dependencia interna)
4. `generator.py` (requiere `calibrator.py` + `exporters.py`)
5. `verify.py` (independiente — puede existir en cualquier momento)
6. `providers.py` (ninguna dependencia interna del paquete)
7. `agent_builder.py` (requiere `providers.py`)
8. `cli.py` (requiere `generator.py` + `verify.py`)

---

## Riesgos identificados y mitigación

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|--------|-------------|---------|------------|
| R-01 | SDV `GaussianCopulaSynthesizer` no preserva distribuciones categóricas con la fidelidad requerida (KS < 0.70) | Media | Alto | Definir `SingleTableMetadata` explícita con tipos `categorical` para variables discretas; usar `fit_processed_data` si es necesario. Validar KS en cada iteración de calibración antes de pasar a generación. |
| R-02 | Correlaciones del JSON fuente (`correlaciones_conocidas`) no se pueden inyectar directamente en SDV | Alta | Medio | SDV aprende correlaciones del dataset de entrenamiento. Pre-procesar: generar un dataset sintético pequeño (~5000 filas) que respete manualmente las correlaciones documentadas, usarlo como datos de entrenamiento del `GaussianCopulaSynthesizer`. |
| R-03 | `langgraph` o `openai` SDK cambian API entre versiones | Media | Medio | Fijar versiones mínimas en `pyproject.toml`. Abstraer todo el acceso a LangGraph detrás de `agent_builder.py` para aislar cambios futuros. |
| R-04 | `verify.py` como script independiente falla si el paquete no está instalado en el path | Baja | Bajo | Agregar `if __name__ == "__main__"` con manejo de path, y documentar en README que se ejecuta como `python -m gen_peru_poblacion.verify`. |
| R-05 | Tiempo de generación excede 10 segundos para N=100 (SPEC CU-01) | Baja | Medio | SDV necesita entrenar el modelo en el primer `fit()`. Cachear el modelo entrenado con `pickle` en un archivo `.sdv_model` dentro del directorio de trabajo para evitar re-entrenamiento. |
| R-06 | Import circular entre `generator.py` y módulos de agentes | Baja | Alto | Regla estricta: el grafo de imports es un DAG sin ciclos. `agent_builder.py` nunca es importado por módulos core; solo `cli.py` lo importa opcionalmente. |

---

## Estimación de complejidad por módulo

| Módulo | Complejidad | Justificación |
|--------|-------------|---------------|
| `config.py` | **S** | Dataclass simple con validación básica |
| `calibrator.py` | **L** | Integración no trivial con SDV: metadata, fit, correlaciones, caché del modelo |
| `generator.py` | **M** | Orquesta calibrator + exporters; maneja uuid4, campos obligatorios, data_sources |
| `exporters.py` | **S** | CSV/JSON/JSONL con pandas — lógica directa |
| `providers.py` | **S** | Interfaz simple + DeepSeekProvider (wrapper OpenAI SDK) + CallableProvider |
| `agent_builder.py` | **M** | LangGraph graph + system prompt contextualizado desde perfil + multi-turn state |
| `verify.py` | **M** | KS test por variable, reporte estructurado, ejecutable como `__main__` |
| `cli.py` | **S** | Typer con 2 comandos, flags documentados, manejo de errores básico |
| `tests/test_generator.py` | **M** | 6 casos, requiere fixtures de datos y mocks de filesystem |
| `tests/test_agents.py` | **M** | 3 casos con CallableProvider mock, verificación de coherencia multi-turno |
| `examples/` | **S** | Scripts de demostración, no lógica nueva |
| `.github/workflows/ci.yml` | **S** | Matrix Python 3.10/3.11/3.12, ruff + pytest |

**Total estimado:** ~3-4 días de implementación concentrada en Fases 1-2. Fases 3-4 son mecánicas.

---

## Criterios de completitud por fase

- **Fase 1 completa:** `test_generator.py` pasa al 100%; `python -m gen_peru_poblacion.verify` corre sin errores; KS ≥ 0.70 en variables principales
- **Fase 2 completa:** `test_agents.py` pasa con `CallableProvider` mock; `AgentBuilder` importa sin errores si `[agents]` no está instalado (muestra `ImportError` claro)
- **Fase 3 completa:** `gen-peru-poblacion --help` muestra todos los flags; `generate` produce archivo válido; `verify` imprime reporte KS
- **Fase 4 completa:** `pip install -e ".[dev,agents]"` funciona en entorno limpio; CI pasa en GitHub Actions; ejemplos corren sin errores
