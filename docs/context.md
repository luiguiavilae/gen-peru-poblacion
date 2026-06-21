# Contexto técnico — gen-peru-poblacion

Documento de referencia para sesiones futuras. Captura el estado real del código,
decisiones no obvias tomadas durante la implementación y advertencias que no están
en el código fuente.

---

## Qué es este proyecto

Librería Python open source que genera **poblaciones sintéticas de emprendedores
peruanos (segmento MYPE)** calibradas con distribuciones reales de INEI, PRODUCE
y SBS 2023. Incluye agentes conversacionales opcionales para simular respuestas
cualitativas en primera persona.

- **Core (offline):** genera perfiles sin necesidad de LLM ni internet.
- **Agentes (opcional):** `pip install gen-peru-poblacion[agents]` activa LangGraph
  + DeepSeek (o cualquier callable compatible).
- **CLI:** `gen-peru-poblacion generate / verify`.

---

## Estado actual (post-implementación completa)

| Módulo | Líneas | Estado | Tests |
|--------|--------|--------|-------|
| `config.py` | 56 | ✓ completo | — |
| `calibrator.py` | 343 | ✓ completo | indirecto vía generator |
| `generator.py` | 130 | ✓ completo | test_generator.py |
| `exporters.py` | 34 | ✓ completo | test_generator.py |
| `verify.py` | 238 | ✓ completo | test_generator.py |
| `providers.py` | 72 | ✓ completo | test_agents.py |
| `agent_builder.py` | 390 | ✓ completo | test_agents.py |
| `cli.py` | 280 | ✓ completo | verificado manualmente |
| `__init__.py` | 20 | ✓ completo | — |
| **Total** | **1 807** | | **11/11 passed** |

**KS similarity real (n=500):** global 0.964 — todas las variables sobre 0.940.
El umbral requerido era ≥ 0.70.

---

## Arquitectura y grafo de dependencias

```
data/fuentes/mype_distribucion_2023.json
         │
         ▼
   calibrator.py ◄── config.py ──► exporters.py
         │                               │
         └──────────► generator.py ◄─────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
            cli.py              verify.py (standalone)

   providers.py ──► agent_builder.py
                          ▲
                     (solo [agents])
                     cli.py la importa con try/except
```

**Regla crítica:** `generator.py` nunca importa `providers.py` ni `agent_builder.py`.
`verify.py` no importa nada del paquete propio — solo `pandas`, `scipy`, `json`.

---

## Datos fuente

**Archivo:** `data/fuentes/mype_distribucion_2023.json`

**Fuentes originales:** INEI-ENAHO 2023, PRODUCE-ENAMIN 2023, SBS-Estadísticas 2023.

**Estructura del JSON — dos formatos coexisten:**

```
# Formato 1: con clave "valores" (region, rubro, tamaño, nivel_educativo, lengua_materna)
"region": { "valores": { "lima_metropolitana": 0.42, ... } }

# Formato 2: sin "valores" — probs al nivel raíz (adopcion_digital, formalizado, credito)
"adopcion_digital": { "nula": 0.31, "baja": 0.35, "media": 0.24, "alta": 0.10, ... }
```

`verify.py._fuente_distribution()` maneja ambos formatos. **Si se agregan nuevas
variables al JSON, verificar en qué formato están antes de modificar esa función.**

**Correlaciones documentadas (usadas en Iman-Conover):**

| Par | Coeficiente | Índices en matriz C (0-8) |
|-----|-------------|---------------------------|
| adopcion_digital ↔ edad | −0.31 | C[2,0] |
| adopcion_digital ↔ nivel_educativo | +0.42 | C[2,1] |
| credito_formal ↔ formalizado | +0.55 | C[4,3] |
| ingreso ↔ tamaño | +0.48 | C[6,5] |
| adopcion_digital ↔ region_lima | +0.38 | C[2,7] |
| canal_whatsapp ↔ edad_menor_45 | +0.29 | → edad ↔ whatsapp: −0.29, C[8,0] |

Más 3 correlaciones secundarias inferidas de dominio (no en fuente):
`nivel_edu ↔ formalizado: 0.20`, `adopcion ↔ canal_whatsapp: 0.30`,
`region_lima ↔ nivel_edu: 0.15`.

---

## Decisiones de implementación no obvias

### 1. Iman-Conover para inducir correlaciones en el training data de SDV

**Problema:** SDV no tiene API para inyectar una matriz de correlaciones directamente.
Si se alimenta con datos muestreados de forma independiente, SDV aprende correlaciones
cercanas a cero.

**Solución implementada:** generar 5 000 filas de training data usando Iman-Conover:
1. Muestrear cada variable según su distribución marginal (respeta INEI/PRODUCE/SBS).
2. Construir una matriz de correlación de rango 9×9 desde las correlaciones documentadas.
3. Proyectarla a PSD via Higham 2002 (`_nearest_psd`).
4. Reordenar las columnas para que sus rankings coincidan con muestras de una normal
   multivariada con esa matriz (`_iman_conover`).
5. Mapear los numéricos inducidos de vuelta a categorías string.
6. Fedear ese dataset corregido a `GaussianCopulaSynthesizer.fit()`.

**Resultado:** SDV aprende las correlaciones latentes gaussianas, que son más altas
que las Spearman/Pearson observadas. Las Pearson en el training data son atenuadas
para variables binarias (la correlación máxima alcanzable está acotada por las proporciones
marginales — esto es matemáticamente correcto, no un bug).

**Archivo:** `calibrator.py` — métodos `_build_corr_matrix`, `_nearest_psd`, `_iman_conover`,
`_build_training_data`.

### 2. Registro de habla en el system prompt del agente

`agent_builder._register_level(perfil)` calcula un score 1–4 desde:

```
score = edu_ordinal + adopcion_ordinal + (1 si Lima) + (1 si edad < 40)
```

| Score | Nivel | Perfil típico |
|-------|-------|---------------|
| ≤ 4 | 1 — muy coloquial/rural | sin escolaridad, mayor, zona rural |
| 5–7 | 2 — informal urbano | secundaria, Lima/provincia, mediana edad |
| 8–10 | 3 — informal técnico | técnica, Lima, adopción media |
| ≥ 11 | 4 — fluido/educado | universitaria, Lima, joven, alta adopción |

El system prompt concatena:
- Instrucciones de registro (`_REGISTER_INSTRUCTIONS[level]`)
- Nota cultural de región (`_REGION_REGISTER_NOTES[region]`)
- Nota de lengua materna, solo para quechua/aymara (`_LENGUA_NOTA[lengua]`)

El agente no revela su naturaleza sintética a menos que le pregunten directamente
— esto está codificado explícitamente en la sección `SOBRE TU NATURALEZA` del prompt.

### 3. Caché del modelo SDV

`Calibrator.fit()` guarda el modelo entrenado en `.sdv_model_{segmento}.pkl`
en el directorio de trabajo. En la segunda ejecución, carga desde caché sin re-entrenar.

**Implicación:** si se modifica `calibrator.py` o el JSON fuente, hay que borrar
`.sdv_model_mype.pkl` para que los cambios surtan efecto. El archivo está en `.gitignore`.

### 4. `verify.py` detecta automáticamente CSV/JSONL/JSON

`_load_generated(data_dir)` lee en orden: `*.csv` → `*.jsonl` → `*.json`.
Si hay varios archivos del mismo tipo, los concatena. Si no hay ninguno, lanza
`FileNotFoundError` con el path del directorio.

### 5. `adopcion_digital` en el JSON no tiene clave `"valores"`

A diferencia de `region`, `rubro`, `tamaño`, `nivel_educativo` y `lengua_materna`,
el bloque `adopcion_digital` en el JSON tiene las probabilidades directamente al
nivel raíz (junto con campos adicionales como `usa_app_bancaria_negocio`).
`verify._fuente_distribution()` lo maneja filtrando claves que empiezan con `_`
y valores que no son `float`. Si en el futuro se reestructura ese bloque, verificar
que el filtro siga siendo correcto.

### 6. LangGraph MemorySaver para multi-turno

El `Agent` usa `langgraph.checkpoint.memory.MemorySaver` con un `thread_id` UUID4
generado en `__init__`. Cada llamada a `chat()` solo pasa el nuevo `HumanMessage` —
el grafo acumula el historial automáticamente. `reset()` genera un nuevo `thread_id`,
lo que efectivamente descarta el estado anterior sin destruir el grafo compilado.

**Versión requerida:** `langgraph >= 1.0` (instalado: 1.2.0). La API de MemorySaver
cambió entre 0.1.x y 0.2.x — el código actual es incompatible con 0.1.x.

### 7. `__init__.py` con import lazy de `AgentBuilder`

`AgentBuilder` se expone en `__all__` pero su import es lazy via `__getattr__`.
Si `langgraph` no está instalado, `from gen_peru_poblacion import AgentBuilder`
lanza `ImportError` con mensaje claro. Esto permite que `from gen_peru_poblacion
import Config, PopulationGenerator` funcione sin `[agents]` instalado.

---

## Advertencias de entorno conocidas

### NumPy 2.x / torch incompatibilidad

El entorno de desarrollo tiene NumPy 2.4.4 instalado. SDV 1.37 arrastra `sdmetrics`
que arrastra `torch`, compilado contra NumPy 1.x. Esto produce un `UserWarning` en
stderr al importar SDV:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.4
```

**No afecta la funcionalidad** — SDV y GaussianCopulaSynthesizer operan correctamente.
Para eliminar el warning: `pip install "numpy<2"`.

### `FutureWarning` de SDV sobre `SingleTableMetadata`

SDV 1.37 deprecó `SingleTableMetadata` en favor de `Metadata`. El código envuelve
el `fit()` en `warnings.catch_warnings()` para suprimir ambos warnings dentro del
bloque. No impacta la API externa.

---

## Variables que participan en correlaciones (índices en matriz 9×9)

```
Índice | Variable          | Tipo
-------|-------------------|-----------
  0    | edad_num          | continua
  1    | nivel_edu_num     | ordinal 0–8
  2    | adopcion_num      | ordinal 0–3
  3    | formalizado_num   | binaria
  4    | credito_formal_num| binaria
  5    | tamaño_num        | ordinal 0–3
  6    | ingreso_num       | continua (log-normal)
  7    | region_lima_num   | binaria (derivada de region)
  8    | canal_whatsapp_num| binaria (derivada de canal_venta)
```

Este orden está hardcodeado en `_IC_VARS` y en `_build_corr_matrix`.
**No cambiar el orden sin actualizar ambas listas simultáneamente.**

---

## Extensión: agregar un nuevo segmento

1. Crear `data/fuentes/{segmento}_distribucion_XXXX.json` con la misma estructura.
2. Agregar la entrada en `calibrator.py → SEGMENTO_FUENTE`.
3. Si el nuevo segmento tiene variables con estructura diferente al MYPE (ej. sin
   `formalizado`), revisar `_build_training_data` — está escrita asumiendo las 11
   columnas del segmento MYPE.
4. Agregar `segmento` a `SEGMENTOS_VALIDOS` en `config.py`.
5. Agregar tests en `tests/test_generator.py`.

---

## Extensión: agregar un nuevo LLM provider

Implementar `LLMProvider` de `providers.py`:

```python
class MiProvider(LLMProvider):
    def complete(self, messages: list[dict], **kwargs) -> str:
        # messages es lista OpenAI-format: [{"role": "...", "content": "..."}]
        ...
        return respuesta_str
```

O usar directamente `CallableProvider(mi_funcion)` sin subclasear.

---

## Archivos que no deben modificarse en producción

| Archivo | Razón |
|---------|-------|
| `data/fuentes/mype_distribucion_2023.json` | Inmutable una vez publicado — fuente de verdad estadística. Nuevas versiones = nuevo archivo con año en el nombre. |
| `.sdv_model_mype.pkl` | No versionar — generado localmente. Borrar si cambia el JSON fuente o `calibrator.py`. |

---

## Comandos frecuentes

```bash
# Instalar en modo dev completo
pip install -e ".[dev,agents]"

# Generar 100 perfiles MYPE en CSV
gen-peru-poblacion generate --segmento mype --n 100 --formato csv

# Generar 50 perfiles de Lima en JSONL
gen-peru-poblacion generate --region lima_metropolitana --n 50 --formato jsonl

# Verificar KS similarity del output generado
gen-peru-poblacion verify --data-dir ./output --verbose

# Correr todos los tests
pytest

# Regenerar el modelo SDV (borrar caché primero)
rm -f .sdv_model_mype.pkl && python -c "from gen_peru_poblacion.calibrator import Calibrator; c = Calibrator('mype'); c.fit(use_cache=False)"
```
