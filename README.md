# gen-peru-poblacion

> Synthetic population generator for Peru — calibrated with real INEI, SBS and BCR data.
> Includes conversational agents ready to connect to any LLM.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/gen-peru-poblacion.svg)](https://pypi.org/project/gen-peru-poblacion/)

**Español** | [English](#english-summary)

Generador de poblaciones sintéticas peruanas calibradas con distribuciones reales del
[INEI](https://www.inei.gob.pe), [SBS](https://www.sbs.gob.pe) y
[BCR](https://www.bcrp.gob.pe). Incluye agentes conversacionales funcionales para
simular respuestas de usuarios peruanos en evaluaciones de producto, research de UX
y benchmarking de sistemas de IA.

> **Nota metodológica importante:** Las poblaciones sintéticas complementan la
> investigación de campo — no la reemplazan. El flujo correcto es:
> simulación sintética → validación en campo → decisión. Los datos generados son
> 100% sintéticos; ningún perfil representa a una persona real.

---

## Instalación

```bash
pip install gen-peru-poblacion

# Con agentes conversacionales (requiere LLM):
pip install "gen-peru-poblacion[agents]"    # + LangGraph + DeepSeek
pip install "gen-peru-poblacion[openai]"    # + OpenAI SDK
pip install "gen-peru-poblacion[all]"       # todo incluido
```

---

## Uso rápido

### Generación de perfiles (sin LLM)

```python
from gen_peru_poblacion import Config, PopulationGenerator

config = Config(
    segmento="mype",
    n=100,
    region="lima",          # lima | norte | sur | oriente | centro
    output_dir="./output",
)

gen = PopulationGenerator(config)
gen.run()
# → output/perfiles_mype_lima_100.csv
```

### CLI

```bash
# 100 perfiles de MYPE limeñas
gen-peru-poblacion --segmento mype --region lima --n 100

# 500 perfiles, output JSONL para eval suite
gen-peru-poblacion --segmento mype --n 500 --formato jsonl --output ./data

# Verificar calibración vs distribuciones fuente
gen-peru-poblacion verify --data-dir ./output
```

### Agente conversacional

```python
from gen_peru_poblacion import PopulationGenerator, AgentBuilder

# Genera un perfil
gen = PopulationGenerator(Config(segmento="mype", n=1))
perfil = gen.run()[0]

# Construye un agente desde el perfil
agent = AgentBuilder.from_profile(
    perfil,
    llm_provider="deepseek",   # o "openai", "callable"
)

# Conversa
resp = agent.chat("¿Usas aplicaciones bancarias para tu negocio?")
print(resp.content)

# Múltiples turnos (memoria conversacional)
resp2 = agent.chat("¿Y para recibir pagos de tus clientes?")
print(resp2.content)
```

### Bring your own LLM

```python
def mi_llm(messages, **kwargs):
    # Conecta tu propia API, modelo local, o gateway interno
    return "respuesta del modelo"

agent = AgentBuilder.from_profile(perfil, llm_provider="callable", callable=mi_llm)
```

---

## Segmentos disponibles

| Segmento | Variables clave | Status |
|---|---|---|
| `mype` | Rubro, formalización, crédito, canal de venta, adopción digital | ✅ v1 |
| `consumidores` | Edad, NSE, canal preferido, apps usadas | 🟡 v2 |
| `financiero` | Bancarización, tipo de producto, mora, ahorro | 🟡 v2 |
| `salud` | Acceso, tipo de seguro, zona, lengua | 🔵 planificado |

---

## Schema de perfiles — segmento MYPE

```json
{
  "perfil_id": "pe-mype-001",
  "synthetic": true,
  "data_sources": ["INEI-ENAHO-2023", "SBS-2023", "PRODUCE-2023"],
  "segmento": "mype",
  "region": "lima_norte",
  "distrito": "Los Olivos",
  "rubro": "comercio_minorista",
  "tamaño": "unipersonal",
  "formalizado": false,
  "tiene_ruc": true,
  "empleados": 1,
  "canal_venta": ["fisico", "whatsapp"],
  "credito_formal": false,
  "entidad_credito": "prestamista_informal",
  "monto_credito_soles": 3500,
  "adopcion_digital": "baja",
  "usa_app_bancaria": false,
  "usa_pago_qr": false,
  "lengua_materna": "castellano",
  "edad": 38,
  "nivel_educativo": "secundaria_completa",
  "ingreso_mensual_estimado_soles": 1800
}
```

---

## Eval suite

Las evaluaciones de confiabilidad están en el repositorio separado
[`peru-poblacion-evals`](https://github.com/luigui-dev/peru-poblacion-evals).

```bash
pip install peru-poblacion-evals
peru-evals run --suite mype_v1 --provider deepseek
# → results.tsv con Composite Reliability Score
```

---

## Fuentes de datos

Todas las distribuciones están en [`data/fuentes/`](data/fuentes/) con sus citas:

| Archivo | Fuente | Año | URL |
|---|---|---|---|
| `inei_mype_2023.json` | INEI — ENAHO | 2023 | https://iinei.inei.gob.pe/microdatos/ |
| `sbs_bancarizacion_2023.json` | SBS — Indicadores | 2023 | https://www.sbs.gob.pe/estadisticas |
| `produce_mype_2023.json` | PRODUCE — ENAMIN | 2023 | https://ogeiee.produce.gob.pe |
| `bcr_remesas_2023.json` | BCR — Nota semanal | 2023 | https://www.bcrp.gob.pe/estadisticas |

---

## Calibración y verificación

```bash
gen-peru-poblacion verify --data-dir ./output --verbose
```

Output:
```
Variable          KS statistic    p-value    Status
edad              0.04            0.82       ✅ PASS
ingreso_mensual   0.06            0.61       ✅ PASS
formalizado       0.02            0.95       ✅ PASS
adopcion_digital  0.09            0.38       ✅ PASS
region            0.03            0.88       ✅ PASS
────────────────────────────────────────────────
Composite KS Score: 0.048   Overall: ✅ CALIBRADO
```

---

## Estructura del proyecto

```
gen-peru-poblacion/
├── src/gen_peru_poblacion/
│   ├── __init__.py
│   ├── cli.py              # Entrypoint CLI
│   ├── config.py           # Config dataclass
│   ├── calibrator.py       # SDV GaussianCopulaSynthesizer
│   ├── generator.py        # Pipeline principal
│   ├── agent_builder.py    # Agentes conversacionales
│   ├── providers.py        # LLM providers (deepseek, openai, callable)
│   ├── exporters.py        # CSV / JSON / JSONL
│   └── verify.py           # KS similarity checker
├── data/
│   └── fuentes/            # Distribuciones agregadas con citas
├── tests/
├── examples/
│   ├── basico.py           # Solo generación, sin LLM
│   └── con_agente.py       # Generación + conversación
├── docs/
│   └── SPEC.md
└── pyproject.toml
```

---

## Contribuciones

Las contribuciones son bienvenidas. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

Prioridades para la comunidad:
- Nuevas distribuciones fuente (si tienes acceso a datos INEI / SBS actualizados)
- Nuevos segmentos (consumidores, sector salud, educación)
- Mejoras en calibración regional
- Traducciones de documentación

---

## Limitaciones conocidas

- Los perfiles sintéticos capturan distribuciones estadísticas, no comportamientos individuales reales
- La calibración actual usa datos 2023; las distribuciones se actualizan con cada release mayor
- Los agentes conversacionales pueden generar respuestas inconsistentes en conversaciones
  muy largas (> 20 turnos) — documentado en el eval suite
- No modelamos eventos de vida (desempleo repentino, migración) — el perfil es estático

---

## Cita

```bibtex
@software{gen_peru_poblacion,
  title   = {gen-peru-poblacion: Synthetic Population Generator for Peru},
  author  = {[Tu nombre]},
  year    = {2025},
  url     = {https://github.com/[tu-usuario]/gen-peru-poblacion},
  license = {Apache-2.0}
}
```

---

## English summary

`gen-peru-poblacion` generates synthetic Peruvian user profiles calibrated against
real statistical distributions from Peru's national statistics institute (INEI),
banking regulator (SBS), and central bank (BCR). It includes conversational agents
that simulate responses from Peruvian users for product evaluation, UX research,
and AI benchmarking — without using real customer data.

**License:** Apache 2.0 — free for commercial and academic use.

---

*Hecho en Perú 🇵🇪 — datos sintéticos, propósito real.*
