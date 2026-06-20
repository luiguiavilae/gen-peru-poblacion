# gen-peru-poblacion

## Resumen

Librería Python open source que genera poblaciones sintéticas de peruanos calibradas
con datos reales del INEI, SBS y BCR, e incluye agentes conversacionales funcionales
listos para conectar a cualquier LLM — permitiendo a equipos de producto, UX research,
data science y empresas simular respuestas de usuarios peruanos sin datos reales.

---

## Problema

No existe una infraestructura pública, calibrada y reutilizable para simular el
comportamiento de usuarios peruanos. Quienes hacen research o evalúan productos
en Perú dependen de fieldwork costoso y lento, o de personas sintéticas construidas
desde supuestos sin respaldo estadístico. Los datasets sintéticos globales (ej.
Faker, SDV defaults) no reflejan las particularidades del mercado peruano:
distribución geográfica real, niveles de bancarización por departamento, prevalencia
de negocio informal, patrones de remesas, lengua materna, etc.

---

## Usuarios objetivo

**Primario — equipos técnicos:**
Desarrolladores, data scientists y UX researchers en startups, fintechs, empresas
de retail o salud que necesitan simular respuestas de usuarios peruanos para evaluar
productos, calibrar modelos o entrenar sistemas antes de ir a campo.

**Secundario — comunidad académica:**
Investigadores y estudiantes universitarios peruanos que necesitan datasets sintéticos
reproducibles para sus estudios sin acceso a datos reales de clientes.

**Terciario — equipos de diseño y producto sin perfil técnico:**
Quienes consumen el eval suite como referencia comparativa sin instalar la librería.

---

## Segmento ancla — v1

**Emprendedores y MYPE peruanas.** Variables clave:
- Rubro del negocio (comercio, servicios, manufactura, agro)
- Tamaño (unipersonal, familiar, 2-10 empleados)
- Nivel de formalización (informal, RUC activo, en SUNAT)
- Región (Lima Metropolitana, ciudades intermedias, zonas rurales)
- Acceso a crédito formal (Mibanco, BCP, caja municipal, prestamista informal)
- Canal de venta (físico, WhatsApp, marketplace, mixto)
- Nivel de adopción digital (apps bancarias, pagos QR, facturación electrónica)
- Lengua (castellano, quechua, bilingüe)

---

## Casos de uso principales

**CU-01 — Generación de perfiles:**
Como data scientist, quiero generar N perfiles sintéticos de emprendedores peruanos
con distribuciones estadísticamente coherentes para calibrar mis modelos sin datos reales.

**CU-02 — Agente conversacional:**
Como UX researcher, quiero hacer preguntas abiertas a un agente que simula un emprendedor
peruano específico para obtener respuestas cualitativas antes de salir a campo.

**CU-03 — Eval suite:**
Como equipo de producto, quiero correr un conjunto fijo de preguntas contra mi integración
de agentes y obtener un score de confiabilidad comparativo (Composite Reliability Score).

**CU-04 — CLI rápido:**
Como desarrollador, quiero generar perfiles desde la terminal con un comando simple para
integrarlos en mis pipelines existentes sin escribir Python.

**CU-05 — Exportación flexible:**
Como analista, quiero exportar perfiles en CSV, JSON y JSONL para alimentar distintos
sistemas sin transformaciones adicionales.

---

## Criterios de aceptación

### CU-01 — Generación de perfiles
- DADO un comando `gen-peru-poblacion --segmento mype --n 100`
- CUANDO se ejecuta
- ENTONCES se genera un archivo con 100 perfiles con todas las variables del segmento MYPE,
  distribuciones coherentes con datos INEI/SBS, sin duplicados exactos, en < 10 segundos.
- ENTONCES el KS similarity entre la muestra generada y las distribuciones fuente es ≥ 0.70
  para las 5 variables principales.

### CU-02 — Agente conversacional
- DADO un perfil sintético cargado y una pregunta en español
- CUANDO se llama a `agent.chat("¿Usas aplicaciones bancarias para tu negocio?")`
- ENTONCES la respuesta es consistente con las características del perfil (región, nivel
  de adopción digital, rubro) y no contradice variables conocidas del perfil.
- ENTONCES el agente mantiene coherencia entre turnos dentro de la misma sesión.

### CU-03 — Eval suite
- DADO un integrador que corre `peru-poblacion-evals run --suite mype_v1`
- CUANDO el suite termina
- ENTONCES se produce un `results.tsv` con score por pregunta y Composite Reliability Score.
- ENTONCES el score es reproducible: dos ejecuciones con el mismo perfil y misma pregunta
  producen scores dentro de ±5 puntos porcentuales.

### CU-04 — CLI
- DADO `pip install gen-peru-poblacion`
- CUANDO se corre `gen-peru-poblacion --help`
- ENTONCES todos los flags están documentados con ejemplos.

### CU-05 — Exportación
- DADO N perfiles generados
- CUANDO se especifica `--formato csv|json|jsonl`
- ENTONCES el archivo resultante es válido, parseable, con schema documentado.

---

## Fuera de alcance — v1

- API REST pública (Canal B — se decide después de tracción en PyPI)
- Segmentos distintos a MYPE (consumidores digitales, salud, etc. son v2+)
- Fine-tuning de modelos
- Interfaz web / dashboard
- Autenticación / rate limiting
- Datos de personas reales identificables (nunca, en ninguna versión)
- Integración con Copilot Studio o Power Automate (se documenta cómo hacerlo, no se construye)
- Soporte a idiomas distintos del español en la interfaz (el agente puede responder en quechua
  si el perfil lo indica, pero la CLI y docs son en español e inglés)

---

## Stack y restricciones técnicas

**Lenguaje:** Python 3.10+
**Generación sintética:** SDV (GaussianCopulaSynthesizer) — ya validado en PescAI
**Agentes:** LangGraph para orquestación, DeepSeek como LLM primario
**LLM bridge:** compatible con `llm_bridge` pattern (callable provider) — cualquier LLM puede sustituir DeepSeek
**CLI:** Click o Typer
**Exportación:** Pandas, json stdlib
**CI/CD:** GitHub Actions (ci.yml, codeql.yml, dependabot)
**Packaging:** pyproject.toml (setuptools o hatchling)
**Licencia:** Apache 2.0
**Datos fuente:** JSON estático en `/data/fuentes/` — distribuciones agregadas, nunca microdatos

**Patrones a respetar (del análisis de SantanderAI):**
- Separación limpia: `generator.py` (pipeline) ← `calibrator.py` (SDV) + `agent_builder.py` (LLM)
- `verify.py` independiente — corre KS similarity sin tocar el pipeline
- Providers pluggables para LLM: quien no tiene DeepSeek pone su propio callable
- `eval_suite.jsonl` es inmutable una vez publicado — nuevas versiones = nuevo archivo

**Constraints de datos:**
- Todas las distribuciones fuente deben tener cita explícita (INEI año, tabla, URL)
- Ningún campo puede ser trazable a una persona real
- Los perfiles generados incluyen `synthetic: true` y `data_sources: [...]` en el schema

---

## Métricas de éxito

**Técnicas:**
- KS similarity ≥ 0.70 en variables principales vs distribuciones INEI/SBS
- Tiempo de generación: 1000 perfiles en < 30 segundos en hardware consumer
- 0 dependencias obligatorias de LLM (agentes son opcionales, core funciona offline)

**De adopción (30/60/90 días post-publicación):**
- 30 días: ≥ 50 descargas en PyPI, ≥ 3 stars en GitHub
- 60 días: primera issue o PR externa
- 90 días: citado o referenciado en al menos un proyecto / tesis peruana

**De posicionamiento:**
- Aparecer en búsquedas "datos sintéticos Perú", "usuarios sintéticos Perú" en GitHub
- Al menos una mención en comunidad de data science peruana (meetup, post, LinkedIn)

---

## Decisiones de arquitectura tomadas

**ADR-01:** Dos repositorios separados (`gen-peru-poblacion` y `peru-poblacion-evals`),
igual que el patrón Santander (generador vs evaluador). El eval suite tiene su propio
contrato de inmutabilidad.

**ADR-02:** DeepSeek como LLM default por costo y acceso en Perú, pero el agente
acepta cualquier callable — compatible con OpenAI, Anthropic, Groq, modelo local.

**ADR-03:** Datos fuente como JSON estático en el repo — sin base de datos. Reproducible,
auditable, versionado con git.

**ADR-04:** KS similarity como métrica primaria de calibración, reportada por `verify.py`
en cada generación.

**ADR-05:** Eval suite v1 anclado a MYPE. Versiones futuras son archivos separados
(`mype_v1.jsonl`, `mype_v2.jsonl`) — nunca se modifica un suite publicado.

---

## Preguntas abiertas resueltas

- ¿Nombre del paquete PyPI? → `gen-peru-poblacion` (verificar disponibilidad antes de publicar)
- ¿Licencia? → Apache 2.0 (igual que SantanderAI, permite uso comercial)
- ¿Organización GitHub? → personal primero, organización cuando haya colaboradores
- ¿Qué pasa si DeepSeek no está disponible? → el core (generación de perfiles) funciona
  offline. Los agentes requieren LLM pero el provider es configurable.

## Preguntas abiertas pendientes

- ¿Cuántas preguntas tendrá el eval suite v1? (recomendación: 30-50 preguntas fijas)
- ¿El score de confiabilidad incluye dimensión de coherencia inter-turno o solo respuesta única?
- ¿Se publica el paper / nota metodológica junto con el repo o después?
