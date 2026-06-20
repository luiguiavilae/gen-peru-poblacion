# Checklist pre-construcción — gen-peru-poblacion

Valida cada ítem antes de pasarle el PROMPT_CLAUDE_CODE.md a Claude Code.

---

## Claridad del spec

- [x] El problema está en términos del usuario, no del sistema
- [x] Cada caso de uso tiene criterios de aceptación verificables (DADO/CUANDO/ENTONCES)
- [x] El "fuera de alcance" está explícito (API REST, otros segmentos, fine-tuning)
- [x] El happy path está claro: `Config → PopulationGenerator → AgentBuilder → chat()`

## Completitud técnica

- [x] Stack definido (SDV, LangGraph, DeepSeek, Typer, pyproject.toml/hatchling)
- [x] Datos fuente existen en `data/fuentes/mype_distribucion_2023.json`
- [x] Separación core/agentes está especificada (core funciona offline)
- [x] Schema de output definido con todos los campos y tipos
- [x] Métricas de verificación definidas (KS similarity ≥ 0.70)
- [x] Eval suite existe en repo hermano (`peru-poblacion-evals/evals/mype_v1.jsonl`)

## Alineación estratégica

- [x] Resuelve el problema declarado (no hay feature creep en v1)
- [x] Alcance ejecutable: un desarrollador puede tener v0.1.0 en 1-2 semanas
- [x] Métricas de éxito son medibles (descargas PyPI, stars, KS score)
- [x] Posicionamiento claro: "primer generador de población sintética peruana"

## Decisiones tomadas (no abrir en Claude Code)

- [x] Dos repos separados (generador + eval suite)
- [x] DeepSeek default, callable pattern para otros LLMs
- [x] JSON estático en repo (no base de datos)
- [x] Apache 2.0
- [x] Segmento ancla: MYPE

## Pendientes ANTES de publicar en PyPI

- [ ] Verificar disponibilidad del nombre `gen-peru-poblacion` en PyPI
- [ ] Definir usuario/organización GitHub
- [ ] Resolución preguntas abiertas del SPEC:
      → ¿30 o 50 preguntas en eval suite v1? (recomendado: 20 para v1, expandir en v1.1)
      → ¿CRS incluye coherencia inter-turno? (respuesta: sí, tipo multi_turno en jsonl)

## Orden de publicación recomendado

1. Publicar `gen-peru-poblacion` en GitHub (privado mientras se construye)
2. Testear con `pip install -e .` localmente
3. Correr eval suite básico con 10 perfiles
4. Publicar `peru-poblacion-evals` en GitHub
5. Publicar `gen-peru-poblacion` en PyPI (primera versión pública)
6. Post en LinkedIn / comunidades de data science peruanas

---

**Estado:** Listo para pasar a Claude Code. ✅
