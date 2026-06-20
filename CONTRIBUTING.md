# Cómo contribuir a gen-peru-poblacion

## Setup de desarrollo

```bash
git clone https://github.com/[tu-usuario]/gen-peru-poblacion
cd gen-peru-poblacion
pip install -e ".[dev,agents]"
```

## Correr tests

```bash
pytest                        # todos los tests
pytest tests/test_generator.py  # solo core
pytest tests/test_agents.py     # solo agentes
pytest --cov --cov-report=html  # con cobertura
```

## Linter y formato

```bash
ruff check .       # linting
ruff check . --fix # auto-fix
black .            # formateo
mypy src/          # type checking
```

## Convención de commits

```
tipo: descripción breve en imperativo

feat:     nueva funcionalidad
fix:      corrección de bug
refactor: refactor sin cambio de comportamiento
test:     agregar o corregir tests
docs:     solo documentación
chore:    tareas de mantenimiento (deps, CI, etc.)
```

## Agregar un nuevo segmento de datos

1. Crear `data/fuentes/{segmento}_distribucion_XXXX.json` con la misma estructura que `mype_distribucion_2023.json`.
   - Incluir `_metadata` con fuente, URL y año.
   - Incluir `correlaciones_conocidas` con coeficientes documentados.
   - Todos los valores deben ser distribuciones agregadas — nunca microdatos individuales.

2. Registrar el segmento en `calibrator.py → SEGMENTO_FUENTE`.

3. Agregar tests en `tests/test_generator.py` con el nuevo segmento.

4. Documentar las fuentes en el PR con cita completa (institución, año, tabla, URL).

## Política de datos

- Ningún archivo en `data/` puede contener microdatos identificables.
- Todas las distribuciones deben tener cita explícita a la fuente primaria.
- Los perfiles generados incluyen siempre `synthetic: true` — nunca se elimina este campo.

## Proceso de PR

1. Abre un issue describiendo el cambio antes de escribir código.
2. Crea una rama desde `main`: `git checkout -b feat/nombre-del-cambio`.
3. Asegúrate de que `pytest` pasa y `ruff check .` no reporta errores.
4. Abre el PR con descripción de qué cambia y por qué.
