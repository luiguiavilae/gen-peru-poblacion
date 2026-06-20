# Política de seguridad y privacidad

## Reportar una vulnerabilidad

Si encuentras una vulnerabilidad de seguridad, **no abras un issue público**.
Repórtala de forma privada mediante uno de estos canales:

- **GitHub Security Advisories**: ve a la pestaña "Security" → "Report a vulnerability".
- **Email**: envía un reporte a la dirección del mantenedor principal (ver perfil de GitHub).

Incluye en el reporte:
- Descripción del problema y su impacto potencial.
- Pasos para reproducirlo.
- Versión de la librería afectada.

Responderemos en un plazo máximo de 72 horas.

## Compromiso de privacidad

Esta librería está diseñada con privacidad por defecto:

- **No procesa datos reales de personas.** La librería genera datos sintéticos, no lee
  ni almacena información personal identificable.

- **Las distribuciones fuente son datos agregados.** Los archivos en `data/fuentes/`
  contienen exclusivamente estadísticas de nivel macro (porcentajes, medianas) citadas
  de fuentes públicas (INEI, PRODUCE, SBS). Ningún valor es trazable a una persona real.

- **Los perfiles generados incluyen `synthetic: true`.** Este campo es obligatorio e
  inmutable en todos los perfiles producidos por la librería. Nunca debe eliminarse
  en pipelines posteriores.

- **No hay telemetría ni tracking.** La librería no realiza llamadas a servicios externos
  salvo las que el usuario configure explícitamente (por ejemplo, una API de LLM con su
  propia API key).

## Versiones con soporte activo

| Versión | Soporte |
|---------|---------|
| 0.1.x   | ✓ Activa |

## Dependencias de terceros

Las vulnerabilidades en dependencias (SDV, LangGraph, openai SDK) deben reportarse
directamente a los respectivos proyectos. Mantenemos `pyproject.toml` actualizado con
rangos de versiones que excluyen releases con CVEs conocidos.
