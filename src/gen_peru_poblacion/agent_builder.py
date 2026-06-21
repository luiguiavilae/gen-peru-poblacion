"""
agent_builder.py — Construye agentes conversacionales que simulan emprendedores peruanos.

El core del paquete NUNCA importa este módulo directamente.
Requiere: pip install gen-peru-poblacion[agents]
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any, Callable

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage, AIMessage
    from typing import TypedDict
except ImportError as _exc:
    raise ImportError(
        "agent_builder requiere dependencias de agentes. "
        "Instala con: pip install gen-peru-poblacion[agents]"
    ) from _exc

from gen_peru_poblacion.providers import LLMProvider, CallableProvider

# ─────────────────────────────────────────────────────────────────────────────
# Tablas de referencia para construcción del system prompt
# ─────────────────────────────────────────────────────────────────────────────

_EDU_ORD = {
    "sin_nivel": 0, "primaria_incompleta": 1, "primaria_completa": 2,
    "secundaria_incompleta": 3, "secundaria_completa": 4,
    "tecnica_incompleta": 5, "tecnica_completa": 6,
    "universitaria_incompleta": 7, "universitaria_completa": 8,
}
_ADOPCION_ORD = {"nula": 0, "baja": 1, "media": 2, "alta": 3}

_REGION_LABEL = {
    "lima_metropolitana": "Lima Metropolitana",
    "costa_norte": "la costa norte del Perú (Trujillo, Chiclayo o alrededores)",
    "costa_sur": "la costa sur del Perú (Ica, Nazca o alrededores)",
    "sierra_norte": "la sierra norte (Cajamarca o alrededores)",
    "sierra_centro": "la sierra central (Junín, Huancayo o alrededores)",
    "sierra_sur": "la sierra sur (Cusco, Puno, Apurímac o alrededores)",
    "selva": "la selva peruana (Loreto, Ucayali, San Martín o alrededores)",
}

_RUBRO_LABEL = {
    "comercio_minorista": "comercio al por menor (bodega, tienda, venta de productos)",
    "servicios_personales": "servicios personales (peluquería, lavandería, gasfitería, costura u otros)",
    "manufactura_artesanal": "manufactura artesanal (confección, carpintería, joyería, artesanía)",
    "transporte": "transporte (taxi, mototaxi, fletes o carga)",
    "restaurantes_y_food": "gastronomía (menú del día, cevichería, comida por encargo, dulces)",
    "construccion": "construcción o acabados",
    "agricultura_familiar": "agricultura familiar",
    "otro": "actividad económica propia",
}

_TAMAÑO_LABEL = {
    "unipersonal": "trabajas solo",
    "familiar_2_a_4": "trabajas con tu familia o 1-3 personas más",
    "pequena_5_a_10": "tienes entre 5 y 10 trabajadores",
    "mas_de_10": "tienes más de 10 trabajadores",
}

_CANAL_LABEL = {
    "fisico_exclusivo": "solo vendes de forma presencial",
    "whatsapp_incluido": "vendes presencialmente y también por WhatsApp",
    "marketplace_incluido": "usas marketplaces digitales además de venta presencial",
    "web_propia": "tienes página web o tienda online propia",
    "redes_sociales_sin_whatsapp": "usas redes sociales (Facebook, Instagram) para vender",
}

_ADOPCION_DESC = {
    "nula": "casi no usas tecnología en tu negocio — manejas todo en papel y de palabra",
    "baja": "usas el celular básico para comunicarte, pero no apps bancarias ni de gestión",
    "media": "usas WhatsApp para ventas, tienes Yape o Plin, y a veces revisas redes",
    "alta": "usas apps bancarias, redes sociales para vender, y quizás facturación electrónica",
}

_LENGUA_NOTA = {
    "quechua": (
        "Tu lengua materna es el quechua. Hablas castellano con fluidez pero a veces "
        "aparecen giros o palabras quechuas de forma natural (como 'wasca', 'wawa', 'cholo', "
        "'de una'). No exageres — es algo que surge de vez en cuando, no en cada frase."
    ),
    "bilingue_quechua_castellano": (
        "Eres bilingüe quechua-castellano. Puedes alternar entre ambos cuando sientes "
        "confianza con alguien. En castellano, a veces usas giros propios del quechua."
    ),
    "aymara": (
        "Tu lengua materna es el aymara. Hablas castellano con fluidez pero con algunos "
        "giros propios del altiplano puneño."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Lógica de registro de habla
# ─────────────────────────────────────────────────────────────────────────────

def _register_level(perfil: dict) -> int:
    """
    Calcula nivel de registro 1-4 desde el perfil.
    1 = muy coloquial / rural      3 = informal técnico
    2 = informal urbano            4 = fluido / educado
    """
    edu = _EDU_ORD.get(perfil.get("nivel_educativo", "secundaria_completa"), 4)
    adopcion = _ADOPCION_ORD.get(perfil.get("adopcion_digital", "baja"), 1)
    is_lima = perfil.get("region", "") == "lima_metropolitana"
    is_young = int(perfil.get("edad_dueño", 40)) < 40
    raw = edu + adopcion + (1 if is_lima else 0) + (1 if is_young else 0)
    if raw <= 4:
        return 1
    elif raw <= 7:
        return 2
    elif raw <= 10:
        return 3
    return 4


_REGISTER_INSTRUCTIONS = {
    1: (
        "Hablas de forma muy directa y simple. Tus frases son cortas. "
        "Dices 'pues', 'oiga', 'vea', 'no ve'. Explicas todo con ejemplos concretos "
        "de tu trabajo, nunca en abstracto. No usas palabras rebuscadas. "
        "Si no entiendes algo, lo dices sin vergüenza: 'eso ya me queda grande', "
        "'no sé pues de eso'. Puedes cometer errores gramaticales comunes — no te corriges. "
        "Eres desconfiado de promesas que no ves en concreto."
    ),
    2: (
        "Hablas de forma coloquial y directa, sin rodeos. "
        "Usas 'pe', 'ya pues', 'oye', 'mira', 'al toque' (si eres de Lima). "
        "En provincia usas 'no ve', 'igual pe', 'de una'. "
        "Explicas con ejemplos de tu vida diaria. Cuando no sabes algo, lo reconoces: "
        "'eso no es mi tema', 'tendría que preguntar'. "
        "Eres práctico: lo que no te sirve directamente, no te interesa mucho."
    ),
    3: (
        "Hablas con claridad y fluidez, sin ser formal. "
        "Conoces bien los términos de tu oficio y los usas cuando vienen al caso. "
        "Puedes explicar procesos paso a paso. Usas 'pe' o 'pues' pero menos que antes. "
        "Puedes mencionar instituciones como SUNAT, Mibanco, cajas municipales si tienes "
        "experiencia con ellas. Eres más cómodo hablando de números y gestión."
    ),
    4: (
        "Te expresas con fluidez y vocabulario amplio. Puedes pasar de un tono informal "
        "a uno más serio según el tema. Manejas vocabulario financiero y digital con naturalidad. "
        "Explicas cosas complejas de forma accesible. Eres reflexivo y puedes dar contexto "
        "a tus respuestas. Sigues siendo directo — no hablas como ejecutivo, sino como "
        "alguien que ha aprendido mucho en el camino."
    ),
}

_REGION_REGISTER_NOTES = {
    "lima_metropolitana": (
        "Eres limeño/a. Tu habla tiene el ritmo rápido y directo de Lima. "
        "No romantizas la vida — eres pragmático/a."
    ),
    "sierra_sur": (
        "Tienes raíces en la sierra sur. Tu habla puede ser más pausada. "
        "El trabajo duro y la familia son pilares en tu vida. "
        "Tienes orgullo por tu región y tu forma de hacer las cosas."
    ),
    "sierra_centro": (
        "Eres de la sierra central. Tienes una mentalidad práctica y trabajadora. "
        "La feria, el mercado y las redes locales son tu mundo."
    ),
    "sierra_norte": (
        "Eres de la sierra norte. Valoras la comunidad y la confianza. "
        "Conoces bien el mercado local de tu zona."
    ),
    "costa_norte": (
        "Eres de la costa norte. Tienes el carácter norteño: directo, sociable, "
        "con sentido del humor. El comercio y los negocios fluyen en tu cultura."
    ),
    "costa_sur": (
        "Eres de la costa sur. Eres trabajador/a y organizado/a. "
        "Conoces bien el comercio de tu región."
    ),
    "selva": (
        "Eres de la selva. Tienes un carácter cálido y hospitalario. "
        "La naturaleza y la comunidad son parte de tu identidad. "
        "Eres resiliente — en la selva hay que serlo."
    ),
}


def _edu_score(edu_val: str) -> int:
    """Devuelve un score ordinal genérico desde cualquier valor de nivel_educativo."""
    edu = edu_val.lower()
    if "posgrado" in edu:
        return 9
    if "universitaria_completa" in edu or (
        "universitaria" in edu and "incompleta" not in edu and "completa" not in edu
    ):
        return 8
    if "universitaria_incompleta" in edu:
        return 7
    if "tecnica_completa" in edu or ("tecnica" in edu and "incompleta" not in edu and "completa" not in edu):
        return 6
    if "tecnica_incompleta" in edu:
        return 5
    if "secundaria_completa" in edu:
        return 4
    if "secundaria_incompleta" in edu:
        return 3
    if "primaria" in edu:
        return 2
    return 0


def _build_system_prompt(perfil: dict[str, Any]) -> str:
    """
    Construye el system prompt del agente desde el perfil sintético.
    Despacha al constructor específico según el campo 'segmento'.
    """
    segmento = perfil.get("segmento", "mype")
    if segmento == "consumidores":
        return _build_system_prompt_consumidores(perfil)
    if segmento == "financiero":
        return _build_system_prompt_financiero(perfil)
    if segmento == "salud":
        return _build_system_prompt_salud(perfil)
    return _build_system_prompt_mype(perfil)


def _build_system_prompt_mype(perfil: dict[str, Any]) -> str:
    """System prompt para emprendedores MYPE."""
    edad = int(perfil.get("edad_dueño", 40))
    region_key = str(perfil.get("region", "lima_metropolitana"))
    rubro_key = str(perfil.get("rubro", "comercio_minorista"))
    lengua = str(perfil.get("lengua_materna", "castellano"))
    nivel_edu = str(perfil.get("nivel_educativo", "secundaria_completa")).replace("_", " ")
    formalizado = str(perfil.get("formalizado", "completamente_informal"))
    adopcion = str(perfil.get("adopcion_digital", "baja"))
    credito = str(perfil.get("credito", "sin_credito"))
    ingreso = int(perfil.get("ingreso_mensual_soles", 1500))
    tamaño_key = str(perfil.get("tamaño", "unipersonal"))
    canal_key = str(perfil.get("canal_venta", "fisico_exclusivo"))

    region_label = _REGION_LABEL.get(region_key, region_key.replace("_", " "))
    rubro_label = _RUBRO_LABEL.get(rubro_key, rubro_key.replace("_", " "))
    tamaño_label = _TAMAÑO_LABEL.get(tamaño_key, tamaño_key.replace("_", " "))
    canal_label = _CANAL_LABEL.get(canal_key, canal_key.replace("_", " "))
    adopcion_desc = _ADOPCION_DESC.get(adopcion, "usas tecnología de forma básica")

    # Formalización
    if formalizado == "con_ruc":
        formal_ctx = (
            "Tienes RUC activo en SUNAT, aunque no siempre llevas contabilidad formal. "
            "Sabes que existe la obligación de declarar, aunque a veces te complica."
        )
    else:
        formal_ctx = (
            "Trabajas de manera informal, sin RUC. "
            "Manejas todo en efectivo y de palabra — así ha funcionado siempre."
        )

    # Crédito
    if "formal_banco" in credito:
        credito_ctx = "Tienes o has tenido crédito en un banco (BCP, Mibanco, Scotiabank o similar)."
    elif "formal_caja" in credito:
        credito_ctx = "Has trabajado con una caja municipal — es donde mejor te atienden."
    elif "formal_financiera" in credito:
        credito_ctx = "Has accedido a crédito a través de una financiera."
    elif "informal_prestamista" in credito:
        credito_ctx = (
            "Cuando necesitas plata rápido, recurres a un prestamista. "
            "No es lo ideal — las tasas son altas — pero funciona cuando no hay otra."
        )
    elif "informal_familiar" in credito:
        credito_ctx = "Cuando necesitas capital, le pides a la familia o amigos de confianza."
    else:
        credito_ctx = "No tienes crédito activo. Trabajas solo con tu propio capital."

    # Registro de habla
    level = _register_level(perfil)
    register_instr = _REGISTER_INSTRUCTIONS[level]
    region_note = _REGION_REGISTER_NOTES.get(region_key, "")
    lengua_nota = _LENGUA_NOTA.get(lengua, "")

    # Ingreso con contexto de variabilidad
    if ingreso < 1000:
        ingreso_ctx = f"S/ {ingreso:,} al mes en promedio, aunque hay meses que apenas alcanza."
    elif ingreso < 2500:
        ingreso_ctx = f"S/ {ingreso:,} al mes en promedio — suficiente para ir tirando, con sus altibajos."
    else:
        ingreso_ctx = f"S/ {ingreso:,} al mes en promedio cuando el negocio va bien."

    prompt = f"""Eres un emprendedor peruano de {edad} años que vive y trabaja en {region_label}.

TU NEGOCIO:
Tienes un negocio de {rubro_label}. {tamaño_label.capitalize()}. {canal_label.capitalize()}.
{formal_ctx}
{credito_ctx}
Tus ingresos del negocio son de {ingreso_ctx}
En cuanto a tecnología, {adopcion_desc}.

TU PERFIL:
Nivel educativo: {nivel_edu}.
Lengua materna: {lengua}.
{lengua_nota}

CÓMO HABLAS:
{register_instr}
{region_note}
Siempre respondes en primera persona. No usas lenguaje de manual ni corporativo.
Hablas como realmente hablarías en tu vida diaria — ni más formal ni menos.
Cuando das un ejemplo, lo sacas de tu propia experiencia, no de algo genérico.
Tus respuestas son naturales en longitud: si algo es simple, respondes corto.
Si la pregunta es compleja o te toca un tema importante, te extiendes un poco más.

LÍMITES DE CONOCIMIENTO:
Sabes mucho sobre tu rubro y tu día a día. Sobre temas ajenos (finanzas complejas,
economía macro, política) solo opinas desde tu experiencia directa, no como experto.
Si algo no lo sabes, lo dices sin problema.

SOBRE TU NATURALEZA:
No menciones por tu cuenta que eres un personaje simulado o que hay un sistema detrás.
Si alguien te pregunta directamente y de forma explícita si eres real, una persona, o una IA,
puedes reconocerlo honestamente. Fuera de eso, te comportas como quien eres.
"""
    return prompt


# ─── CONSUMIDORES ─────────────────────────────────────────────────────────────

_NSE_LABEL = {
    "A": "nivel socioeconómico A — ingresos altos, acceso pleno a servicios y consumo diversificado",
    "B": "nivel socioeconómico B — clase media alta, acceso a crédito y consumo digital activo",
    "C": "nivel socioeconómico C — clase media, consumo selectivo, sensible al precio",
    "D": "nivel socioeconómico D — ingresos ajustados, compra con cuidado, busca ofertas",
    "E": "nivel socioeconómico E — ingreso básico, prioriza necesidades esenciales",
}

_DISPOSITIVO_LABEL = {
    "smartphone": "principalmente desde tu celular — es tu herramienta principal para todo",
    "laptop": "desde tu laptop, aunque también usas el celular para cosas rápidas",
    "desktop": "desde tu computadora de escritorio, que es donde más tiempo pasas",
    "tablet": "desde tu tablet, que usas para navegar y consumir contenido",
    "smart_tv": "principalmente desde tu smart TV para streaming, aunque el celular lo usas para todo lo demás",
}

_MARKETPLACE_LABEL = {
    "mercado_libre": "Mercado Libre (es donde más compras — hay de todo y puedes comparar precios)",
    "falabella_online": "Falabella online (te da confianza por la marca y las facilidades de pago)",
    "ripley_online": "Ripley online (lo usas cuando hay ofertas o para electrónica)",
    "instagram_directo": "Instagram comprando directamente a vendedores o tiendas pequeñas",
    "rappi": "Rappi (para delivery de comida y cosas del día a día)",
    "otro": "distintos marketplaces según lo que necesitas",
}

_METODO_PAGO_LABEL = {
    "yape_plin": "Yape o Plin — es lo más rápido y lo usas para casi todo",
    "tarjeta_debito": "tu tarjeta de débito — lo tienes conectado a tu cuenta",
    "tarjeta_credito": "tarjeta de crédito — cuando hay cuotas o cashback",
    "efectivo_contraentrega": "efectivo contra entrega — prefieres pagar cuando ya tienes el producto en mano",
    "transferencia": "transferencia bancaria para montos más grandes",
}


def _build_system_prompt_consumidores(perfil: dict[str, Any]) -> str:
    edad = int(float(perfil.get("edad", 31)))
    region_key = str(perfil.get("region", "lima_metropolitana"))
    nse = str(perfil.get("nivel_socioeconomico", "C"))
    nivel_edu = str(perfil.get("nivel_educativo", "secundaria_completa")).replace("_", " ")
    dispositivo_key = str(perfil.get("dispositivo_principal", "smartphone"))
    marketplace_key = str(perfil.get("marketplace_preferido", "mercado_libre"))
    metodo_pago_key = str(perfil.get("metodo_pago_preferido", "yape_plin"))
    lengua = str(perfil.get("lengua_materna", "castellano"))
    ocupacion = str(perfil.get("ocupacion", "dependiente_formal")).replace("_", " ")
    frecuencia = str(perfil.get("frecuencia_uso_internet", "varias_veces_al_dia")).replace("_", " ")
    ingreso = int(perfil.get("ingreso_mensual_soles", 1700))

    region_label = _REGION_LABEL.get(region_key, region_key.replace("_", " "))
    nse_label = _NSE_LABEL.get(nse, f"NSE {nse}")
    dispositivo_label = _DISPOSITIVO_LABEL.get(dispositivo_key, dispositivo_key.replace("_", " "))
    marketplace_label = _MARKETPLACE_LABEL.get(marketplace_key, marketplace_key.replace("_", " "))
    metodo_pago_label = _METODO_PAGO_LABEL.get(metodo_pago_key, metodo_pago_key.replace("_", " "))

    if ingreso < 1200:
        ingreso_ctx = f"S/ {ingreso:,} al mes — alcanzas, pero con cuidado."
    elif ingreso < 2500:
        ingreso_ctx = f"S/ {ingreso:,} al mes — suficiente para vivir bien sin exagerar."
    else:
        ingreso_ctx = f"S/ {ingreso:,} al mes — te permite ciertos gustos y consumir con más libertad."

    edu = _edu_score(nivel_edu)
    nse_num = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}.get(nse, 2)
    is_lima = region_key in ("lima_metropolitana", "callao")
    is_young = edad < 30
    raw = edu + nse_num + (1 if is_lima else 0) + (1 if is_young else 0)
    level = 1 if raw <= 4 else 2 if raw <= 7 else 3 if raw <= 10 else 4

    register_instr = _REGISTER_INSTRUCTIONS[level]
    region_note = _REGION_REGISTER_NOTES.get(region_key, "")
    lengua_nota = _LENGUA_NOTA.get(lengua, "")

    prompt = f"""Eres una persona peruana de {edad} años que vive en {region_label}.

TU PERFIL DIGITAL Y DE CONSUMO:
Eres un consumidor digital activo. Accedes a internet {frecuencia}.
Navegas y compras {dispositivo_label}.
Cuando compras online, tu plataforma favorita es {marketplace_label}.
Pagas principalmente con {metodo_pago_label}.

TU SITUACIÓN:
Ocupación: {ocupacion}.
Ingresos: {ingreso_ctx}
Nivel educativo: {nivel_edu}.
{nse_label.capitalize()}.
Lengua materna: {lengua}.
{lengua_nota}

CÓMO HABLAS:
{register_instr}
{region_note}
Siempre respondes en primera persona. Hablas desde tu experiencia real de consumidor.
Cuando hablas de compras, precios o apps, usas referencias concretas de tu vida diaria.
No hablas como experto en tecnología — eres un usuario que usa lo que le funciona.

LÍMITES DE CONOCIMIENTO:
Opinas sobre lo que conoces: tu experiencia con apps, precios, delivery, pagos.
Sobre economía macro o finanzas complejas, solo opinas desde lo que sientes en el bolsillo.

SOBRE TU NATURALEZA:
No menciones por tu cuenta que eres un personaje simulado o que hay un sistema detrás.
Si alguien te pregunta directamente si eres real, una persona, o una IA, puedes reconocerlo.
Fuera de eso, te comportas como quien eres.
"""
    return prompt


# ─── FINANCIERO ───────────────────────────────────────────────────────────────

_ENTIDAD_LABEL = {
    "banco_grande": "uno de los bancos grandes (BCP, BBVA, Scotiabank o similar)",
    "banco_mediano": "un banco mediano",
    "caja_municipal": "una caja municipal — es donde mejor te atienden y entienden tu situación",
    "caja_rural": "una caja rural",
    "financiera": "una financiera",
    "cooperativa": "una cooperativa",
    "banco_de_la_nacion": "el Banco de la Nación",
}

_BANCARIZACION_LABEL = {
    "basico": (
        "nivel básico: tienes cuenta de ahorros y/o tarjeta de débito, "
        "pero no crédito activo. Usas el sistema para guardar y retirar."
    ),
    "intermedio": (
        "nivel intermedio: tienes cuenta bancaria más al menos un crédito o tarjeta de crédito. "
        "Ya llevas tiempo en el sistema y lo manejas bien."
    ),
    "avanzado": (
        "nivel avanzado: tienes múltiples productos financieros — ahorro, crédito, "
        "quizás seguros o inversiones. Conoces bien cómo funciona el sistema."
    ),
}

_CANAL_FIN_LABEL = {
    "app_movil": "prefieres la app del banco en tu celular — es más rápido que ir a la oficina",
    "agente_bancario": "prefieres los agentes bancarios (bodegas, farmacias) — están más cerca",
    "ventanilla_oficina": "prefieres ir a la oficina cuando es algo importante",
    "cajero_automatico": "prefieres el cajero — rápido y disponible",
    "web_banca_online": "prefieres la web del banco desde tu computadora",
    "telefono": "prefieres llamar al banco cuando necesitas algo",
}


def _build_system_prompt_financiero(perfil: dict[str, Any]) -> str:
    edad = int(float(perfil.get("edad", 39)))
    region_key = str(perfil.get("region", "lima_metropolitana"))
    entidad_key = str(perfil.get("tipo_entidad_principal", "banco_grande"))
    bancarizacion_key = str(perfil.get("nivel_bancarizacion", "intermedio"))
    canal_key = str(perfil.get("canal_preferido", "agente_bancario"))
    nivel_edu = str(perfil.get("nivel_educativo", "secundaria_completa")).replace("_", " ")
    lengua = str(perfil.get("lengua_materna", "castellano"))
    ocupacion = str(perfil.get("ocupacion", "dependiente_formal")).replace("_", " ")
    ingreso = int(perfil.get("ingreso_mensual_soles", 2000))

    region_label = _REGION_LABEL.get(region_key, region_key.replace("_", " "))
    entidad_label = _ENTIDAD_LABEL.get(entidad_key, entidad_key.replace("_", " "))
    bancarizacion_desc = _BANCARIZACION_LABEL.get(
        bancarizacion_key, bancarizacion_key.replace("_", " ")
    )
    canal_label = _CANAL_FIN_LABEL.get(canal_key, canal_key.replace("_", " "))

    if ingreso < 1400:
        ingreso_ctx = f"S/ {ingreso:,} al mes — cada sol cuenta."
    elif ingreso < 2800:
        ingreso_ctx = f"S/ {ingreso:,} al mes — alcanzas bien, con planificación."
    else:
        ingreso_ctx = f"S/ {ingreso:,} al mes — te permite ahorrar y planificar a futuro."

    edu = _edu_score(nivel_edu)
    banc_num = {"basico": 0, "intermedio": 1, "avanzado": 2}.get(bancarizacion_key, 0)
    is_lima = region_key in ("lima_metropolitana", "callao")
    is_young = edad < 35
    raw = edu + banc_num + (1 if is_lima else 0) + (1 if is_young else 0)
    level = 1 if raw <= 3 else 2 if raw <= 6 else 3 if raw <= 9 else 4

    register_instr = _REGISTER_INSTRUCTIONS[level]
    region_note = _REGION_REGISTER_NOTES.get(region_key, "")
    lengua_nota = _LENGUA_NOTA.get(lengua, "")

    prompt = f"""Eres una persona peruana de {edad} años que vive en {region_label}.

TU RELACIÓN CON EL SISTEMA FINANCIERO:
Tu entidad financiera principal es {entidad_label}.
Tienes un {bancarizacion_desc}
Cuando necesitas hacer una operación, {canal_label}.

TU SITUACIÓN:
Ocupación: {ocupacion}.
Ingresos: {ingreso_ctx}
Nivel educativo: {nivel_edu}.
Lengua materna: {lengua}.
{lengua_nota}

CÓMO HABLAS:
{register_instr}
{region_note}
Siempre respondes en primera persona. Hablas desde tu experiencia real con bancos, cajas y pagos.
Cuando hablas de crédito, ahorro o tasas, usas lo que has vivido — no términos de manual.
Si algo del sistema financiero te ha molestado o beneficiado, lo dices con naturalidad.

LÍMITES DE CONOCIMIENTO:
Sabes bien lo que has usado y lo que te ha pasado. No eres experto en economía.
Sobre productos que no has usado, respondes con desconocimiento honesto.

SOBRE TU NATURALEZA:
No menciones por tu cuenta que eres un personaje simulado o que hay un sistema detrás.
Si alguien te pregunta directamente si eres real, una persona, o una IA, puedes reconocerlo.
Fuera de eso, te comportas como quien eres.
"""
    return prompt


# ─── SALUD ────────────────────────────────────────────────────────────────────

_SEGURO_LABEL = {
    "sis_gratuito": "SIS gratuito — te lo asignaron por tu situación económica",
    "essalud": "EsSalud — accedes porque trabajas o trabajabas en planilla",
    "seguro_privado_eps": "un seguro privado (EPS) — lo tienes por trabajo o lo pagas tú",
    "ffaa_policia": "el seguro de Fuerzas Armadas / PNP",
    "sis_independiente_pagante": "SIS pagante — lo pagas tú siendo independiente",
    "sin_seguro": "no tienes seguro actualmente — cuando te enfermas, pagas de tu bolsillo",
}

_ESTABLECIMIENTO_LABEL = {
    "centro_salud_minsa": "un centro de salud del MINSA — es lo más cercano y accesible",
    "hospital_minsa": "un hospital del MINSA cuando es algo más serio",
    "essalud_posta": "la posta de EsSalud — vas ahí porque es tu derecho",
    "essalud_hospital": "el hospital de EsSalud cuando necesitas especialista",
    "clinica_privada": "una clínica privada — prefieres la atención aunque cueste más",
    "farmacia_botica": "la farmacia o botica — para cosas sencillas es suficiente",
    "medico_particular": "un médico particular de confianza",
    "otro": "distintos establecimientos según la situación",
}

_BARRERA_LABEL = {
    "tiempo_espera": "la espera es larga — a veces horas para una consulta de 10 minutos",
    "costo": "el costo es lo que más pesa — a veces no alcanza para medicamentos o consultas",
    "distancia": "la distancia — el establecimiento queda lejos y eso complica todo",
    "falta_medicamentos": "que no siempre hay medicamentos en el establecimiento",
    "trato_personal": "el trato — a veces no te atienden bien o te hacen esperar sin explicarte nada",
}


def _build_system_prompt_salud(perfil: dict[str, Any]) -> str:
    edad = int(float(perfil.get("edad", 37)))
    region_key = str(perfil.get("region", "lima_metropolitana"))
    seguro_key = str(perfil.get("tipo_seguro", "sis_gratuito"))
    establecimiento_key = str(perfil.get("establecimiento_preferido", "centro_salud_minsa"))
    nivel_edu = str(perfil.get("nivel_educativo", "secundaria_completa")).replace("_", " ")
    lengua = str(perfil.get("lengua_materna", "castellano"))
    sexo = str(perfil.get("sexo", "masculino"))
    zona = str(perfil.get("zona", "urbana"))
    ingreso = int(perfil.get("ingreso_mensual_soles", 1500))

    region_label = _REGION_LABEL.get(region_key, region_key.replace("_", " "))
    seguro_label = _SEGURO_LABEL.get(seguro_key, seguro_key.replace("_", " "))
    establecimiento_label = _ESTABLECIMIENTO_LABEL.get(
        establecimiento_key, establecimiento_key.replace("_", " ")
    )

    pronombre = "una mujer" if sexo == "femenino" else "un hombre"
    zona_ctx = "en zona rural" if zona == "rural" else "en zona urbana"

    if ingreso < 1000:
        ingreso_ctx = f"S/ {ingreso:,} al mes — cualquier gasto en salud impacta fuerte."
    elif ingreso < 2200:
        ingreso_ctx = f"S/ {ingreso:,} al mes — alcanza para lo básico, pero hay que cuidar los gastos médicos."
    else:
        ingreso_ctx = f"S/ {ingreso:,} al mes — puedes costear atención cuando la necesitas."

    edu = _edu_score(nivel_edu)
    seguro_num = {
        "sin_seguro": 0, "sis_gratuito": 1, "sis_independiente_pagante": 2,
        "essalud": 3, "ffaa_policia": 4, "seguro_privado_eps": 5,
    }.get(seguro_key, 1)
    is_lima = region_key in ("lima_metropolitana", "callao")
    is_young = edad < 35
    raw = edu + seguro_num + (1 if is_lima else 0) + (1 if is_young else 0)
    level = 1 if raw <= 3 else 2 if raw <= 7 else 3 if raw <= 10 else 4

    register_instr = _REGISTER_INSTRUCTIONS[level]
    region_note = _REGION_REGISTER_NOTES.get(region_key, "")
    lengua_nota = _LENGUA_NOTA.get(lengua, "")

    prompt = f"""Eres {pronombre} peruana/o de {edad} años que vive en {region_label}, {zona_ctx}.

TU SITUACIÓN DE SALUD:
Tienes {seguro_label}.
Cuando necesitas atención médica, acudes principalmente a {establecimiento_label}.
Ingresos del hogar: {ingreso_ctx}

TU PERFIL:
Nivel educativo: {nivel_edu}.
Lengua materna: {lengua}.
{lengua_nota}

CÓMO HABLAS:
{register_instr}
{region_note}
Siempre respondes en primera persona. Hablas desde tu experiencia real con el sistema de salud peruano.
Cuando hablas de enfermedades, medicamentos o consultas, usas el lenguaje que usarías con un familiar.
No hablas como médico ni como experto — eres un usuario del sistema, con sus frustraciones y sus recursos.

LÍMITES DE CONOCIMIENTO:
Sabes lo que has vivido: esperas largas, medicamentos caros, médicos que te explican poco.
No eres experto en medicina. Si algo no lo entiendes, lo dices sin problema.

SOBRE TU NATURALEZA:
No menciones por tu cuenta que eres un personaje simulado o que hay un sistema detrás.
Si alguien te pregunta directamente si eres real, una persona, o una IA, puedes reconocerlo.
Fuera de eso, te comportas como quien eres.
"""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Agente LangGraph
# ─────────────────────────────────────────────────────────────────────────────

class _State(TypedDict):
    messages: Annotated[list, add_messages]


class Agent:
    """
    Agente conversacional multi-turno que simula a un emprendedor peruano.
    El estado de conversación persiste entre llamadas a chat() usando MemorySaver.
    """

    def __init__(self, provider: LLMProvider, system_prompt: str) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._thread_id = str(uuid.uuid4())
        self._graph = self._build_graph()

    def _build_graph(self):
        system_prompt = self._system_prompt
        provider = self._provider

        def llm_node(state: _State) -> dict:
            msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            for m in state["messages"]:
                if hasattr(m, "type"):
                    role = "user" if m.type == "human" else "assistant"
                    msgs.append({"role": role, "content": str(m.content)})
                elif isinstance(m, dict):
                    msgs.append(m)
            response = provider.complete(msgs)
            return {"messages": [AIMessage(content=response)]}

        checkpointer = MemorySaver()
        graph = StateGraph(_State)
        graph.add_node("llm", llm_node)
        graph.set_entry_point("llm")
        graph.add_edge("llm", END)
        return graph.compile(checkpointer=checkpointer)

    def chat(self, message: str) -> str:
        """Envía un mensaje y devuelve la respuesta. El historial persiste automáticamente."""
        config = {"configurable": {"thread_id": self._thread_id}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        last = result["messages"][-1]
        return str(last.content) if hasattr(last, "content") else str(last)

    def reset(self) -> None:
        """Inicia una nueva conversación descartando el historial."""
        self._thread_id = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Builder público
# ─────────────────────────────────────────────────────────────────────────────

class AgentBuilder:
    """
    Factory que construye un Agent desde un perfil sintético.

    Uso con CallableProvider (mock/local):
        provider = CallableProvider(lambda msgs, **kw: "respuesta")
        agent = AgentBuilder.from_profile(perfil, provider)
        agent.chat("¿Qué vendes?")

    Uso con DeepSeek:
        provider = DeepSeekProvider(api_key="...")
        agent = AgentBuilder.from_profile(perfil, provider)
    """

    @classmethod
    def from_profile(
        cls,
        perfil: dict[str, Any],
        llm_provider: LLMProvider | None = None,
        callable: Callable[..., str] | None = None,
    ) -> Agent:
        """
        Crea un Agent desde un perfil sintético.

        Parámetros
        ----------
        perfil       : dict con las variables del perfil (output de PopulationGenerator)
        llm_provider : instancia de LLMProvider (tiene precedencia sobre callable)
        callable     : función (messages, **kwargs) -> str; se envuelve en CallableProvider
        """
        if llm_provider is None and callable is not None:
            llm_provider = CallableProvider(callable)
        elif llm_provider is None:
            raise ValueError(
                "Proporciona llm_provider o callable. "
                "Ejemplo: AgentBuilder.from_profile(perfil, CallableProvider(mi_fn))"
            )
        system_prompt = _build_system_prompt(perfil)
        return Agent(provider=llm_provider, system_prompt=system_prompt)
