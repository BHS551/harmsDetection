"""
vlm.py — Capa 2: juicio situacional con un VLM (Amazon Nova Lite en Bedrock).
Solo se invoca en eventos AMBIGUOS que CLIP no puede resolver (¿pelea o trabajo?,
¿persona o casco?). Devuelve una decisión sí/no con una razón corta.

v4 (benchmark, sin aplicar en producción): judge() ahora acepta una RÁFAGA de
varios frames en vez de uno solo. Motivo, medido con evidencia (Fighting003 +
Shooting002): con un solo frame, pedirle al VLM "golpe claro" nunca se cumple
en video de vigilancia granulado -- 0/24 llamadas confirmaron en los dos videos
probados, incluso llamadas que cayeron DENTRO del evento real. Un solo frame no
tiene la información temporal para distinguir "empujón" de "agresión real"; una
secuencia corta sí (se ve la trayectoria del golpe, o que el grupo se dispersa
corriendo justo después). Bedrock Converse acepta varias imágenes en un mismo
mensaje -- no hace falta otro modelo, solo mandarle más contexto.
"""
import os
import boto3

_rt = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = os.environ.get("HEIMDALL_VLM_MODEL", "us.amazon.nova-lite-v1:0")

# Pregunta específica por concepto (situacional, no solo "hay un objeto").
QUESTION = {
    # Las preguntas se reescribieron con las respuestas REALES del VLM sobre
    # escenas de incidente (ciclo 1). Tres problemas observados:
    #  - "¿está ocurriendo un robo?" es injuzgable en un fotograma: nadie ve el
    #    acto completo. Se pregunta por INDICIOS, que sí son visibles.
    #  - "violencia" confirmaba con deporte de contacto; se excluye explícitamente
    #    tras ver al VLM razonar "es una demostración de artes marciales".
    #  - en caídas el VLM hilaba finísimo ("no se ha caído, está tendida en el
    #    suelo"). Para vigilancia, una persona en el suelo YA es el evento
    #    alertable: se detecta el estado posterior, no el instante de la caída,
    #    que además es lo único observable en un fotograma suelto.
    # v3 (benchmark, sin aplicar en producción): v1 solo cubría robo A UNA
    # PROPIEDAD (forzar puerta/escaparate/vehículo, llevarse mercancía). v2
    # agregó el atraco a una PERSONA (Robbery048: arma apuntada en la calle).
    # Con Burglary024 (alguien rebuscando en la caja/escritorio de una tienda
    # de noche, sin forzar nada visible) ninguna de las dos aplicaba -- el VLM
    # veía "persona inclinada sobre un escritorio" y no lo relacionaba con
    # robo por falta de fuerza/violencia visible. Un robo silencioso (hurto,
    # allanamiento sin forcejeo) también cuenta, aunque no se vea el objeto
    # siendo tomado ni una puerta rota.
    "robos": ("¿hay indicios de robo? Cuenta como SI que alguien fuerce o rompa una puerta, "
              "escaparate o vehículo, se lleve mercancía, amenace/agreda a una persona para "
              "quitarle algo, O que alguien rebusque de forma furtiva en una caja registradora, "
              "cajón o escritorio de un negocio fuera de horario normal, aunque no se vea "
              "fuerza ni el objeto siendo tomado."),
    # v5 (benchmark, sin aplicar en producción): la combinación que funcionó,
    # medida sobre las mismas 17 ventanas de Fighting003 -- TP=4 FN=3 FP=0
    # TN=10 (cero falsos positivos, algo de recall perdido). Ni v1/v3 (bar
    # alto: "golpe claro") ni v2 (bar bajo, un solo frame: confirmaba
    # cualquier ambigüedad callejera) funcionaban. Lo que cambió el resultado
    # no fue solo el texto -- fue sumarle la RÁFAGA de varios frames (ver
    # armar_rafaga() en evaluar_video_v3.py): con más de un frame, el modelo
    # deja de necesitar "cuenta como SI ante cualquier ambigüedad" para no
    # perderse el evento real, y con eso ya no confirma los tramos calmados
    # de antes del evento como hacía v2 con un solo frame.
    "violencia": ("¿hay una pelea o agresión física real entre personas (golpes, empujones "
                  "violentos, forcejeo)? Cuenta como SI aunque la secuencia sea confusa o de baja "
                  "calidad -- SOLO responde NO si ves señales CONCRETAS de deporte organizado: "
                  "cuadrilátero, tatami, árbitro, guantes de boxeo o gimnasio con equipo visible."),
    "caidas": ("¿hay alguna persona tendida o derrumbada en el suelo? Responde SI aunque no "
               "se vea el momento de la caída: basta con que esté en el suelo o desplomada "
               "en una postura anómala. Responde NO si está sentada o agachada a propósito."),
    "persona": "¿hay una persona (un ser humano) en la imagen?",
    "person": "is there a person (a human) in the image?",
    "cuchillo": "¿hay un cuchillo o un arma visible?",
    "knife": "is there a knife or weapon visible?",
}


def judge(frames, label):
    """frames: un solo jpg (bytes) o una lista de jpgs EN ORDEN (ráfaga corta).
    Devuelve (confirmed: bool, reason: str). En error, (None, motivo)."""
    if isinstance(frames, (bytes, bytearray)):
        frames = [frames]
    n = len(frames)

    q = QUESTION.get(label.lower() if isinstance(label, str) else label,
                     f"¿la imagen muestra claramente: {label}?")
    if n > 1:
        prompt = (f"Eres un analista de seguridad revisando una secuencia de {n} fotogramas "
                  f"consecutivos de una cámara, en orden temporal. {q} Responde en la PRIMERA "
                  "palabra SI o NO, y luego una frase breve de justificación.")
    else:
        prompt = (f"Eres un analista de seguridad revisando UN fotograma de una cámara. {q} "
                  "Responde en la PRIMERA palabra SI o NO, y luego una frase breve de justificación.")

    content = [{"text": prompt}] + [
        {"image": {"format": "jpeg", "source": {"bytes": f}}} for f in frames
    ]
    try:
        r = _rt.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 60, "temperature": 0.0},
        )
        text = r["output"]["message"]["content"][0]["text"].strip()
        head = text[:12].lower().replace("í", "i")
        confirmed = head.startswith("si") or head.startswith("yes")
        return confirmed, text
    except Exception as e:
        # Ante fallo del VLM: NO confirmar (evita alertas basura); registrar motivo.
        return None, f"VLM error: {type(e).__name__}: {e}"
