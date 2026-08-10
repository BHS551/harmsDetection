"""
vlm.py — Capa 2: juicio situacional con un VLM (Amazon Nova Lite en Bedrock).
Solo se invoca en eventos AMBIGUOS que CLIP no puede resolver (¿pelea o trabajo?,
¿persona o casco?). Devuelve una decisión sí/no con una razón corta.
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
    "robos": ("¿hay indicios de robo, saqueo o allanamiento? Cuenta como SI que alguien "
              "fuerce o rompa una puerta, escaparate o vehículo, o se lleve mercancía."),
    "violencia": ("¿hay una pelea o agresión física real entre personas? Responde NO si es "
                  "deporte de contacto, un entrenamiento o una demostración controlada."),
    "caidas": ("¿hay alguna persona tendida o derrumbada en el suelo? Responde SI aunque no "
               "se vea el momento de la caída: basta con que esté en el suelo o desplomada "
               "en una postura anómala. Responde NO si está sentada o agachada a propósito."),
    "persona": "¿hay una persona (un ser humano) en la imagen?",
    "person": "is there a person (a human) in the image?",
    "cuchillo": "¿hay un cuchillo o un arma visible?",
    "knife": "is there a knife or weapon visible?",
}


def judge(jpg_bytes, label):
    """Devuelve (confirmed: bool, reason: str). En error, (None, motivo)."""
    q = QUESTION.get(label.lower() if isinstance(label, str) else label,
                     f"¿la imagen muestra claramente: {label}?")
    prompt = (f"Eres un analista de seguridad revisando UN fotograma de una cámara. {q} "
              "Responde en la PRIMERA palabra SI o NO, y luego una frase breve de justificación.")
    try:
        r = _rt.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [
                {"text": prompt},
                {"image": {"format": "jpeg", "source": {"bytes": jpg_bytes}}},
            ]}],
            inferenceConfig={"maxTokens": 60, "temperature": 0.0},
        )
        text = r["output"]["message"]["content"][0]["text"].strip()
        head = text[:12].lower().replace("í", "i")
        confirmed = head.startswith("si") or head.startswith("yes")
        return confirmed, text
    except Exception as e:
        # Ante fallo del VLM: NO confirmar (evita alertas basura); registrar motivo.
        return None, f"VLM error: {type(e).__name__}: {e}"
