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
    "robos": "¿está ocurriendo un robo o alguien está robando/forzando algo?",
    "violencia": "¿hay violencia real: una pelea o agresión física entre personas?",
    "caidas": "¿una persona se ha caído al suelo o está tendida como tras una caída?",
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
