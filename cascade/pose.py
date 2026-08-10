"""
pose.py — Detección de caídas por POSTURA, dentro de Heimdall (capa 1).

Por qué existe: una caída es un evento temporal y CLIP no la ve. Medido sobre
escenas reales, la clase `caidas` solo se detectaba de rebote, y cada intento
costaba una llamada al VLM (Mimir). La geometría del cuerpo, en cambio, sí es
visible en un fotograma suelto: una persona tumbada ocupa una caja ancha y baja,
y su torso está casi horizontal.

Esto resuelve `caidas` **sin llamar al VLM**: es la clase entera saliendo del
coste variable. La literatura da 92–98% de precisión a este enfoque.

El modelo (yolov8n-pose, 6,8 MB) se descarga de S3, no de internet, para no
depender de GitHub en el arranque de cada instancia.
"""
import os

# Umbrales de geometría. Conservadores a propósito: un falso positivo de caída
# despierta a alguien de madrugada.
#   - ancho/alto de la caja: una persona de pie ronda 0,3-0,5; tumbada supera 1.
RATIO_CAJA_TUMBADO = float(os.environ.get("HEIMDALL_POSE_RATIO", "1.2"))
#   - inclinación del torso (hombros->caderas) respecto a la horizontal, en grados.
#     De pie ~90°, tumbado ~0°. Se exige claramente horizontal.
GRADOS_TORSO_TUMBADO = float(os.environ.get("HEIMDALL_POSE_GRADOS", "35"))
#   - confianza mínima de la detección de persona.
CONF_MIN = float(os.environ.get("HEIMDALL_POSE_CONF", "0.4"))

MODELO_S3 = ("detection-frames-tests", "worker/models/yolov8n-pose.pt")
RUTA_LOCAL = os.environ.get("HEIMDALL_POSE_MODEL", "/home/ubuntu/app/cascade/yolov8n-pose.pt")

# Índices de keypoints COCO usados: hombros 5/6, caderas 11/12.
HOMBRO_IZQ, HOMBRO_DER, CADERA_IZQ, CADERA_DER = 5, 6, 11, 12

_modelo = None
_no_disponible = False


def _cargar():
    """Carga perezosa. Si falla, se marca no disponible y el sistema sigue
    funcionando exactamente como antes (la caída se decidirá por el VLM)."""
    global _modelo, _no_disponible
    if _modelo is not None or _no_disponible:
        return _modelo
    try:
        if not os.path.exists(RUTA_LOCAL):
            import boto3
            os.makedirs(os.path.dirname(RUTA_LOCAL), exist_ok=True)
            boto3.client("s3", region_name="us-east-1").download_file(
                MODELO_S3[0], MODELO_S3[1], RUTA_LOCAL)
        from ultralytics import YOLO
        _modelo = YOLO(RUTA_LOCAL)
        print("[pose] yolov8n-pose cargado")
    except Exception as e:
        _no_disponible = True
        print(f"[pose] no disponible ({type(e).__name__}: {e}); se seguirá usando el VLM")
    return _modelo


def disponible():
    return _cargar() is not None


def _angulo_torso(kp):
    """Grados del torso respecto a la horizontal. None si faltan keypoints."""
    import math
    try:
        hx = (kp[HOMBRO_IZQ][0] + kp[HOMBRO_DER][0]) / 2.0
        hy = (kp[HOMBRO_IZQ][1] + kp[HOMBRO_DER][1]) / 2.0
        cx = (kp[CADERA_IZQ][0] + kp[CADERA_DER][0]) / 2.0
        cy = (kp[CADERA_IZQ][1] + kp[CADERA_DER][1]) / 2.0
    except (IndexError, TypeError):
        return None
    dx, dy = abs(cx - hx), abs(cy - hy)
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))    # 90 = vertical, 0 = horizontal


def analizar(frame_bgr):
    """Devuelve (hay_caida, motivo). Nunca lanza: ante cualquier fallo devuelve
    (False, motivo) y el flujo normal por el VLM sigue disponible."""
    modelo = _cargar()
    if modelo is None:
        return False, "pose no disponible"
    try:
        res = modelo.predict(frame_bgr, verbose=False, conf=CONF_MIN, classes=[0])
    except Exception as e:
        return False, f"error de inferencia: {type(e).__name__}"
    if not res:
        return False, "sin resultados"

    r = res[0]
    cajas = getattr(r, "boxes", None)
    kps = getattr(r, "keypoints", None)
    if cajas is None or len(cajas) == 0:
        return False, "sin personas"

    for i in range(len(cajas)):
        try:
            x1, y1, x2, y2 = [float(v) for v in cajas.xyxy[i][:4]]
        except Exception:
            continue
        ancho, alto = x2 - x1, y2 - y1
        if alto <= 0:
            continue
        ratio = ancho / alto

        angulo = None
        if kps is not None and getattr(kps, "xy", None) is not None and i < len(kps.xy):
            angulo = _angulo_torso([[float(c) for c in p] for p in kps.xy[i]])

        # Dos evidencias independientes; basta una, pero se informa cuál.
        if ratio >= RATIO_CAJA_TUMBADO:
            return True, f"caja tumbada (ancho/alto={ratio:.2f})"
        if angulo is not None and angulo <= GRADOS_TORSO_TUMBADO:
            return True, f"torso horizontal ({angulo:.0f}° sobre la horizontal)"

    return False, "personas de pie"
