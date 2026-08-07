"""
common.py — I/O compartido: token Firebase, subida de frames a S3, registro de
detección, notificación al usuario, heartbeat y auto-terminación de la instancia.
Extraído del worker monolítico para que todas las capas alerten igual.
"""
import os
import time
import json
import uuid
import threading
import http.client
import urllib.request
import boto3

S3_BUCKET = "detection-frames-tests"
S3_PREFIX = "cameras/"
WORKER_EVENTS_HOST = os.environ.get("WORKER_EVENTS_HOST", "p4nojr0ec5.execute-api.us-east-1.amazonaws.com")
STORE_REGISTER_HOST = os.environ.get("STORE_REGISTER_HOST", "c038gkbfm8.execute-api.us-east-1.amazonaws.com")
HEIMDAL_MANAGER_HOST = os.environ.get("HEIMDAL_MANAGER_HOST", "a2ukt8vyhb.execute-api.us-east-1.amazonaws.com")
HEIMDAL_MANAGER_PATH = os.environ.get("HEIMDAL_MANAGER_PATH", "/default/heimdalManager")
# Secreto compartido para que el motion box invoque el "ensureAnalysis" de
# HeimdalManager sin token de usuario (autenticación máquina-a-máquina).
INTERNAL_SECRET = os.environ.get("HEIMDALL_INTERNAL_SECRET", "")

_s3 = boto3.client("s3", region_name="us-east-1")

# Supresión de alertas repetidas. Cada candidato que confirman CLIP o el VLM escribe
# un frame en S3, un item en DynamoDB y dispara SMS + email; ante una escena con
# movimiento sostenido eso son miles de eventos por hora sobre la MISMA persona.
# Se suprime por (cámara, etiqueta): mientras la amenaza persista se re-alerta cada
# ALERT_COOLDOWN s —misma semántica que el REALERT_INTERVAL del monolito— y se avisa
# al usuario como mucho cada NOTIFY_COOLDOWN s, porque el SMS es el canal caro.
# Esto es la segunda línea de defensa: la primera es el modo ráfaga de la capa 0,
# pero raise_alert la comparten también las cajas distribuidas.
ALERT_COOLDOWN = float(os.environ.get("HEIMDALL_ALERT_COOLDOWN", "20"))
NOTIFY_COOLDOWN = float(os.environ.get("HEIMDALL_NOTIFY_COOLDOWN", "300"))

_cooldown_lock = threading.Lock()
_last_alert = {}
_last_notify = {}


def _should_fire(store, key, cooldown):
    """True si toca disparar, registrando el instante. raise_alert corre en un pool
    de 4 hilos, así que la comprobación y la marca van bajo el mismo lock."""
    now = time.time()
    with _cooldown_lock:
        if now - store.get(key, 0.0) < cooldown:
            return False
        store[key] = now
        return True

try:
    from firebase_auth import get_firebase_token
except Exception:  # en entornos sin firebase_admin (p. ej. pruebas)
    def get_firebase_token():
        return os.environ.get("HEIMDALL_TEST_TOKEN", "")


def get_instance_id():
    try:
        token = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-id",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=2).read().decode()
    except Exception:
        return None


def terminate_self(reason):
    print(f"Auto-terminando la instancia: {reason}")
    iid = get_instance_id()
    if not iid:
        os._exit(3)
    try:
        boto3.client("ec2", region_name="us-east-1").terminate_instances(InstanceIds=[iid])
    except Exception as e:
        print("terminate_self error:", e)
        os._exit(3)


def upload_frame_to_s3(jpg_bytes, ts, score, coords=None, detection_id=None):
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(ts))
    millis = int((ts % 1) * 1000)
    coord_str = f"_{coords[0]}-{coords[1]}-{coords[2]}-{coords[3]}" if coords else ""
    uuid_str = f"_{detection_id}" if detection_id else ""
    key = f"{S3_PREFIX}{timestr}_{millis:03d}_score-{score:.3f}{coord_str}{uuid_str}.jpg"
    _s3.put_object(Bucket=S3_BUCKET, Key=key, Body=jpg_bytes, ContentType="image/jpeg")
    return key


def _post(host, path, payload):
    conn = http.client.HTTPSConnection(host, timeout=10)
    conn.request("POST", path, json.dumps(payload),
                 {"Content-Type": "application/json", "Authorization": f"Bearer {get_firebase_token()}"})
    res = conn.getresponse()
    res.read()
    return res.status


def store_register(payload):
    return _post(STORE_REGISTER_HOST, "/default/storeRegister", payload)


def post_worker_event(payload):
    return _post(WORKER_EVENTS_HOST, "/", payload)


def notify_user(owner_uid, event_type, camera, score, detection_id):
    try:
        post_worker_event({"action": "notify", "owner_uid": owner_uid, "event_type": event_type,
                           "camera": camera, "cosine_sim": round(float(score), 3),
                           "detection_id": detection_id})
    except Exception as e:
        print("notify_user error:", e)


def heartbeat(device_id, owner_uid, camera_name, status="running"):
    try:
        post_worker_event({"action": "heartbeat", "device_id": device_id, "owner_uid": owner_uid,
                           "camera_name": camera_name, "status": status})
    except Exception as e:
        print("heartbeat error:", e)


_last_wake = {"t": 0.0}
_WAKE_DEBOUNCE = 30.0  # como mucho una llamada de "despierta" cada 30 s (todas las cámaras)


def ensure_analysis():
    """Despierta la caja de análisis (CLIP+VLM) si estuviera apagada. La llama el
    motion box antes de encolar un candidato. Idempotente y con debounce: HeimdalManager
    no arranca una segunda caja si ya hay una viva. Falla en silencio (no debe tumbar
    la detección de movimiento)."""
    now = time.time()
    if now - _last_wake["t"] < _WAKE_DEBOUNCE:
        return
    _last_wake["t"] = now
    try:
        conn = http.client.HTTPSConnection(HEIMDAL_MANAGER_HOST, timeout=8)
        conn.request("POST", HEIMDAL_MANAGER_PATH, json.dumps({"action": "ensureAnalysis"}),
                     {"Content-Type": "application/json", "x-internal-secret": INTERNAL_SECRET})
        res = conn.getresponse(); res.read(); conn.close()
    except Exception as e:
        print("ensure_analysis error:", e)


def raise_alert(jpg_bytes, meta, score, coords, label, source):
    """S3 + storeRegister + notificación. `source` etiqueta la capa que confirmó
    (p. ej. 'clip' o 'vlm') para trazabilidad. No bloquea (llamar en hilo)."""
    try:
        camera = meta.get("camera_name", "entrance")
        key = (camera, label)
        if not _should_fire(_last_alert, key, ALERT_COOLDOWN):
            return None
        detection_id = str(uuid.uuid4())
        image_key = upload_frame_to_s3(jpg_bytes, meta.get("ts", time.time()), score, coords, detection_id)
        store_register({
            "cammera": meta.get("camera_name", "entrance"),
            "clientId": meta.get("client_id", 1),
            "event_type": label,
            "detection_id": detection_id,
            "cosine_sim": score,
            "image_key": image_key,
            "owner_uid": meta.get("owner_uid", ""),
            "confirmed_by": source,
        })
        # La detección queda registrada siempre; el aviso al usuario va con su propio
        # cooldown, más largo, para no convertir una escena transitada en cientos de
        # SMS. El historial de la consola sigue mostrando todos los eventos.
        notified = _should_fire(_last_notify, key, NOTIFY_COOLDOWN)
        if notified:
            notify_user(meta.get("owner_uid", ""), label, camera, score, detection_id)
        print(f"ALERTA [{source}] {label} score={score:.3f} "
              f"notificado={'si' if notified else 'no'} -> {image_key}")
        return detection_id
    except Exception as e:
        print("raise_alert error:", e)
        return None
