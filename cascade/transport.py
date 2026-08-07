"""
transport.py — cola entre capas, con la MISMA interfaz en local y distribuido.

LocalQueue: cola en memoria (las 3 capas en un proceso). Útil para una instancia
            barata que corre todo, y para pruebas.
SqsQueue:   Amazon SQS (capas en instancias separadas). Los mensajes son JSON
            pequeños; los frames viajan por S3 (referencia por key), porque un
            JPEG excede el límite de 256 KB de un mensaje SQS.
"""
import json
import queue
import base64
import uuid
import boto3

FRAME_BUCKET = "detection-frames-tests"
FRAME_PREFIX = "pipeline/candidates/"


class LocalQueue:
    def __init__(self):
        self._q = queue.Queue()

    def send(self, msg):
        self._q.put(dict(msg))

    def receive(self, wait=1, max_msgs=5):
        out = []
        try:
            out.append(self._q.get(timeout=wait))
        except queue.Empty:
            return []
        return out

    def delete(self, msg):
        pass  # nada que borrar en memoria


class SqsQueue:
    def __init__(self, url, region="us-east-1"):
        self.url = url
        self.sqs = boto3.client("sqs", region_name=region)

    def send(self, msg):
        self.sqs.send_message(QueueUrl=self.url, MessageBody=json.dumps(msg))

    def receive(self, wait=5, max_msgs=5):
        r = self.sqs.receive_message(QueueUrl=self.url, MaxNumberOfMessages=max_msgs,
                                     WaitTimeSeconds=min(wait, 20))
        out = []
        for m in r.get("Messages", []):
            d = json.loads(m["Body"])
            d["_handle"] = m["ReceiptHandle"]
            out.append(d)
        return out

    def delete(self, msg):
        h = msg.get("_handle")
        if h:
            self.sqs.delete_message(QueueUrl=self.url, ReceiptHandle=h)


# --- Empaquetado de frames: inline (local) o por S3 (distribuido) ---
_s3 = boto3.client("s3", region_name="us-east-1")


def pack_frame(jpg_bytes, distributed):
    """Devuelve un dict de referencia al frame, según el modo."""
    if distributed:
        key = f"{FRAME_PREFIX}{uuid.uuid4()}.jpg"
        _s3.put_object(Bucket=FRAME_BUCKET, Key=key, Body=jpg_bytes, ContentType="image/jpeg")
        return {"frame_key": key}
    return {"frame_b64": base64.b64encode(jpg_bytes).decode("ascii")}


def load_frame(msg):
    """Recupera los bytes JPEG de un mensaje (inline o desde S3)."""
    if msg.get("frame_b64"):
        return base64.b64decode(msg["frame_b64"])
    if msg.get("frame_key"):
        return _s3.get_object(Bucket=FRAME_BUCKET, Key=msg["frame_key"])["Body"].read()
    return None


def cleanup_frame(msg):
    """Borra el frame temporal de S3 tras procesarlo (solo modo distribuido)."""
    if msg.get("frame_key"):
        try:
            _s3.delete_object(Bucket=FRAME_BUCKET, Key=msg["frame_key"])
        except Exception:
            pass
