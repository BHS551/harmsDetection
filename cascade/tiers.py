"""
tiers.py — las tres capas de la cascada como funciones de bucle. Cada una lee de
una cola de entrada y escribe en la(s) de salida; funcionan igual con LocalQueue
(un proceso) o SqsQueue (instancias separadas).

  Capa 0 (motion): fuente de frames + MOG2 -> candidato (solo con movimiento).
  Capa 1 (clip):   score contrastivo -> "clear" (alerta), "ambiguous" (->VLM), "none".
  Capa 2 (vlm):    juicio situacional sobre lo ambiguo -> alerta si confirma.
"""
import time
import cv2
import numpy as np

import common
import vlm as vlm_mod
from vision import MotionDetector, ClipScorer, normalize_word, MOTION_ONLY_LABELS
from transport import pack_frame, load_frame, cleanup_frame

# Un evento abstracto (robo/violencia/caída) NUNCA se resuelve solo con CLIP:
# siempre pasa al VLM. Persona/objeto pueden cerrarse con CLIP si el margen es alto.
ABSTRACT_EVENTS = {"caidas", "robos", "violencia"}


def frames_from_rtsp(rtsp_url):
    """Generador de frames BGR desde RTSP (usa las opciones TCP/ngrok del worker)."""
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000|analyzeduration;1000000|probesize;1000000|max_delay;500000")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        common.terminate_self("RTSP inalcanzable en el arranque")
        return
    fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                cap.release(); time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fails = 0
            time.sleep(0.05); continue
        fails = 0
        yield frame


def run_motion(source, out_queue, meta, distributed=False, window_seconds=60.0,
               emit_interval=0.4, stop_event=None):
    """Capa 0. `source` es un iterable de frames BGR. Emite candidatos a out_queue,
    como mucho uno cada `emit_interval` s (no inunda a la capa 1)."""
    md = MotionDetector()
    active_until = 0.0
    last_hb = 0.0
    last_emit = 0.0
    for frame in source:
        if stop_event is not None and stop_event.is_set():
            break
        now = time.time()
        if now - last_hb > 30:
            common.heartbeat(meta.get("device_id", ""), meta.get("owner_uid", ""), meta.get("camera_name", ""))
            last_hb = now
        rois = md.rois(frame)
        if rois:
            active_until = now + window_seconds  # cada movimiento re-arma la ventana
        if now <= active_until and rois and (now - last_emit) >= emit_interval:
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            last_emit = now
            msg = {**meta, "ts": now, "rois": [list(r) for r in rois]}
            msg.update(pack_frame(buf.tobytes(), distributed))
            out_queue.send(msg)


def run_clip(in_queue, vlm_queue, scorer, distributed=False, stop_event=None,
             clear_margin=0.15, alert_pool=None):
    """Capa 1. Consume candidatos, puntúa, decide clear/ambiguous/none."""
    while stop_event is None or not stop_event.is_set():
        for msg in in_queue.receive(wait=5):
            decision = "none"
            try:
                jpg = load_frame(msg)
                if jpg is None:
                    continue
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                rois = [tuple(r) for r in msg.get("rois", [])]
                score, label, coords = scorer.score(frame, rois)
                if label is None or score < scorer.threshold_for(label):
                    decision = "none"
                elif normalize_word(label) in ABSTRACT_EVENTS or score < clear_margin:
                    decision = "ambiguous"
                else:
                    decision = "clear"
                print(f"[clip] label={label} score={score:.3f} -> {decision}")
                if decision == "clear":
                    _fire(alert_pool, jpg, msg, score, coords, label, "clip")
                elif decision == "ambiguous":
                    out = {**{k: v for k, v in msg.items() if k != "_handle"},
                           "label": label, "score": score, "coords": list(coords) if coords else None}
                    vlm_queue.send(out)
            finally:
                in_queue.delete(msg)
                # El frame temporal en S3: si fue AMBIGUO lo consume la capa 2 (lo borra
                # ella); en 'clear'/'none' ya no se necesita -> se limpia aquí.
                if distributed and decision != "ambiguous":
                    cleanup_frame(msg)


def run_vlm(in_queue, distributed=False, stop_event=None, alert_pool=None):
    """Capa 2. Consume ambiguos, pregunta al VLM, alerta si confirma."""
    while stop_event is None or not stop_event.is_set():
        for msg in in_queue.receive(wait=5):
            try:
                jpg = load_frame(msg)
                if jpg is None:
                    continue
                label = msg.get("label", "?")
                # recortar a la zona sospechosa para dar al VLM el contexto justo
                crop_bytes = _crop_jpg(jpg, msg.get("coords"))
                confirmed, reason = vlm_mod.judge(crop_bytes, label)
                print(f"[vlm] label={label} confirmed={confirmed} :: {reason[:80]}")
                if confirmed:
                    coords = tuple(msg["coords"]) if msg.get("coords") else None
                    _fire(alert_pool, jpg, msg, msg.get("score", 0.0), coords, label, "vlm")
            finally:
                in_queue.delete(msg)
                if distributed:
                    cleanup_frame(msg)


def _crop_jpg(jpg_bytes, coords):
    if not coords:
        return jpg_bytes
    frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = [int(v) for v in coords]
    # margen de contexto alrededor de la caja para el VLM
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * 0.5), int((y2 - y1) * 0.5)
    crop = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    if crop.size == 0:
        return jpg_bytes
    ok, buf = cv2.imencode(".jpg", crop)
    return buf.tobytes() if ok else jpg_bytes


def _fire(alert_pool, jpg, msg, score, coords, label, source):
    meta = {k: msg.get(k) for k in ("camera_name", "client_id", "owner_uid", "ts")}
    if alert_pool is not None:
        alert_pool.submit(common.raise_alert, jpg, meta, score, coords, label, source)
    else:
        common.raise_alert(jpg, meta, score, coords, label, source)
