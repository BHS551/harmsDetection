"""
tiers.py — las tres capas de la cascada como funciones de bucle. Cada una lee de
una cola de entrada y escribe en la(s) de salida; funcionan igual con LocalQueue
(un proceso) o SqsQueue (instancias separadas).

  Capa 0 (motion): fuente de frames + MOG2 -> candidato (solo con movimiento).
  Capa 1 (clip):   score contrastivo -> "clear" (alerta), "ambiguous" (->VLM), "none".
  Capa 2 (vlm):    juicio situacional sobre lo ambiguo -> alerta si confirma.
"""
import os
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

# Caudal máximo hacia la capa VLM: un juicio por (cámara, etiqueta) cada N segundos.
# Es el regulador de coste del sistema; súbelo para gastar menos, bájalo para
# reaccionar antes. El cooldown de alertas de common.py actúa DESPUÉS del VLM, así
# que no sirve para esto: cuando llega, el gasto ya se produjo.
VLM_MIN_INTERVAL = float(os.environ.get("HEIMDALL_VLM_MIN_INTERVAL", "6"))
_ultimo_vlm = {}

# Etiquetas que valen como prueba de que hay alguien en el fotograma.
ETIQUETAS_PERSONA = ("persona", "person")

# Corte por encima del cual CLIP decide SOLO, sin consultar al VLM.
# El valor global (0.15) resultó INALCANZABLE: medido sobre 161 candidatos de
# "persona", el margen máximo fue 0.106 y la mediana 0.063, así que la rama "clear"
# no se ejecutó ni una vez y TODO el tráfico de personas se pagaba en el VLM —
# que además confirmó 38 de 39, o sea que venía bien de origen.
# Para persona se baja a 0.08: por encima del ruido medido (0.058 sobre una montaña
# vacía, que el VLM rechazó con razón) y por debajo de los aciertos claros.
# Los eventos abstractos NO usan esto: siempre pasan por el VLM.
CLEAR_MARGIN_POR_ETIQUETA = {"persona": 0.08, "person": 0.08}


def _corte_claro(label, por_defecto):
    return CLEAR_MARGIN_POR_ETIQUETA.get(normalize_word(label), por_defecto)


def _postura_horizontal(frame):
    """True si hay alguien en postura horizontal. Es un DISPARADOR, no un veredicto.

    Medido en el ciclo 4: usar la postura para alertar directamente dobló el recall
    (1/4 -> 2/4) pero rompió dos negativos, porque la geometría distingue
    "horizontal" de "vertical", no "se ha caído" de "está tumbado a propósito".
    Un obrero agachado y un judoca proyectado son geométricamente idénticos a una
    víctima en el suelo.

    Por eso la postura ya no alerta: solo decide a QUIÉN vale la pena preguntar.
    Aporta el recall que CLIP no tiene; el criterio lo sigue poniendo el VLM, que
    sí sabe descartar deporte y posturas voluntarias. Y sigue siendo mucho más
    barato que preguntar por cada candidato con persona, porque alguien realmente
    horizontal es una fracción pequeña.
    """
    try:
        import pose
        if not pose.disponible():
            return False
        hay, motivo = pose.analizar(frame)
        if hay:
            print(f"[pose] postura horizontal detectada: {motivo} -> se consulta al VLM")
        return hay
    except Exception as e:
        print(f"[pose] fallo, se ignora la postura: {type(e).__name__}: {e}")
        return False


def _hay_persona(por_etiqueta, scorer):
    """True si CLIP ve una persona con margen suficiente.

    Sirve de puerta para los eventos abstractos: sin persona no hay robo, ni
    pelea, ni caída, así que no hace falta gastar una llamada al VLM.

    Si la cámara NO monitoriza personas, `persona` no está entre los prompts y no
    hay nada que comprobar: la puerta se abre. Cerrarla dejaría ciega a una cámara
    configurada solo con "robos", que es una configuración perfectamente legítima.
    """
    if not any(normalize_word(e) in ETIQUETAS_PERSONA for e in getattr(scorer, "labels", [])):
        return True
    for etiqueta, margen in (por_etiqueta or {}).items():
        if normalize_word(etiqueta) in ETIQUETAS_PERSONA and margen >= scorer.threshold_for(etiqueta):
            return True
    return False


def frames_from_rtsp(rtsp_url, fatal_on_fail=True, label=""):
    """Generador de frames BGR desde RTSP (usa las opciones TCP/ngrok del worker).

    fatal_on_fail: en el modo de UNA cámara por instancia, si el RTSP no conecta al
    arrancar la instancia sobra -> se auto-termina. En el motion box MULTI-cámara eso
    NO aplica: una cámara caída no debe tumbar la caja (las demás siguen), así que se
    reintenta la reconexión sin terminar el proceso."""
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000|analyzeduration;1000000|probesize;1000000|max_delay;500000")

    def _open():
        c = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c

    cap = _open()
    if not cap.isOpened():
        if fatal_on_fail:
            common.terminate_self("RTSP inalcanzable en el arranque")
            return
        # multi-cámara: seguir reintentando en caliente sin matar la caja
        for _ in range(30):
            time.sleep(4)
            cap = _open()
            if cap.isOpened():
                break
        else:
            print(f"[motion] cámara {label} sin conexión RTSP, hilo termina (la caja sigue)")
            return
    fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                cap.release(); time.sleep(2)
                cap = _open()
                fails = 0
            time.sleep(0.05); continue
        fails = 0
        yield frame


def _emit(out_queue, meta, frame, rois, distributed):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return False
    msg = {**meta, "ts": time.time(), "rois": [list(r) for r in rois]}
    msg.update(pack_frame(buf.tobytes(), distributed))
    out_queue.send(msg)
    return True


def run_motion(source, out_queue, meta, distributed=False, window_seconds=60.0,
               emit_interval=0.4, stop_event=None,
               burst_frames=0, burst_span=3.0, burst_cooldown=15.0):
    """Capa 0. `source` es un iterable de frames BGR. Dos modos de emisión:

    - VENTANA (por defecto, `burst_frames=0`): mientras haya movimiento, emite un
      candidato cada `emit_interval` s dentro de una ventana de `window_seconds` que se
      re-arma con cada movimiento. Es el modo del worker "todo-en-uno" (local).

    - RÁFAGA (`burst_frames>0`): al detectar movimiento captura hasta `burst_frames`
      frames repartidos en ~`burst_span` s y luego queda en silencio `burst_cooldown` s
      antes de poder disparar otra ráfaga (aunque el movimiento continúe). Es el modo
      del motion box barato: manda ~10 frames por evento a la capa CLIP y no la inunda.
    """
    md = MotionDetector()
    last_hb = 0.0

    if burst_frames > 0:
        per = max(0.05, burst_span / burst_frames)   # separación entre frames de la ráfaga
        remaining = 0
        last_emit = 0.0
        last_burst_end = -1e9
        for frame in source:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.time()
            if now - last_hb > 30:
                common.heartbeat(meta.get("device_id", ""), meta.get("owner_uid", ""), meta.get("camera_name", ""))
                last_hb = now
            rois = md.rois(frame)
            if not rois:
                continue
            if remaining == 0 and (now - last_burst_end) >= burst_cooldown:
                remaining = burst_frames          # nuevo evento -> arma la ráfaga
                last_emit = 0.0
            if remaining > 0 and (now - last_emit) >= per:
                if _emit(out_queue, meta, frame, rois, distributed):
                    last_emit = now
                    remaining -= 1
                    if remaining == 0:
                        last_burst_end = now       # arranca el enfriamiento
        return

    # --- modo ventana ---
    active_until = 0.0
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
            if _emit(out_queue, meta, frame, rois, distributed):
                last_emit = now


def run_motion_multi(cameras, out_queue, distributed=True, stop_event=None,
                     burst_frames=10, burst_span=3.0, burst_cooldown=15.0,
                     wake_hook=None):
    """Capa 0 EMPAQUETADA: corre la detección de movimiento de VARIAS cámaras en un
    solo proceso (el motion box barato 24/7). Una hebra por cámara, todas compartiendo
    la misma `out_queue` (SQS). `cameras` = lista de dicts {"rtsp": url, "meta": {...}}.

    `wake_hook` (opcional): callable que se invoca justo antes de encolar un candidato,
    para despertar la caja de análisis si estuviera apagada. Debe ser barato/idempotente
    (el hilo de cada cámara lo llama; la implementación debe hacer su propio debounce).
    """
    import threading

    def _cam_loop(cam):
        src = frames_from_rtsp(cam["rtsp"], fatal_on_fail=False,
                               label=cam["meta"].get("camera_name", ""))
        sink = out_queue
        if wake_hook is not None:
            sink = _WakingQueue(out_queue, wake_hook)
        try:
            run_motion(src, sink, cam["meta"], distributed=distributed, stop_event=stop_event,
                       burst_frames=burst_frames, burst_span=burst_span, burst_cooldown=burst_cooldown)
        except Exception as e:
            print(f"[motion] cámara {cam['meta'].get('camera_name','')} error: {e}")

    threads = []
    for cam in cameras:
        t = threading.Thread(target=_cam_loop, args=(cam,), daemon=True,
                             name=f"motion-{cam['meta'].get('camera_name','?')}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class _WakingQueue:
    """Envuelve una cola: dispara `wake_hook` antes de cada envío (para levantar la
    caja de análisis) y luego delega. El debounce vive dentro del hook."""
    def __init__(self, inner, wake_hook):
        self._inner = inner
        self._wake = wake_hook

    def send(self, msg):
        try:
            self._wake()
        except Exception as e:
            print("[motion] wake_hook error:", e)
        self._inner.send(msg)


def run_clip(in_queue, vlm_queue, scorer, distributed=False, stop_event=None,
             clear_margin=0.15, alert_pool=None, activity=None):
    """Capa 1. Consume candidatos, puntúa, decide clear/ambiguous/none.

    `activity`: dict opcional {"last": ts} que se sella con la hora en cada lote recibido;
    lo lee el watchdog de inactividad de la caja de análisis para apagarse tras 2h.
    Si un mensaje trae `blacklist` (conceptos de ESA cámara), solo se aceptan labels de
    esa lista: la caja es compartida y puntúa el universo de conceptos, pero cada cámara
    filtra a los suyos."""
    while stop_event is None or not stop_event.is_set():
        batch = in_queue.receive(wait=5)
        if batch and activity is not None:
            activity["last"] = time.time()
        for msg in batch:
            decision = "none"
            try:
                jpg = load_frame(msg)
                if jpg is None:
                    continue
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                rois = [tuple(r) for r in msg.get("rois", [])]
                score, label, coords, por_etiqueta = scorer.score_detallado(frame, rois)
                cam_bl = msg.get("blacklist")
                allowed = None if not cam_bl else {normalize_word(b) for b in cam_bl}
                vigila_caidas = allowed is None or "caidas" in allowed
                if vigila_caidas and _hay_persona(por_etiqueta, scorer) \
                        and _postura_horizontal(frame):
                    # La postura DISPARA la consulta, no la alerta. Se fuerza la
                    # etiqueta a "caidas" porque la pregunta que hay que hacerle al
                    # VLM es la de la caída, gane lo que gane CLIP —que casi nunca
                    # elige "caidas", precisamente por lo mala que es en ese concepto.
                    decision = "ambiguous"
                    label = "caidas"
                    score = max(score, por_etiqueta.get("caidas", 0.0))
                elif label is None or score < scorer.threshold_for(label):
                    decision = "none"
                elif allowed is not None and normalize_word(label) not in allowed:
                    decision = "none"   # concepto no monitoreado por esta cámara
                elif normalize_word(label) in ABSTRACT_EVENTS and not _hay_persona(por_etiqueta, scorer):
                    # PUERTA DE PERSONAS. Un robo, una pelea y una caída son, por
                    # definición, cosas que le pasan a alguien. Si CLIP no ve una
                    # persona en el fotograma, preguntarle al VLM es tirar dinero:
                    # medido, CLIP disparaba "violencia" con margen 0.088 sobre una
                    # montaña vacía y el VLM lo rechazaba invariablemente. Filtrar
                    # aquí sale gratis; preguntar cuesta.
                    decision = "none"
                    print(f"[clip] {label} score={score:.3f} -> descartado (sin persona)")
                elif normalize_word(label) in ABSTRACT_EVENTS or score < _corte_claro(label, clear_margin):
                    decision = "ambiguous"
                else:
                    decision = "clear"
                print(f"[clip] label={label} score={score:.3f} -> {decision}")
                if decision == "clear":
                    _fire(alert_pool, jpg, msg, score, coords, label, "clip")
                elif decision == "ambiguous":
                    # La capa 2 es el coste dominante (~120 USD/cámara/mes medidos:
                    # 1.877 llamadas/hora, el 97,8% de los candidatos). Una ráfaga de
                    # movimiento manda ~10 fotogramas casi idénticos y se pagaba un
                    # juicio por cada uno. Basta con juzgar uno por ventana y etiqueta:
                    # la escena no cambia en 3 s, así que no se pierde señal.
                    clave = (msg.get("camera_name", "?"), normalize_word(label))
                    ahora = time.time()
                    if ahora - _ultimo_vlm.get(clave, 0.0) < VLM_MIN_INTERVAL:
                        decision = "none"   # descartado por caudal, no por puntuación
                        print(f"[clip] {label} score={score:.3f} -> vlm omitido (caudal)")
                    else:
                        _ultimo_vlm[clave] = ahora
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
                # Un evento abstracto (robo, pelea, caída) se define por la relación
                # entre personas y con el entorno, no por lo que hay dentro de una
                # mancha de movimiento. Recortar borra justo esa información: medido
                # sobre 4 escenas reales de incidente, el VLM rechazaba el 100% de los
                # candidatos (0 de 210) porque se le preguntaba "¿hay un robo?" sobre
                # un rectángulo de ~130 px. La literatura apunta a lo mismo: para
                # reconocer interacción hace falta contexto global además del local.
                # Para conceptos de OBJETO ("persona", "cuchillo") el recorte sigue
                # siendo mejor: concentra resolución donde está la evidencia.
                if normalize_word(label) in ABSTRACT_EVENTS:
                    img_bytes = jpg
                else:
                    img_bytes = _crop_jpg(jpg, msg.get("coords"))
                confirmed, reason = vlm_mod.judge(img_bytes, label)
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
