import cv2
import time
import os
import re
import sys
import threading
import unicodedata
import urllib.request
from collections import deque
from PIL import Image
import torch
import clip
import concurrent.futures
import http.client
import json
import boto3
import uuid
from firebase_auth import get_firebase_token

# === Load context ===
context_path = sys.argv[1] if len(sys.argv) > 1 else "context.json"
with open(context_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data)
print(type(data))

# === S3 Configuration ===
S3_BUCKET_NAME = "detection-frames-tests"
S3_PREFIX = "cameras/"

s3_client = boto3.client("s3", region_name="us-east-1")
secrets_client = boto3.client("secretsmanager", region_name="us-east-1")

# Endpoint de eventos del worker (heartbeat + notificación al usuario).
WORKER_EVENTS_HOST = os.environ.get(
    "WORKER_EVENTS_HOST", "p4nojr0ec5.execute-api.us-east-1.amazonaws.com"
)

# Región e id de esta instancia EC2 (para auto-terminarse si la cámara no conecta).
def get_instance_id():
    try:
        # IMDSv2
        token = urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            ),
            timeout=2,
        ).read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=2).read().decode()
    except Exception:
        return None

# === Runtime tuning ===
# Cada parámetro se puede sobreescribir por cámara desde context.json (lo envía
# la UI/heimdalManager) o por variable de entorno; si no, usa el default.
def tune(ctx_key, env_key, default, cast=float):
    if isinstance(data, dict) and ctx_key in data and data[ctx_key] is not None:
        return cast(data[ctx_key])
    return cast(os.environ.get(env_key, default))

# Live debug windows: OFF by default so headless/EC2 hosts don't crash on cv2.imshow.
SHOW_WINDOWS = 0
# Max detections-per-second we actually run CLIP on (time-based, FPS-independent).
MIN_DETECTION_INTERVAL = tune("min_interval", "HEIMDALL_MIN_INTERVAL", "0.4", float)
# Consecutive positive frames required before firing an alert (debounces false positives).
ALERT_THRESHOLD = tune("alert_threshold", "HEIMDALL_ALERT_THRESHOLD", "3", int)
# Minimum seconds between two alerts for the same camera (avoids spamming S3/API).
ALERT_COOLDOWN = tune("alert_cooldown", "HEIMDALL_ALERT_COOLDOWN", "10", float)
# Frames to let the background model warm up before trusting motion (skips alerts).
WARMUP_FRAMES = tune("warmup_frames", "HEIMDALL_WARMUP_FRAMES", "30", int)
# Umbral del MARGEN contrastivo (concepto − mejor distractor) para un positivo.
# Antes era similitud coseno absoluta (0.27); con el score contrastivo el número
# es un margen pequeño. Default bajo = prioriza recall (no perder amenazas); la
# validación temporal (ALERT_THRESHOLD frames seguidos) filtra las falsas alarmas.
DETECTION_THRESHOLD = tune("threshold", "HEIMDALL_THRESHOLD", "0.02", float)
# Motion detection runs on a downscaled frame for speed; ROIs are scaled back up.
MOTION_DOWNSCALE = 0.5
# Minimum contour area (in downscaled pixels) to count as real motion.
MIN_MOTION_AREA = 500
# Cap ROIs per frame so a noisy scene can't blow up the CLIP batch.
MAX_ROIS = 10
# Region proposal (validado offline sobre imágenes reales de Sohas): la ganancia
# viene de CUADRAR la caja de movimiento y garantizar un lado mínimo, NO de añadir
# contexto. Barrido medido (recall @umbral fijo): crudo 88% / pequeños 77%  ->
# solo-min-size 93% / pequeños 87%  ->  con padding 0.4 baja a 80% / 64% (el
# contexto extra diluye el objeto y hunde el score contrastivo). Por eso el padding
# default es 0.0: solo cuadramos y forzamos tamaño mínimo. Configurable por cámara.
ROI_PADDING = tune("roi_padding", "HEIMDALL_ROI_PADDING", "0.0", float)
# Lado mínimo del recorte (px, resolución completa) para que el upscale a 224 no
# quede borroso. Cajas de movimiento diminutas se expanden hasta este tamaño; es
# lo que recupera los objetos pequeños (<1% del frame), la debilidad medida.
MIN_ROI_SIZE = tune("min_roi_size", "HEIMDALL_MIN_ROI_SIZE", "96", int)
# Tolerate this many consecutive failed reads (jittery ngrok/RTSP) before reconnecting.
# A single dropped read is normal; reconnecting on every one just thrashes the tunnel.
MAX_READ_FAILURES = int(os.environ.get("HEIMDALL_MAX_READ_FAILURES", "30"))
# Seconds to wait for the first decodable frame within ONE open attempt. H264 only
# decodes from a keyframe and TP-Link GOPs are long, so the first frame can lag a few
# seconds — wait it out instead of tearing down and re-handshaking.
RTSP_OPEN_TIMEOUT = float(os.environ.get("HEIMDALL_OPEN_TIMEOUT", "12"))
# EL MOVIMIENTO ES LA PRIMERA CAPA. La detección está APAGADA hasta que MOG2 detecta
# movimiento; cada movimiento abre/RE-ARMA una VENTANA de detección de esta duración
# (SEGUNDOS, independiente de los FPS) en la que se corre CLIP sobre la zona del
# movimiento. La ventana se apaga cuando transcurre este tiempo desde el ÚLTIMO
# movimiento. (Se eliminó el barrido periódico sobre escena estática: fabricaba falsas
# alarmas —casco->persona, barrotes->robo—. Un objeto totalmente inmóvil ya no se
# detecta, por diseño: si no hay movimiento, no hay nada que un vigilante consideraría.)
DETECTION_WINDOW_SECONDS = tune("detection_window_seconds", "HEIMDALL_DETECTION_WINDOW_SEC", "60", float)



def mask_rtsp_url(url):
    return re.sub(r":([^:@/]+)@", ":****@", url)

def open_rtsp_capture(url, retries=5, delay_sec=2):
    """Open RTSP with TCP transport; ngrok tunnels often fail on UDP."""
    # rtsp_transport;tcp     -> ngrok forwards TCP only (no UDP/RTP)
    # stimeout;5000000       -> 5s socket I/O timeout (OpenCV's bundled ffmpeg accepts this)
    # analyzeduration/probesize -> cap stream analysis at ~1s/1MB so open() returns fast
    #                              instead of probing for the default 5s
    # max_delay;500000       -> small reorder buffer for lower latency
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp"
        "|stimeout;5000000"
        "|analyzeduration;1000000"
        "|probesize;1000000"
        "|max_delay;500000"
    )
    for attempt in range(1, retries + 1):
        print(f"Opening RTSP stream (attempt {attempt}/{retries}): {mask_rtsp_url(url)}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Keep only the freshest frame buffered so processing lag doesn't replay stale frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            # Hold the connection and poll for the first keyframe rather than failing the
            # whole attempt on one slow read (which forces a costly fresh handshake).
            deadline = time.time() + RTSP_OPEN_TIMEOUT
            while time.time() < deadline:
                ret, _ = cap.read()
                if ret:
                    print("RTSP stream opened successfully")
                    return cap
                time.sleep(0.1)
            cap.release()
            print(f"Opened but no frame within {RTSP_OPEN_TIMEOUT:.0f}s")
        else:
            print("VideoCapture.isOpened() returned False")
        if attempt < retries:
            time.sleep(delay_sec)
    return None

# === Camera and Detection Configuration ===
# La URL RTSP (con credenciales) se obtiene de Secrets Manager por referencia
# (rtsp_secret_id); así no viaja en el UserData ni queda en claro en el disco.
# Se mantiene compatibilidad con context.json que traiga rtsp_path directo.
def resolve_rtsp_url(ctx):
    if ctx.get("rtsp_path"):
        return ctx["rtsp_path"]
    secret_id = ctx.get("rtsp_secret_id")
    if secret_id:
        return secrets_client.get_secret_value(SecretId=secret_id)["SecretString"]
    return None

rtsp_url = resolve_rtsp_url(data)
if not rtsp_url:
    raise Exception("No RTSP source in context (rtsp_path o rtsp_secret_id)")
owner_uid = data.get('owner_uid', '')
device_id = str(data.get('instance_id') or data.get('device_id') or '')

# Prohibited items to detect come from the context blacklist (e.g. ["knife"]).
detection_blacklist = data.get("detection_blacklist") or ["person"]
print("Detection targets (blacklist):", detection_blacklist)

# The UI sends user-facing detection words (Spanish, configurable per camera).
# CLIP was trained mostly on English alt-text, so a bare Spanish word embeds
# poorly; each known word expands to several English prompt variants that all
# report the user's original word as the event label. Words not in the map
# (custom words typed in the UI) pass through to CLIP verbatim.
PROMPT_MAP = {
    "caidas": [
        "a person fallen on the floor",
        "a person falling down",
        "a person collapsed on the ground",
    ],
    "robos": [
        "a robbery in progress",
        "a person stealing from someone",
        "a burglar breaking into a building",
    ],
    "violencia": [
        "people fighting violently",
        "a person hitting another person",
        "a violent physical assault",
    ],
    "persona": ["a photo of a person"],
    "person": ["a photo of a person"],
    # Prompts centrados en la HOJA/metal (no en "sostener"), para subir el recall del
    # cuchillo SIN dispararse con objetos de mano (teléfono/billetera comparten el
    # contexto "en mano", pero no la hoja metálica). Medido: recall 48%->65% a 10% FA.
    "cuchillo": ["a photo of a knife", "a sharp knife blade", "a metal knife blade",
                 "the blade of a knife", "a kitchen knife"],
    "knife": ["a photo of a knife", "a sharp knife blade", "a metal knife blade",
              "the blade of a knife", "a kitchen knife"],
}


def normalize_word(word):
    """Lowercase and strip accents so 'Caídas' still hits the 'caidas' map key."""
    word = str(word).strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", word)
        if unicodedata.category(c) != "Mn"
    )


# Umbral por concepto, ahora sobre el MARGEN contrastivo (no similitud absoluta).
# Valores bajos = prioriza recall. Sobreescribible desde context.json con
# "thresholds": { "caidas": 0.03, ... }. Fallback: DETECTION_THRESHOLD.
DEFAULT_PROMPT_THRESHOLDS = {
    "persona": 0.03,
    "person": 0.03,
    "cuchillo": 0.03,
    "knife": 0.03,
    "pistola": 0.02,
    "pistol": 0.02,
    "caidas": 0.02,
    "robos": 0.02,
    "violencia": 0.02,
}
_ctx_thresholds = data.get("thresholds") if isinstance(data, dict) else None
if isinstance(_ctx_thresholds, dict):
    for k, v in _ctx_thresholds.items():
        try:
            DEFAULT_PROMPT_THRESHOLDS[normalize_word(k)] = float(v)
        except (TypeError, ValueError):
            pass


def threshold_for(label):
    """Umbral aplicable a la palabra ganadora (con fallback al global)."""
    return DEFAULT_PROMPT_THRESHOLDS.get(normalize_word(label), DETECTION_THRESHOLD)


# Empty/whitespace-only entries would become empty CLIP prompts that can still
# fire alerts (with a blank event_type), so they are dropped up front.
cleaned_blacklist = [w for w in detection_blacklist if str(w).strip()] or ["person"]

detection_prompts = []
prompt_labels = []  # aligned with detection_prompts: prompt i reports label prompt_labels[i]
for word in cleaned_blacklist:
    for prompt in PROMPT_MAP.get(normalize_word(word), [str(word)]):
        detection_prompts.append(prompt)
        prompt_labels.append(word)
print("CLIP prompts:", detection_prompts)

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# FP16 on GPU: ~30% faster inference and half the VRAM. CPU stays FP32 (half is slow there).
use_half = device == "cuda"
if use_half:
    model = model.half()
model.eval()

# Precompute one text embedding per prompt variant. A frame matches if ANY prompt scores high.
# truncate=True: a custom word longer than CLIP's 77-token context would otherwise
# raise inside tokenize and kill the worker at startup.
text_tokens = clip.tokenize(detection_prompts, truncate=True).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

# Con el gating por movimiento (ver DETECTION_WINDOW_SECONDS) TODOS los conceptos se
# evalúan únicamente dentro de la ventana de movimiento. Ya no existe barrido sobre
# escena estática —que era la fuente de falsas alarmas (casco->persona, barrotes->robo)—
# así que no hace falta separar conceptos por tipo: el movimiento es el discriminador.
print("Conceptos (evaluados SOLO con movimiento):", sorted(set(prompt_labels)))

# Score contrastivo: en escena real, el score absoluto de CLIP sigue al CONTEXTO
# (cocina/mesa/objeto-en-mano) casi tanto como al objeto, lo que dispara falsas
# alarmas. Restar el mejor "distractor" (objeto/escena cotidiana) por parche
# cancela ese sesgo. Validado en imágenes reales de armas vs objetos de mano:
# falsas alarmas 78% -> 17% a igual recall. Configurable con context["distractor_prompts"].
#
# NOTA: se quitó "a photo of food on a table". Medido sobre cuchillos reales de
# COCO (mayoría en cocina), ese distractor colisionaba con los positivos —el
# cuchillo ESTÁ sobre una mesa con comida— y hundía el margen: AUC 0.465 (peor que
# azar) con él vs 0.739 sin él; recall en cocina 10% -> 24% a igual tasa de falsas
# alarmas. Como la prioridad es no perder amenazas, se elimina.
DEFAULT_DISTRACTORS = [
    "a photo of a smartphone",
    "a photo of a wallet",
    "a photo of a hand",
    "a person standing normally",
    "an empty room",
    "furniture",
]
_ctx_distractors = data.get("distractor_prompts") if isinstance(data, dict) else None
distractor_prompts = _ctx_distractors if (isinstance(_ctx_distractors, list) and _ctx_distractors) else DEFAULT_DISTRACTORS

# Distractores CONSCIENTES del concepto: si el usuario quiere detectar "persona",
# restar "a person standing normally"/"a photo of a hand" cancela el propio objetivo
# (el margen contrastivo se hunde y NUNCA detecta personas). Medido sobre imágenes con
# personas: con esos distractores detecta 4/35; sin ellos, 16/35. Se eliminan los
# distractores que colisionan con algún concepto que el usuario SÍ quiere detectar.
CONFLICTING_DISTRACTORS = {
    "persona": {"a person standing normally", "a photo of a hand"},
    "person": {"a person standing normally", "a photo of a hand"},
}
_active_targets = {normalize_word(w) for w in cleaned_blacklist}
_drop_distractors = set()
for _c in _active_targets:
    _drop_distractors |= CONFLICTING_DISTRACTORS.get(_c, set())
if _drop_distractors:
    distractor_prompts = [d for d in distractor_prompts if d not in _drop_distractors]
    print("Distractores eliminados por colisión con el objetivo:", sorted(_drop_distractors))

distractor_tokens = clip.tokenize(distractor_prompts, truncate=True).to(device)
with torch.no_grad():
    distractor_embeddings = model.encode_text(distractor_tokens)
    distractor_embeddings /= distractor_embeddings.norm(dim=-1, keepdim=True)
print("Distractores (contraste):", distractor_prompts)

# Stateful background subtractor for the cheap motion stage (main thread only).
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def pad_square_roi(box, w, h, pad=ROI_PADDING, min_size=MIN_ROI_SIZE):
    """Cuadra la caja de movimiento y le garantiza un lado mínimo (así un objeto
    pequeño se amplía limpio en vez de diluirse al reescalar a 224). Con pad=0 no
    añade contexto (el contexto extra baja el score contrastivo). El recorte
    cuadrado hace que el center-crop de CLIP deje fuera menos contenido. Recorta a
    los límites del frame. Devuelve coords en resolución completa."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # medio-lado = mayor de: caja+padding, o el mínimo exigido.
    half = max(bw * (1 + 2 * pad) / 2.0, bh * (1 + 2 * pad) / 2.0, min_size / 2.0)
    nx1 = int(max(0, cx - half)); ny1 = int(max(0, cy - half))
    nx2 = int(min(w, cx + half)); ny2 = int(min(h, cy + half))
    return (nx1, ny1, nx2, ny2)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0


def merge_boxes(boxes, iou_thr=0.3):
    """Fusiona cajas solapadas para no fragmentar un mismo objeto en varios recortes
    (cada fragmento perdería contexto y bajaría el score)."""
    merged = []
    for b in boxes:
        placed = False
        for i, m in enumerate(merged):
            if _iou(b, m) > iou_thr:
                merged[i] = (min(b[0], m[0]), min(b[1], m[1]),
                             max(b[2], m[2]), max(b[3], m[3]))
                placed = True
                break
        if not placed:
            merged.append(b)
    return merged


def get_motion_rois(frame):
    """Stage 1 (cheap): return bounding boxes of moving regions in full-res coords.

    Cada caja de movimiento se padea + cuadra + garantiza tamaño mínimo (region
    proposal) y luego se fusionan las solapadas, de modo que CLIP recibe recortes
    bien enmarcados en vez de cajas apretadas que diluyen los objetos pequeños."""
    small = cv2.resize(frame, None, fx=MOTION_DOWNSCALE, fy=MOTION_DOWNSCALE)
    mask = bg_subtractor.apply(small)
    # Drop MOG2 shadow pixels (value 127) and denoise.
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, motion_kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inv = 1.0 / MOTION_DOWNSCALE
    h, w = frame.shape[:2]
    rois = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_MOTION_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x1 = max(0, int(x * inv))
        y1 = max(0, int(y * inv))
        x2 = min(w, int((x + bw) * inv))
        y2 = min(h, int((y + bh) * inv))
        rois.append((area, (x1, y1, x2, y2)))
    # Keep the largest motion regions only, then frame them with context and merge.
    rois.sort(key=lambda r: r[0], reverse=True)
    boxes = [pad_square_roi(box, w, h) for _, box in rois[:MAX_ROIS]]
    return merge_boxes(boxes)


def sliding_window_rois(frame, patch_size=224, stride=192):
    """Full-frame tiling for the periodic safety-net scan (catches stationary targets).

    Coarser stride than a dense search — a target large enough to matter still lands in
    a tile — and the edges are always covered so nothing at the borders is missed.
    """
    h, w = frame.shape[:2]
    last_top = max(0, h - patch_size)
    last_left = max(0, w - patch_size)
    tops = list(range(0, last_top + 1, stride)) or [0]
    lefts = list(range(0, last_left + 1, stride)) or [0]
    if tops[-1] != last_top:
        tops.append(last_top)
    if lefts[-1] != last_left:
        lefts.append(last_left)
    rois = []
    for top in tops:
        for left in lefts:
            rois.append((left, top, min(left + patch_size, w), min(top + patch_size, h)))
    return rois


def multiscale_rois(frame):
    """Barrido SAHI multiescala: dos tamaños de ventana para no perder objetos
    pequeños (que a una sola escala quedan diluidos en el parche). Recupera
    recall en escenas amplias donde el objeto ocupa ~1% del frame."""
    return sliding_window_rois(frame, 256, 224) + sliding_window_rois(frame, 384, 320)


def enhance_frame(frame):
    """CLAHE contrast enhancement in LAB space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def run_detection(frame, rois, concept_embeddings=None, concept_labels=None):
    """Stage 2 (expensive): run CLIP only on the given ROIs, batched.

    concept_embeddings/concept_labels seleccionan QUÉ conceptos puntuar: en la ruta de
    movimiento se usan todos; en el barrido estático solo los de objeto físico (para no
    fabricar falsas alarmas con eventos abstractos). Por defecto usa todos (compat).

    Returns (enhanced_frame, best_score, detected, best_coords, best_label).
    """
    if concept_embeddings is None:
        concept_embeddings = text_embeddings
        concept_labels = prompt_labels
    enhanced = enhance_frame(frame)
    image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

    tensors = []
    coords_list = []
    for (x1, y1, x2, y2) in rois:
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        patch = image.crop((x1, y1, x2, y2))
        tensors.append(preprocess(patch))
        coords_list.append((x1, y1, x2, y2))

    if not tensors:
        return enhanced, 0.0, False, None, None

    batch = torch.stack(tensors).to(device)
    if use_half:
        batch = batch.half()
    with torch.no_grad():
        patch_embeddings = model.encode_image(batch)
        patch_embeddings /= patch_embeddings.norm(dim=-1, keepdim=True)
        # [parches, prompts] similitud con el concepto.
        sims = patch_embeddings @ concept_embeddings.T
        # [parches, distractores] similitud con objetos/escenas cotidianas.
        dsims = patch_embeddings @ distractor_embeddings.T
        # Score CONTRASTIVO: por cada parche, resta el mejor distractor. Así un
        # cuchillo puntúa alto pero una cocina/celular/mesa (sin arma) no.
        margins = sims - dsims.max(dim=1, keepdim=True).values

    # Mejor par (parche, prompt) por MARGEN contrastivo.
    flat_idx = int(torch.argmax(margins).item())
    num_prompts = margins.shape[1]
    patch_idx = flat_idx // num_prompts
    prompt_idx = flat_idx % num_prompts
    best_score = float(margins[patch_idx, prompt_idx].item())   # margen (confianza)
    best_sim = float(sims[patch_idx, prompt_idx].item())        # sim cruda (para log)
    best_coords = coords_list[patch_idx]
    # Report the user's word for the winning prompt, not the internal English prompt.
    best_label = concept_labels[prompt_idx]
    # Umbral específico del concepto ganador, aplicado sobre el margen.
    detected = best_score > threshold_for(best_label)
    if detected:
        print(f"  match {best_label}: margen {best_score:.3f} (sim {best_sim:.3f})")
    return enhanced, best_score, detected, best_coords, best_label


def draw_best_patch(frame, coords, score, label, detected):
    """Return a copy of frame with a green box on the highest-similarity patch."""
    annotated = frame.copy()
    if coords is not None:
        left, top, right, bottom = coords
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{label}: {score:.3f}{' *' if detected else ''}"
        cv2.putText(annotated, text, (left, max(top - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated

def format_full_time(ts):
    return f"{time.strftime('%H:%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

def reinitialize_capture():
    global cap
    print("Reinitializing stream...")
    cap.release()
    time.sleep(2)
    cap = open_rtsp_capture(rtsp_url, retries=3, delay_sec=2)
    if cap is None:
        # La cámara cayó de forma persistente: auto-terminar en vez de reintentar
        # para siempre gastando la instancia.
        terminate_self("RTSP inalcanzable tras reconexión")
        sys.exit(1)

def storeRegister(data):
    token = get_firebase_token()
    # Explicit timeout so a hung API can never block the I/O worker forever.
    conn = http.client.HTTPSConnection(
        "c038gkbfm8.execute-api.us-east-1.amazonaws.com", timeout=10
    )
    payload = json.dumps(data)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    conn.request("POST", "/default/storeRegister", payload, headers)
    res = conn.getresponse()
    print("Status:", res.status)
    response_data = res.read()
    print(response_data.decode("utf-8"))


def post_worker_event(payload):
    """POST autenticado al Lambda workerEvents (heartbeat / notify)."""
    token = get_firebase_token()
    conn = http.client.HTTPSConnection(WORKER_EVENTS_HOST, timeout=10)
    conn.request(
        "POST", "/", json.dumps(payload),
        {"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    res = conn.getresponse()
    res.read()
    return res.status


def notify_user(event_type, camera, score, detection_id):
    """Avisa al usuario por los canales que configuró (SMS/email)."""
    try:
        post_worker_event({
            "action": "notify",
            "owner_uid": owner_uid,
            "event_type": event_type,
            "camera": camera,
            "cosine_sim": round(float(score), 3),
            "detection_id": detection_id,
        })
    except Exception as e:
        print("notify_user error:", e)


def terminate_self(reason):
    """Auto-termina esta instancia EC2 (la cámara no conecta -> no seguir facturando)."""
    print(f"Auto-terminando la instancia: {reason}")
    iid = get_instance_id()
    if not iid:
        # Sin IMDS (p.ej. entorno local): salir para que el supervisor no reintente en vano.
        os._exit(3)
    try:
        boto3.client("ec2", region_name="us-east-1").terminate_instances(InstanceIds=[iid])
    except Exception as e:
        print("terminate_self error:", e)
        os._exit(3)


def heartbeat_loop():
    """Reporta 'vivo' cada 30s para que la consola muestre el estado real."""
    while True:
        try:
            post_worker_event({
                "action": "heartbeat",
                "device_id": device_id,
                "owner_uid": owner_uid,
                "camera_name": data.get("camera_name", ""),
                "status": "running",
            })
        except Exception as e:
            print("heartbeat error:", e)
        time.sleep(30)

def upload_frame_to_s3(frame, ts, detection_score, coords=None, detection_id=None):
    timestr = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(ts))
    millis = int((ts % 1) * 1000)
    if coords is not None:
        left, top, right, bottom = coords
        coord_str = f"_{left}-{top}-{right}-{bottom}"
    else:
        coord_str = ""
    uuid_str = f"_{detection_id}" if detection_id else ""
    filename = f"{timestr}_{millis:03d}_score-{detection_score:.3f}{coord_str}{uuid_str}.jpg"
    key = S3_PREFIX + filename
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print("Failed to encode frame as JPEG, not uploading to S3")
        return None
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=buffer.tobytes(),
            ContentType="image/jpeg",
        )
        print(f"Uploaded frame to s3://{S3_BUCKET_NAME}/{key}")
        return key
    except Exception as e:
        print("Error uploading frame to S3:", e)
        return None


def handle_alert(frame, ts, score, coords, label):
    """Runs on the I/O thread pool: upload frame + register the event. Never blocks capture."""
    try:
        detection_id = str(uuid.uuid4())
        image_key = upload_frame_to_s3(frame, ts, score, coords, detection_id)
        storeRegister({
            "cammera": data.get('camera_name', 'entrance'),
            "clientId": data.get('client_id', 1),
            "event_type": label,
            "detection_id": detection_id,
            "cosine_sim": score,
            "image_key": image_key,
            "owner_uid": owner_uid,
        })
        # Avisar al usuario por su canal configurado (email/SMS).
        notify_user(label, data.get('camera_name', 'entrance'), score, detection_id)
    except Exception as e:
        print("Alert handler error:", e)


# Heartbeat en segundo plano (daemon): la consola ve el estado real del worker.
threading.Thread(target=heartbeat_loop, daemon=True).start()

# Open the RTSP stream
cap = open_rtsp_capture(rtsp_url)
if cap is None:
    # La cámara no conecta: no dejar la instancia encendida facturando sin hacer
    # nada. Se auto-termina en vez de morir y quedar colgada.
    print(
        f"Failed to open RTSP stream: {mask_rtsp_url(rtsp_url)}. "
        "Revisa el túnel/puerto y que ffmpeg esté instalado."
    )
    terminate_self("RTSP inalcanzable en el arranque")
    sys.exit(1)

print("Processing frames from RTSP stream...")

# One worker runs CLIP on the latest motion frame; I/O uploads run fully in the background.
detection_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
pending_detection = None
pending_meta = None  # (frame, ts) tied to the in-flight detection

frame_counter = 0
frames_this_second = 0
processed_counter = 0
read_failures = 0   # consecutive failed reads; reconnect only after MAX_READ_FAILURES
last_second_time = time.time()
last_detection_time = 0.0
active_until_time = 0.0        # detección ACTIVA mientras frame_time <= este instante
notified_this_window = False   # una sola notificación por ventana de movimiento
last_motion_rois = None        # última zona de movimiento (se re-puntúa durante la ventana)
consecutive_detection_count = 0
cosine_history = deque(maxlen=30)   # bounded: no unbounded growth / GC churn
last_annotated = None  # most recent detection overlay, kept for continuous display

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # A single dropped read is normal over a jittery ngrok/RTSP tunnel.
            # Only tear down and reconnect after many consecutive failures — rapid
            # reconnects leave zombie sessions on the camera (TP-Link caps concurrent
            # RTSP sessions) and exhaust its pool, making things worse.
            read_failures += 1
            if read_failures < MAX_READ_FAILURES:
                time.sleep(0.05)
                continue
            print(f"{read_failures} consecutive read failures, reconnecting...")
            reinitialize_capture()
            read_failures = 0
            time.sleep(1)  # let the camera reclaim the old session before streaming
            continue
        read_failures = 0  # healthy read resets the failure streak

        frame_counter += 1
        frames_this_second += 1
        frame_time = time.time()

        # --- Collect a finished detection (single in-flight future, no leak) ---
        if pending_detection is not None and pending_detection.done():
            det_frame, det_ts = pending_meta
            try:
                enhanced, score, detected, coords, label = pending_detection.result()
            except Exception as e:
                print("Error processing frame:", e)
                enhanced, score, detected, coords, label = det_frame, 0.0, False, None, None
            pending_detection = None
            pending_meta = None
            processed_counter += 1

            cosine_history.append((score, det_ts))
            print(f"[{format_full_time(det_ts)}] score: {score:.3f} | "
                  f"match: {label} | detected: {detected}")

            warmed_up = frame_counter > WARMUP_FRAMES
            if detected and warmed_up:
                consecutive_detection_count += 1
                print("Consecutive detections:", consecutive_detection_count)
                # UNA sola notificación por ventana de movimiento (dedupe del burst):
                # tras ALERT_THRESHOLD frames positivos seguidos se avisa una vez y no
                # se vuelve a avisar hasta que un nuevo movimiento abra otra ventana.
                if consecutive_detection_count >= ALERT_THRESHOLD and not notified_this_window:
                    notified_this_window = True
                    # Fire-and-forget: S3 + API never block the capture loop.
                    io_executor.submit(handle_alert, enhanced.copy(), det_ts, score, coords, label)
            else:
                consecutive_detection_count = 0

            if SHOW_WINDOWS:
                last_annotated = draw_best_patch(enhanced, coords, score, label, detected)

        # --- Stage 1 (PRIMERA CAPA): puerta de movimiento. La detección está APAGADA
        #     salvo que el movimiento abra/re-arme una ventana de DETECTION_WINDOW_SECONDS. ---
        rois = get_motion_rois(frame)
        if rois:
            last_motion_rois = rois
            if frame_time > active_until_time:
                # Estaba apagada -> este movimiento ABRE una nueva ventana (reset de aviso).
                notified_this_window = False
                print(f"[movimiento] ventana de detección abierta ({DETECTION_WINDOW_SECONDS:.0f}s)")
            # Cada movimiento RE-ARMA el minuto: la ventana dura hasta 60s tras el último.
            active_until_time = frame_time + DETECTION_WINDOW_SECONDS

        if time.time() - last_second_time >= 1.0:
            busy = pending_detection is not None and not pending_detection.done()
            last_score = cosine_history[-1][0] if cosine_history else 0.0
            print(f"FPS in: {frames_this_second} | detections/s: {processed_counter} | "
                  f"cosine sim: {last_score:.3f} | "
                  f"motion ROIs: {len(rois)} | worker busy: {busy}")
            frames_this_second = 0
            processed_counter = 0
            last_second_time = time.time()

        # --- Stage 2: dentro de la ventana de movimiento se corre CLIP sobre la ZONA del
        #     movimiento (rois actuales, o la última zona conocida si MOG2 la pierde un
        #     instante). Fuera de la ventana, la detección está apagada. ---
        detection_active = frame_time <= active_until_time
        scan_rois = (rois or last_motion_rois) if detection_active else None

        worker_busy = pending_detection is not None and not pending_detection.done()
        throttled = frame_time - last_detection_time < MIN_DETECTION_INTERVAL

        if scan_rois and not worker_busy and not throttled:
            last_detection_time = frame_time
            # Todos los conceptos se evalúan (default): el movimiento ya es el filtro.
            pending_detection = detection_executor.submit(run_detection, frame.copy(), scan_rois)
            pending_meta = (frame.copy(), frame_time)

        # --- Display every frame so the feed is visible even with no motion/detections ---
        if SHOW_WINDOWS:
            live = frame.copy()
            for (x1, y1, x2, y2) in rois:
                cv2.rectangle(live, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange = motion
            cv2.imshow("Live (raw + motion)", live)
            if last_annotated is not None:
                cv2.imshow("Detection (enhanced + best patch)", last_annotated)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("'q' pressed, stopping...")
                break

except KeyboardInterrupt:
    print("Stopping processing...")

finally:
    cap.release()
    detection_executor.shutdown()
    io_executor.shutdown()
    if SHOW_WINDOWS:
        cv2.destroyAllWindows()
