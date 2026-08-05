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
# Tolerate this many consecutive failed reads (jittery ngrok/RTSP) before reconnecting.
# A single dropped read is normal; reconnecting on every one just thrashes the tunnel.
MAX_READ_FAILURES = int(os.environ.get("HEIMDALL_MAX_READ_FAILURES", "30"))
# Seconds to wait for the first decodable frame within ONE open attempt. H264 only
# decodes from a keyframe and TP-Link GOPs are long, so the first frame can lag a few
# seconds — wait it out instead of tearing down and re-handshaking.
RTSP_OPEN_TIMEOUT = float(os.environ.get("HEIMDALL_OPEN_TIMEOUT", "12"))
# Safety net: even with zero motion, run a FULL-frame sliding-window scan this often
# (seconds). Motion-gating alone would miss a stationary target (knife on a table, a
# person standing still) once MOG2 learns it into the background. 0 disables the sweep.
FULL_SCAN_INTERVAL = float(os.environ.get("HEIMDALL_FULL_SCAN_INTERVAL", "3"))



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
    "cuchillo": ["a photo of a knife"],
    "knife": ["a photo of a knife"],
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
    "cuchillo": 0.02,
    "knife": 0.02,
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

# Score contrastivo: en escena real, el score absoluto de CLIP sigue al CONTEXTO
# (cocina/mesa/objeto-en-mano) casi tanto como al objeto, lo que dispara falsas
# alarmas. Restar el mejor "distractor" (objeto/escena cotidiana) por parche
# cancela ese sesgo. Validado en imágenes reales de armas vs objetos de mano:
# falsas alarmas 78% -> 17% a igual recall. Configurable con context["distractor_prompts"].
DEFAULT_DISTRACTORS = [
    "a photo of a smartphone",
    "a photo of a wallet",
    "a photo of a hand",
    "a person standing normally",
    "an empty room",
    "furniture",
    "a photo of food on a table",
]
_ctx_distractors = data.get("distractor_prompts") if isinstance(data, dict) else None
distractor_prompts = _ctx_distractors if (isinstance(_ctx_distractors, list) and _ctx_distractors) else DEFAULT_DISTRACTORS
distractor_tokens = clip.tokenize(distractor_prompts, truncate=True).to(device)
with torch.no_grad():
    distractor_embeddings = model.encode_text(distractor_tokens)
    distractor_embeddings /= distractor_embeddings.norm(dim=-1, keepdim=True)
print("Distractores (contraste):", distractor_prompts)

# Stateful background subtractor for the cheap motion stage (main thread only).
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def get_motion_rois(frame):
    """Stage 1 (cheap): return bounding boxes of moving regions in full-res coords."""
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
    # Keep the largest motion regions only.
    rois.sort(key=lambda r: r[0], reverse=True)
    return [box for _, box in rois[:MAX_ROIS]]


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


def run_detection(frame, rois):
    """Stage 2 (expensive): run CLIP only on the motion ROIs, batched.

    Returns (enhanced_frame, best_score, detected, best_coords, best_label).
    """
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
        sims = patch_embeddings @ text_embeddings.T
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
    best_label = prompt_labels[prompt_idx]
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
last_full_scan = 0.0   # 0 => a full sweep is due immediately on the first frame
last_alert_time = 0.0
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
                if (consecutive_detection_count >= ALERT_THRESHOLD
                        and frame_time - last_alert_time >= ALERT_COOLDOWN):
                    last_alert_time = frame_time
                    # Fire-and-forget: S3 + API never block the capture loop.
                    io_executor.submit(handle_alert, enhanced.copy(), det_ts, score, coords, label)
            else:
                consecutive_detection_count = 0

            if SHOW_WINDOWS:
                last_annotated = draw_best_patch(enhanced, coords, score, label, detected)

        # --- Stage 1: cheap motion gate on every frame (keeps bg model current) ---
        rois = get_motion_rois(frame)

        if time.time() - last_second_time >= 1.0:
            busy = pending_detection is not None and not pending_detection.done()
            last_score = cosine_history[-1][0] if cosine_history else 0.0
            print(f"FPS in: {frames_this_second} | detections/s: {processed_counter} | "
                  f"cosine sim: {last_score:.3f} | "
                  f"motion ROIs: {len(rois)} | worker busy: {busy}")
            frames_this_second = 0
            processed_counter = 0
            last_second_time = time.time()

        # --- Stage 2: motion gives a fast path; a periodic full-frame sweep is the
        #     safety net so a stationary target is still caught even with no motion. ---
        full_scan_due = FULL_SCAN_INTERVAL > 0 and frame_time - last_full_scan >= FULL_SCAN_INTERVAL
        if full_scan_due:
            scan_rois = multiscale_rois(frame)   # barrido SAHI multiescala
        elif rois:
            scan_rois = rois                         # motion regions only
        else:
            scan_rois = None                         # nothing to scan this frame

        worker_busy = pending_detection is not None and not pending_detection.done()
        # A due full sweep bypasses the motion throttle so it never gets starved.
        throttled = (frame_time - last_detection_time < MIN_DETECTION_INTERVAL) and not full_scan_due

        if scan_rois is not None and not worker_busy and not throttled:
            last_detection_time = frame_time
            if full_scan_due:
                last_full_scan = frame_time
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
