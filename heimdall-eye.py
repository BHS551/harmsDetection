import cv2
import time
import os
import re
import sys
from collections import deque
from PIL import Image
import torch
import clip
from twilio.rest import Client
import numpy as np
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

# === Twilio configuration ===
account_sid = ""
auth_token = ""
twilio_phone = "+18164767447"
recipient_phone = "+573043566310"

# === S3 Configuration ===
S3_BUCKET_NAME = "detection-frames-tests"
S3_PREFIX = "cameras/"
client = Client(account_sid, auth_token)

s3_client = boto3.client(
    "s3",
    region_name="us-east-1",
)

# === Runtime tuning (override via env vars without touching code) ===
# Live debug windows: OFF by default so headless/EC2 hosts don't crash on cv2.imshow.
SHOW_WINDOWS = 0
# Max detections-per-second we actually run CLIP on (time-based, FPS-independent).
MIN_DETECTION_INTERVAL = float(os.environ.get("HEIMDALL_MIN_INTERVAL", "0.4"))
# Consecutive positive frames required before firing an alert (debounces false positives).
ALERT_THRESHOLD = int(os.environ.get("HEIMDALL_ALERT_THRESHOLD", "3"))
# Minimum seconds between two alerts for the same camera (avoids spamming S3/API).
ALERT_COOLDOWN = float(os.environ.get("HEIMDALL_ALERT_COOLDOWN", "10"))
# Frames to let the background model warm up before trusting motion (skips alerts).
WARMUP_FRAMES = int(os.environ.get("HEIMDALL_WARMUP_FRAMES", "30"))
# CLIP cosine-similarity threshold for a positive match.
DETECTION_THRESHOLD = float(os.environ.get("HEIMDALL_THRESHOLD", "0.27"))
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


def send_sms_alert(message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

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
rtsp_url = data['rtsp_path']
owner_uid = data.get('owner_uid', '')

# Prohibited items to detect come from the context blacklist (e.g. ["knife"]).
detection_blacklist = data.get("detection_blacklist") or ["person"]
print("Detection targets (blacklist):", detection_blacklist)

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# FP16 on GPU: ~30% faster inference and half the VRAM. CPU stays FP32 (half is slow there).
use_half = device == "cuda"
if use_half:
    model = model.half()
model.eval()

# Precompute one text embedding per blacklisted item. A frame matches if ANY item scores high.
text_tokens = clip.tokenize(detection_blacklist).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

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
        # [num_patches, num_prompts] cosine similarities.
        sims = patch_embeddings @ text_embeddings.T

    # Best (patch, prompt) pair across the whole batch.
    flat_idx = int(torch.argmax(sims).item())
    num_prompts = sims.shape[1]
    patch_idx = flat_idx // num_prompts
    prompt_idx = flat_idx % num_prompts
    best_score = float(sims[patch_idx, prompt_idx].item())
    best_coords = coords_list[patch_idx]
    best_label = detection_blacklist[prompt_idx]
    detected = best_score > DETECTION_THRESHOLD
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
        raise Exception(f"Failed to reconnect RTSP stream: {mask_rtsp_url(rtsp_url)}")

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
    except Exception as e:
        print("Alert handler error:", e)


# Open the RTSP stream
cap = open_rtsp_capture(rtsp_url)
if cap is None:
    raise Exception(
        f"Failed to open RTSP stream: {mask_rtsp_url(rtsp_url)}. "
        "Check that the ngrok tunnel is running, the port matches context.json, "
        "and ffmpeg is installed on the host (apt install ffmpeg)."
    )

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
patch_history = deque(maxlen=30)
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
            patch_history.append((score, det_ts, coords, label, detected))
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
            scan_rois = sliding_window_rois(frame)   # whole frame
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
