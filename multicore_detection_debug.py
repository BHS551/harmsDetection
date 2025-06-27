import cv2
import time
from PIL import Image
import torch
import clip
from twilio.rest import Client
import numpy as np
import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt
import subprocess

# === Twilio configuration ===
account_sid     = ""  # Replace with your Account SID
auth_token      = "c7ea6fdcd8bd75d3b9ab6ec2df33e1f5"
twilio_phone    = "+14793178516"
recipient_phone = "+573043566310"
client = Client(account_sid, auth_token)

def send_sms_alert(message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

# === Camera and Detection Configuration ===
rtsp_url = "rtsp://bhsentrance:mainSecurePass1@8.tcp.ngrok.io:15666/stream1"

# Set device and load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"is cuda available: [{torch.cuda.is_available()}]")
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute the text embedding for the query "person"
text_prompt = "person walking, black and white scene"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def sliding_window_detection(image, patch_size=224, stride=112, threshold=0.260):
    width, height = image.size
    patches, orig_patches, coords_list = [], [], []
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            orig_patches.append(patch)
            patches.append(preprocess(patch))
            coords_list.append((left, top, left + patch_size, top + patch_size))
    if not patches:
        return 0.0, False, None, None

    batch_tensor = torch.stack(patches).to(device)
    with torch.no_grad():
        patch_embeddings = model.encode_image(batch_tensor)
        patch_embeddings /= patch_embeddings.norm(dim=-1, keepdim=True)

    cosine_sim = (patch_embeddings @ text_embedding.T).squeeze()
    max_val, max_index = torch.max(cosine_sim, dim=0)
    max_cosine = max_val.item()
    best_patch = orig_patches[max_index]
    best_coords = coords_list[max_index]
    detection_found = max_cosine > threshold
    return max_cosine, detection_found, best_patch, best_coords

def process_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    image = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
    return (*sliding_window_detection(image, patch_size=224, stride=112, threshold=0.268), enhanced_frame)

# Helper functions
def format_full_time(ts):
    return f"{time.strftime('%H:%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

def format_short_time(ts):
    return f"{time.strftime('%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

# Thread pool setup
max_workers = max(1, multiprocessing.cpu_count() - 2)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

# Loop variables
frame_counter = 0
ignored_frames = 0
last_second_time = time.time()
cosine_history = []
patch_history = []
last_taken_frame = None
last_processed_frame = None
last_best_coords = None
consecutive_detection_count = 0
futures = []
future_timestamps = {}

# Plot setup
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_title("Cosine Similarity History")
ax.set_xlabel("Frame")
ax.set_ylabel("Cosine Similarity")
ax.set_xlim(0, 30)
ax.set_ylim(0.230, 0.300)

def update_plot(history):
    history_vals = [rec[0] for rec in history[-30:]]
    x = np.arange(len(history_vals))
    line.set_xdata(x)
    line.set_ydata(history_vals)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.230, 0.300)
    fig.canvas.draw()
    fig.canvas.flush_events()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        frame_counter += 1
        frame_time = time.time()

        # Process every 2nd frame
        if frame_counter % 2 != 0:
            ignored_frames += 1
            continue

        active_futures = [f for f in futures if not f.done()]
        num_active = len(active_futures)
        available_workers = max_workers - num_active

        # Log GPU or CPU usage every second
        if time.time() - last_second_time >= 1.0:
            if device == "cuda":
                # Query nvidia-smi for GPU stats
                try:
                    result = subprocess.run([
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"
                    ], capture_output=True, text=True)
                    gpu_util, mem_used, mem_total = [int(x.strip()) for x in result.stdout.split(",")]
                    print(f"[{format_full_time(time.time())}] GPU Util: {gpu_util}% | Mem: {mem_used}/{mem_total} MiB | Ignored frames/sec: {ignored_frames}")
                except Exception as e:
                    print(f"Failed to query nvidia-smi: {e}")
            else:
                print(f"[{format_full_time(time.time())}] Processors in use: {num_active} | Available: {available_workers} | Ignored frames/sec: {ignored_frames}")
            ignored_frames = 0
            last_second_time = time.time()

        if available_workers <= 0:
            continue

        last_taken_frame = frame.copy()
        future = executor.submit(process_frame, frame.copy())
        futures.append(future)
        future_timestamps[future] = frame_time

        finished = []
        for f in futures:
            if f.done():
                ts = future_timestamps.pop(f, time.time())
                try:
                    cosine_sim, knife_detected, best_patch, best_coords, processed_frame = f.result()
                except Exception as e:
                    print("Error processing frame:", e)
                    finished.append(f)
                    continue

                last_processed_frame = processed_frame.copy()
                last_best_coords = best_coords
                cosine_history.append((cosine_sim, ts))
                if len(cosine_history) > 30:
                    cosine_history = cosine_history[-30:]
                print(f"[{format_full_time(ts)}] Cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {knife_detected}")

                if knife_detected:
                    consecutive_detection_count += 1
                else:
                    consecutive_detection_count = 0

                patch_history.append((cosine_sim, ts, best_patch, best_coords, knife_detected))
                if len(patch_history) > 30:
                    patch_history = patch_history[-30:]
                finished.append(f)

        futures = [f for f in futures if f not in finished]

        # Display best patch and frames
        best_record = None
        over_threshold = [rec for rec in patch_history if rec[4]]
        if over_threshold:
            best_record = max(over_threshold, key=lambda r: r[0])
        elif patch_history:
            best_record = max(patch_history, key=lambda r: r[0])

        if best_record:
            patch_cv2 = cv2.cvtColor(np.array(best_record[2]), cv2.COLOR_RGB2BGR)
            overlay_text = f"{best_record[0]:.3f} {format_short_time(best_record[1])}"
            cv2.putText(patch_cv2, overlay_text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Best Patch', patch_cv2)

        if last_processed_frame is not None and last_best_coords:
            display_frame = last_processed_frame.copy()
            cv2.rectangle(display_frame, (last_best_coords[0], last_best_coords[1]), (last_best_coords[2], last_best_coords[3]), (0,255,0), 2)
            cv2.imshow('Last Processed Frame', display_frame)
        elif last_processed_frame:
            cv2.imshow('Last Processed Frame', last_processed_frame)

        if last_taken_frame is not None:
            cv2.imshow('Last Taken Frame', last_taken_frame)

        update_plot(cosine_history)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
