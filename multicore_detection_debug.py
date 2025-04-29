import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import torch
import clip
from twilio.rest import Client
import numpy as np
import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt

# === Twilio configuration ===
account_sid = "ACCOUNT_SID"  # Replace with your Account SID
auth_token = "c7ea6fdcd8bd75d3b9ab6ec2df33e1f5"       # Replace with your Auth Token
twilio_phone = "+14793178516"                         # Your Twilio phone number
recipient_phone = "+573043566310"                     # The number to alert

client = Client(account_sid, auth_token)

def send_sms_alert(message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

# === Camera and Detection Configuration ===
rtsp_url = "rtsp://admin551:123456789@0.tcp.ngrok.io:12644/stream1"  # Update with your camera's credentials

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute the text embedding for the query "knife"
text_prompt = "knife"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def sliding_window_detection(image, patch_size=224, stride=112, threshold=0.260):
    """
    Slides a window across the image, computes the CLIP embeddings for each patch,
    and returns the maximum cosine similarity (from the best matching patch),
    a flag indicating whether that maximum exceeds the threshold,
    the best patch (as a PIL image), and its coordinates (left, top, right, bottom).
    """
    width, height = image.size
    patches = []
    orig_patches = []
    coords_list = []
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
    """
    Process a single frame: apply CLAHE contrast enhancement,
    convert to PIL image, run sliding window detection (using only the best patch),
    and return the enhanced frame with detection result, the best patch, and its coordinates.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    image = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
    cosine_sim, knife_detected, best_patch, best_coords = sliding_window_detection(
        image, patch_size=224, stride=112, threshold=0.268)
    return enhanced_frame, cosine_sim, knife_detected, best_patch, best_coords

# Helper functions for time formatting.
def format_full_time(ts):
    return f"{time.strftime('%H:%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

def format_short_time(ts):
    return f"{time.strftime('%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

# Open the RTSP stream using the FFMPEG backend.
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

# Use available cores minus 2 for processing.
max_workers = max(1, multiprocessing.cpu_count() - 2)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

# Counters and history.
frame_counter = 0
ignored_frames = 0
last_second_time = time.time()
cosine_history = []  # List of (cosine_sim, timestamp)

# Patch history: list of tuples (cosine_sim, timestamp, best_patch, best_coords, knife_detected).
patch_history = []

# Variables to hold the last taken and last processed frames.
last_taken_frame = None
last_processed_frame = None
last_best_coords = None  # from last processed frame

# SMS and detection settings.
consecutive_detection_count = 0  
alert_threshold = 1  # Number of consecutive detections required to send an alert.

# For futures management.
futures = []
future_timestamps = {}  # Map future -> frame capture timestamp.

# Set up matplotlib for live plotting.
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_title("Cosine Similarity History")
ax.set_xlabel("Frame")
ax.set_ylabel("Cosine Similarity")
ax.set_xlim(0, 30)
ax.set_ylim(0.230, 0.300)

def update_plot(history):
    # Use only the last 30 values.
    history_vals = [rec[0] for rec in history[-30:]]
    x = np.arange(len(history_vals))
    line.set_xdata(x)
    line.set_ydata(history_vals)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.230, 0.300)
    fig.canvas.draw()
    fig.canvas.flush_events()

def pil_to_cv2(pil_img):
    """Convert a PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Font for overlaying text (using cv2.putText).
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        frame_counter += 1
        frame_time = time.time()  # capture time for this frame

        # Process only 1 frame every 17.
        if frame_counter % 17 != 0:
            ignored_frames += 1
            continue

        # Check current active futures.
        active_futures = [f for f in futures if not f.done()]
        num_active = len(active_futures)
        available_workers = max_workers - num_active

        # Log processor usage and ignored frames every second.
        if time.time() - last_second_time >= 1.0:
            print(f"Processors in use: {num_active} | Available: {available_workers} | Ignored frames/sec: {ignored_frames}")
            ignored_frames = 0
            last_second_time = time.time()

        # If no available worker, skip this frame.
        if available_workers <= 0:
            print("No available processors, skipping frame")
            continue

        # Save the last taken frame (raw frame for display).
        last_taken_frame = frame.copy()
        # Submit frame processing and store its timestamp.
        future = executor.submit(process_frame, frame.copy())
        futures.append(future)
        future_timestamps[future] = frame_time

        # Check completed futures.
        finished = []
        for f in futures:
            if f.done():
                ts = future_timestamps.pop(f, time.time())
                try:
                    processed_frame, cosine_sim, knife_detected, best_patch, best_coords = f.result()
                except Exception as e:
                    print("Error processing frame:", e)
                    finished.append(f)
                    continue

                last_processed_frame = processed_frame.copy()
                last_best_coords = best_coords
                # Append to cosine history with timestamp.
                cosine_history.append((cosine_sim, ts))
                # Keep only the last 30 values.
                if len(cosine_history) > 30:
                    cosine_history = cosine_history[-30:]
                print(f"[{format_full_time(ts)}] Cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {knife_detected}")

                if knife_detected:
                    consecutive_detection_count += 1
                    print("Consecutive detections:", consecutive_detection_count)
                else:
                    consecutive_detection_count = 0

                # Append to patch history.
                patch_history.append((cosine_sim, ts, best_patch, best_coords, knife_detected))
                if len(patch_history) > 30:
                    patch_history = patch_history[-30:]
                finished.append(f)
        futures = [f for f in futures if f not in finished]

        # Determine the best patch for display in the fourth window.
        # If any patch in the history is over threshold, select the one with the highest cosine among those.
        # Otherwise, select the patch with the highest cosine among all.
        best_record = None
        over_threshold = [rec for rec in patch_history if rec[4]]
        if over_threshold:
            best_record = max(over_threshold, key=lambda r: r[0])
        elif patch_history:
            best_record = max(patch_history, key=lambda r: r[0])
        
        if best_record is not None:
            best_patch_img = best_record[2]  # PIL image
            best_ts = best_record[1]
            best_cosine = best_record[0]
            # Overlay text (using short format: MM:SS mmm) on the patch image.
            patch_cv2 = pil_to_cv2(best_patch_img)
            overlay_text = f"{best_cosine:.3f} {format_short_time(best_ts)}"
            cv2.putText(patch_cv2, overlay_text, (5, 25), font, 0.8, (0, 255, 0), 2)
            cv2.imshow('Best Patch', patch_cv2)

        # On the last processed frame, draw a green rectangle where the best patch was found.
        if last_processed_frame is not None and last_best_coords is not None:
            display_frame = last_processed_frame.copy()
            cv2.rectangle(display_frame, (last_best_coords[0], last_best_coords[1]), (last_best_coords[2], last_best_coords[3]), (0,255,0), 2)
            cv2.imshow('Last Processed Frame', display_frame)
        elif last_processed_frame is not None:
            cv2.imshow('Last Processed Frame', last_processed_frame)

        # Display the last taken frame.
        if last_taken_frame is not None:
            cv2.imshow('Last Taken Frame', last_taken_frame)

        # Update the cosine similarity history plot.
        update_plot(cosine_history)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
