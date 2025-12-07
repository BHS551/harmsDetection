import cv2
import time
from PIL import Image
import torch
import clip
from twilio.rest import Client
import numpy as np
import concurrent.futures
import multiprocessing
import http.client
import json
import boto3
import http.client
import uuid

# === Twilio configuration ===
account_sid = ""  # Replace with your Account SID
auth_token = ""       # Replace with your Auth Token
twilio_phone = "+18164767447"                         # Your Twilio phone number
recipient_phone = "+573043566310"                     # The number to alert

# === S3 Configuration ===
S3_BUCKET_NAME = "detection-frames-tests"
S3_PREFIX = "cameras/"
client = Client(account_sid, auth_token)

s3_client = boto3.client(
    "s3",
    aws_access_key_id="",
    aws_secret_access_key="",
    region_name="us-east-1",
)

def send_sms_alert(message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

# === Camera and Detection Configuration ===
# url for entrance camera 
rtsp_url = "rtsp://test123:123456789@4.tcp.ngrok.io:18172/stream1"

# url for studio camera
#rtsp_url = "rtsp://admin551:123456789@2.tcp.ngrok.io:15165/stream1"

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
    and returns the maximum cosine similarity, a flag indicating whether it exceeds
    the threshold, the best patch, and its coordinates.
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
    convert to a PIL image, run sliding window detection, and return
    the enhanced frame along with detection results.
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

# Helper function for time formatting.
def format_full_time(ts):
    return f"{time.strftime('%H:%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

# Function to reinitialize the capture in case of failure.
def reinitialize_capture():
    global cap
    print("Reinitializing stream...")
    cap.release()
    time.sleep(2)  # wait a bit before reconnecting
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

def storeRegister(data):
    conn = http.client.HTTPSConnection("c038gkbfm8.execute-api.us-east-1.amazonaws.com")
    payload = json.dumps(data)
    headers = {
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/default/storeRegister", payload, headers)
    res = conn.getresponse()
    response_data = res.read()
    print(response_data.decode("utf-8"))

def upload_frame_to_s3(frame, ts, detection_score, coords=None, detection_id=None):
    """
    Uploads a single frame (numpy array, BGR) to S3 as a JPEG.
    ts: timestamp (float)
    detection_score: cosine similarity score for logging
    coords: (left, top, right, bottom) for the best patch (optional, used to name file)
    """
    # Build a nice filename: 2025-12-07_13-45-23_123ms_score-0.812.jpg
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

    # OpenCV frame is BGR, encode as JPEG
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print("Failed to encode frame as JPEG, not uploading to S3")
        return

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=buffer.tobytes(),
            ContentType="image/jpeg",
        )
        print(f"Uploaded frame to s3://{S3_BUCKET_NAME}/{key}")
    except Exception as e:
        print("Error uploading frame to S3:", e)

# Open the RTSP stream using the FFMPEG backend.
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

# Use available cores minus 2 for processing.
max_workers = max(1, multiprocessing.cpu_count() - 2)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

frame_counter = 0
ignored_frames = 0
last_second_time = time.time()
cosine_history = []  # List of (cosine_sim, timestamp)
patch_history = []   # List of tuples (cosine_sim, timestamp, best_patch, best_coords, knife_detected)

consecutive_detection_count = 0  
alert_threshold = 1  # Number of consecutive detections required to send an alert.

futures = []
future_timestamps = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame, attempting to reconnect...")
            reinitialize_capture()
            continue

        frame_counter += 1
        frame_time = time.time()

        # Process only 1 frame every 17.
        if frame_counter % 17 != 0:
            ignored_frames += 1
            continue

        active_futures = [f for f in futures if not f.done()]
        num_active = len(active_futures)
        available_workers = max_workers - num_active

        if time.time() - last_second_time >= 1.0:
            print(f"Processors in use: {num_active} | Available: {available_workers} | Ignored frames/sec: {ignored_frames}")
            ignored_frames = 0
            last_second_time = time.time()

        if available_workers <= 0:
            print("No available processors, skipping frame")
            continue

        # Submit frame processing.
        future = executor.submit(process_frame, frame.copy())
        futures.append(future)
        future_timestamps[future] = frame_time

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

                # Log the detection results.
                cosine_history.append((cosine_sim, ts))
                if len(cosine_history) > 30:
                    cosine_history = cosine_history[-30:]
                print(f"[{format_full_time(ts)}] Cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {knife_detected}")

                if knife_detected:
                    consecutive_detection_count += 1
                    print("Consecutive detections:", consecutive_detection_count)
                    # Optionally, send an SMS alert:
                    if consecutive_detection_count >= alert_threshold:
                        detection_id = str(uuid.uuid4())
                        # send_sms_alert(f"Alert: {text_prompt} detected at {format_full_time(ts)}")
                        storeRegister({
                            "cammera": "entrance",
                            "clientId": 1,
                            "event_type": "manual_test",
                            "detection_id": detection_id,
                            "cosine_sim": cosine_sim,
                        })
                        upload_frame_to_s3(
                            processed_frame,   # full enhanced frame
                            ts,                # timestamp
                            cosine_sim,        # detection score
                            best_coords,        # coordinates of best patch (optional)
                            detection_id
                        )
                else:
                    consecutive_detection_count = 0

                patch_history.append((cosine_sim, ts, best_patch, best_coords, knife_detected))
                if len(patch_history) > 30:
                    patch_history = patch_history[-30:]
                finished.append(f)
        futures = [f for f in futures if f not in finished]

except KeyboardInterrupt:
    print("Stopping processing...")

finally:
    cap.release()
    executor.shutdown()
