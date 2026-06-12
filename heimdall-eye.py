import cv2
import time
import os
import sys
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

def send_sms_alert(message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

# === Camera and Detection Configuration ===
rtsp_url = data['rtsp_path']
owner_uid = data.get('owner_uid', '')

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute the text embedding for the query "knife"
text_prompt = "knife"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def sliding_window_detection(image, patch_size=224, stride=112, threshold=0.29):
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
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    image = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
    cosine_sim, knife_detected, best_patch, best_coords = sliding_window_detection(
        image, patch_size=224, stride=112, threshold=0.29)
    return enhanced_frame, cosine_sim, knife_detected, best_patch, best_coords

def format_full_time(ts):
    return f"{time.strftime('%H:%M:%S', time.localtime(ts))} {int((ts % 1)*1000):03d}"

def reinitialize_capture():
    global cap
    print("Reinitializing stream...")
    cap.release()
    time.sleep(2)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

def storeRegister(data):
    token = get_firebase_token()
    conn = http.client.HTTPSConnection("c038gkbfm8.execute-api.us-east-1.amazonaws.com")
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

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

max_workers = max(1, multiprocessing.cpu_count() - 2)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

frame_counter = 0
ignored_frames = 0
last_second_time = time.time()
cosine_history = []
patch_history = []
consecutive_detection_count = 0
alert_threshold = 1

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

                cosine_history.append((cosine_sim, ts))
                if len(cosine_history) > 30:
                    cosine_history = cosine_history[-30:]
                print(f"[{format_full_time(ts)}] Cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {knife_detected}")

                if knife_detected:
                    consecutive_detection_count += 1
                    print("Consecutive detections:", consecutive_detection_count)
                    if consecutive_detection_count >= alert_threshold:
                        detection_id = str(uuid.uuid4())
                        image_key = upload_frame_to_s3(
                            processed_frame,
                            ts,
                            cosine_sim,
                            best_coords,
                            detection_id
                        )
                        storeRegister({
                            "cammera": "entrance",
                            "clientId": 1,
                            "event_type": "manual_test",
                            "detection_id": detection_id,
                            "cosine_sim": cosine_sim,
                            "image_key": image_key,
                            "owner_uid": owner_uid,
                        })
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