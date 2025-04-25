import time
import cv2
from PIL import Image
import torch
import clip
from twilio.rest import Client

# ================================
#          TWILIO CONFIG
# ================================
# Replace the credentials below with your own
account_sid = "AC56e980334782eb62fabb2dced530ad84"
auth_token = "c7ea6fdcd8bd75d3b9ab6ec2df33e1f5"
twilio_phone = "+14793178516"
recipient_phone = "+573043566310"

client = Client(account_sid, auth_token)

def send_sms_alert(message_body: str) -> None:
    """Send an SMS alert using Twilio."""
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

# ================================
#     CAMERA & DETECTION CONFIG
# ================================
# Replace with your RTSP URL (username, password, host, port)
rtsp_url = "rtsp://admin551:123456789@0.tcp.ngrok.io:12644/stream1"

# Choose device for Torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessor
model, preprocess = clip.load("ViT-B/32", device=device)

# Text prompt you want to detect
text_prompt = "knife"

# Precompute the text embedding for the prompt
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def detect_prompt(image: Image.Image, threshold: float = 0.245) -> (float, bool):
    """
    Given a PIL image, compute its CLIP embedding and return:
      - The cosine similarity with the precomputed text embedding
      - A boolean indicating whether detection is above threshold
    """
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    cosine_sim = (image_embedding @ text_embedding.T).item()
    return cosine_sim, cosine_sim > threshold

# Open the RTSP stream using FFMPEG backend (recommended for many IP cameras)
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check your URL or camera settings.")

print("Processing frames from RTSP stream...")

consecutive_detection_count = 0  # Counter for consecutive detections
alert_threshold = 5              # Number of consecutive detections required to send an alert
detection_threshold = 0.25       # Similarity threshold for "knife"

try:
    while True:
        # Flush the camera buffer for ~0.2 seconds to get the latest frame
        flush_start = time.time()
        latest_frame = None

        while time.time() - flush_start < 0.2:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
        
        if latest_frame is None:
            print("Failed to capture a fresh frame.")
            continue

        # Convert BGR (OpenCV) -> RGB (PIL)
        image_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Run CLIP detection
        cosine_sim, is_detected = detect_prompt(pil_image, threshold=detection_threshold)
        print(f"Cosine similarity: {cosine_sim:.3f} | '{text_prompt}' detected: {is_detected}")

        # Track consecutive detections
        if is_detected:
            consecutive_detection_count += 1
            print("Consecutive detections:", consecutive_detection_count)
        else:
            consecutive_detection_count = 0

        # Send SMS alert if threshold is reached
        if consecutive_detection_count >= alert_threshold:
            send_sms_alert(
                f"ALERT: '{text_prompt}' detected {alert_threshold} times in a row!"
            )
            consecutive_detection_count = 0  # Reset after alert

        # Show the frame in a window (optional)
        cv2.imshow('RTSP Stream', latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Small delay before next loop iteration
        time.sleep(0.2)

finally:
    cap.release()
    cv2.destroyAllWindows()
