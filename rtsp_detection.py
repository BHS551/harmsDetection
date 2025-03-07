import cv2
import time
from PIL import Image
import torch
import clip
from twilio.rest import Client

# === Twilio configuration ===
account_sid = "AC56e980334782eb62fabb2dced530ad84"         # Replace with your Account SID
auth_token = "c7ea6fdcd8bd75d3b9ab6ec2df33e1f5"           # Replace with your Auth Token
twilio_phone = "+14793178516"             # Your Twilio phone number
recipient_phone = "+573043566310"          # The number to alert

client = Client(account_sid, auth_token)

def send_sms_alert(message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone,
        to=recipient_phone
    )
    print("SMS sent:", message.sid)

# === Camera and Detection Configuration ===
rtsp_url = "rtsp://admin551:123456789@192.168.20.102:554/stream1"  # Update with your camera's credentials

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute the text embedding for the query "person asleep"
text_prompt = "knife"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def detect_person(image, threshold=0.245):
    """
    Given a PIL image, compute its embedding and return the cosine similarity
    with the 'person asleep' text embedding along with a boolean flag.
    """
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    cosine_sim = (image_embedding @ text_embedding.T).item()
    return cosine_sim, cosine_sim > threshold

# Open the RTSP stream using FFMPEG backend
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

consecutive_detection_count = 0  # Counter for consecutive detections
alert_threshold = 5              # Number of consecutive detections required to send an alert

try:
    while True:
        # Flush the stream for 0.1 seconds to discard queued frames
        flush_start = time.time()
        latest_frame = None
        while time.time() - flush_start < 0.2:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
        
        if latest_frame is None:
            print("Failed to capture a fresh frame")
            continue

        # Convert the latest frame (BGR) to a PIL Image (RGB) for processing
        image = Image.fromarray(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))
        
        # Run detection on the image
        cosine_sim, person_detected = detect_person(image, threshold=0.250)
        print(f"Cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {person_detected}")

        # Count consecutive detections
        if person_detected:
            consecutive_detection_count += 1
            print("Consecutive detections:", consecutive_detection_count)
        else:
            consecutive_detection_count = 0
        
        # Send SMS alert when detections hit threshold
        if consecutive_detection_count >= alert_threshold:
            send_sms_alert(f"Alert: '{text_prompt}' detected consecutively {alert_threshold} times!")
            consecutive_detection_count = 0  # Reset after alert
        
        # Display the most recent frame
        cv2.imshow('RTSP Stream', latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Wait for 2 seconds before processing the next frame
        time.sleep(0.2)

finally:
    cap.release()
    cv2.destroyAllWindows()
