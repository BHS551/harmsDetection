import cv2
import time
from PIL import Image
import torch
import clip
from twilio.rest import Client

# === Twilio configuration ===
account_sid = "AC56e980334782eb62fabb2dced530ad84"  # Replace with your Account SID
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
rtsp_url = "rtsp://admin551:123456789@192.168.20.102:554/stream1"  # Update with your camera's credentials

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute the text embedding for the query "knife"
text_prompt = "person_holding_knife"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def sliding_window_detection(image, patch_size=224, stride=112, threshold=0.260):
    """
    Slides a window across the image, computes the CLIP embeddings for each patch in a batch,
    and returns the maximum cosine similarity and a flag indicating whether any patch exceeds the threshold.
    """
    width, height = image.size  # PIL image dimensions (width, height)
    patches = []
    
    # Collect patches using the defined stride
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            patches.append(preprocess(patch))
    
    if not patches:
        return 0.0, False

    # Stack patches into a single tensor and move to device
    batch_tensor = torch.stack(patches).to(device)
    
    with torch.no_grad():
        # Compute embeddings for all patches in one forward pass
        patch_embeddings = model.encode_image(batch_tensor)
        patch_embeddings /= patch_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarities between each patch and the text embedding
    cosine_sim = (patch_embeddings @ text_embedding.T).squeeze()
    
    # Get maximum similarity and determine if any patch exceeds the threshold
    max_cosine = cosine_sim.max().item() if cosine_sim.numel() > 0 else 0.0
    detection_found = (cosine_sim > threshold).any().item()
    
    return max_cosine, detection_found

# Open the RTSP stream using FFMPEG backend
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

consecutive_detection_count = 0  # Counter for consecutive detections
alert_threshold = 1              # Number of consecutive detections required to send an alert

try:
    while True:
        # Flush the stream for 1 second to get the latest frame
        flush_start = time.time()
        latest_frame = None
        while time.time() - flush_start < 0.1:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
        
        if latest_frame is None:
            print("Failed to capture a fresh frame")
            continue

        # --- Apply contrast enhancement using CLAHE ---
        # Convert the frame to LAB color space
        lab = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Create CLAHE object (you can adjust clipLimit and tileGridSize as needed)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        # Merge channels and convert back to BGR
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        # -------------------------------------------------

        # Convert the enhanced frame (BGR) to a PIL Image (RGB) for detection
        image = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        
        # Run sliding window detection on the enhanced image
        cosine_sim, knife_detected = sliding_window_detection(image, patch_size=224, stride=112, threshold=0.290)
        print(f"Max cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {knife_detected}")

        # Count consecutive detections
        if knife_detected:
            consecutive_detection_count += 1
            print("Consecutive detections:", consecutive_detection_count)
        else:
            consecutive_detection_count = 0
        
        # Send SMS alert when consecutive detections hit the threshold
        if consecutive_detection_count >= alert_threshold:
            send_sms_alert(f"Alert: '{text_prompt}' detected consecutively {alert_threshold} times!")
            consecutive_detection_count = 0  # Reset after alert
        
        # Display the enhanced frame (with the applied contrast mask)
        cv2.imshow('RTSP Stream', enhanced_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Wait a short time before processing the next frame
        time.sleep(0.1)

finally:
    cap.release()
    cv2.destroyAllWindows()
