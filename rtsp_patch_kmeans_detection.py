import cv2
import time
from PIL import Image
import torch
import clip
import numpy as np
from twilio.rest import Client

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
rtsp_url = "rtsp://admin551:123456789@192.168.20.102:554/stream1"  # Update with your camera's credentials

# Set device and load the CLIP model with its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute the text embedding for the query "person_holding_knife"
text_prompt = "knife"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

def kmeans_detection(image, k=3, threshold=0.290):
    """
    Uses k-means clustering to segment the image into k clusters, reduces colors, 
    and extracts patches based on each cluster's bounding box. Each patch is resized 
    and processed through CLIP to compute the cosine similarity with the text embedding.
    
    Returns:
      - max_cosine: the highest cosine similarity across patches.
      - detection_found: True if any patch exceeds the threshold.
      - segmented_image: the color-reduced (segmented) image.
    """
    # Convert PIL image to NumPy array (RGB)
    np_image = np.array(image)
    h, w, c = np_image.shape
    # Reshape for k-means (each row is a pixel in RGB space)
    pixels = np_image.reshape((-1, 3)).astype(np.float32)
    
    # Run k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape((h, w, 3))
    
    max_cosine = 0.0
    detection_found = False
    
    # Process each cluster to form a patch
    for cluster_idx in range(k):
        mask = (labels.flatten() == cluster_idx).reshape((h, w))
        # Find coordinates where the cluster is present
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            continue
        # Compute bounding box for the cluster
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Crop the patch from the original image
        patch = image.crop((x_min, y_min, x_max, y_max))
        # Resize the patch to 224x224 (CLIP input size)
        patch = patch.resize((224, 224))
        patch_tensor = preprocess(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            patch_embedding = model.encode_image(patch_tensor)
            patch_embedding /= patch_embedding.norm(dim=-1, keepdim=True)
        cosine_sim = (patch_embedding @ text_embedding.T).squeeze().item()
        max_cosine = max(max_cosine, cosine_sim)
        if cosine_sim > threshold:
            detection_found = True
    return max_cosine, detection_found, segmented_image

# Open the RTSP stream using FFMPEG backend
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

consecutive_detection_count = 0  # Counter for consecutive detections
alert_threshold = 1              # Number of consecutive detections required to send an alert

try:
    while True:
        # Flush the stream for a short time to get the latest frame
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
        lab = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(1, 1))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        # -------------------------------------------------

        # Convert the enhanced frame (BGR) to a PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        
        # Run k-means detection on the image
        cosine_sim, detection_found, segmented_image = kmeans_detection(pil_image, k=30, threshold=0.260)
        print(f"Max cosine similarity: {cosine_sim:.3f} | {text_prompt} detected: {detection_found}")

        # Count consecutive detections
        if detection_found:
            consecutive_detection_count += 1
            print("Consecutive detections:", consecutive_detection_count)
        else:
            consecutive_detection_count = 0
        
        # Send SMS alert when consecutive detections hit the threshold
        if consecutive_detection_count >= alert_threshold:
            send_sms_alert(f"Alert: '{text_prompt}' detected consecutively {alert_threshold} times!")
            consecutive_detection_count = 0
        
        # Display the enhanced frame and the segmented (color reduced) image
        cv2.imshow('RTSP Stream', enhanced_frame)
        # Convert segmented image (RGB) to BGR for display with OpenCV
        segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Segmented (Color Reduced)', segmented_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)

finally:
    cap.release()
    cv2.destroyAllWindows()
