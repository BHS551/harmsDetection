import cv2
import time
import torch
from twilio.rest import Client

# === Twilio configuration ===
account_sid = null  # Replace with your Account SID
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

# Load the YOLOv5 model from PyTorch Hub (pretrained on COCO)
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Detection parameters
target_class = "knife"         # We're looking for knives
confidence_threshold = 0.84    # Minimum confidence required for detection

def detect_knife(frame):
    """
    Runs YOLOv5 inference on the frame and returns a tuple:
    (True/False if a knife is detected, DataFrame of detections)
    """
    results = model_yolo(frame)
    df = results.pandas().xyxy[0]  # Get results as a pandas DataFrame
    # Filter for target class "knife" with confidence above threshold
    detections = df[(df['confidence'] >= confidence_threshold)]
    print("SMS sent:", df['confidence'])
    return (len(detections) > 0), detections

# Open the RTSP stream using the FFMPEG backend
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise Exception("Failed to open RTSP stream")

print("Processing frames from RTSP stream...")

consecutive_detection_count = 0  # Counter for consecutive detections
alert_threshold = 5              # Number of consecutive detections required to send an alert

try:
    while True:
        # Flush the stream for 0.2 seconds to discard queued frames
        flush_start = time.time()
        latest_frame = None
        while time.time() - flush_start < 0.2:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
        
        if latest_frame is None:
            print("Failed to capture a fresh frame")
            continue

        # Run detection on the latest frame using YOLOv5
        knife_detected, detections = detect_knife(latest_frame)
        print(f"Knife detected: {knife_detected}")

        # Count consecutive detections
        if knife_detected:
            consecutive_detection_count += 1
            print("Consecutive detections:", consecutive_detection_count)
        else:
            consecutive_detection_count = 0
        
        # Send SMS alert when detections hit the threshold
        if consecutive_detection_count >= alert_threshold:
            send_sms_alert(f"Alert: '{target_class}' detected consecutively {alert_threshold} times!")
            consecutive_detection_count = 0  # Reset after alert
        
        # Optionally, draw bounding boxes on the frame for detected knives
        if knife_detected:
            for idx, row in detections.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = row['confidence']
                cv2.rectangle(latest_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(latest_frame, f"{target_class} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the most recent frame
        cv2.imshow('RTSP Stream', latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Wait for 2 seconds before processing the next frame
        time.sleep(0.2)

finally:
    cap.release()
    cv2.destroyAllWindows()
