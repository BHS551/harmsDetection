import cv2

# Replace this with your URL (or use an env var)
RTSP_URL = "rtsp://test123:123456789@4.tcp.ngrok.io:19953/stream1"

print("Opening:", RTSP_URL)
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream")

frame_idx = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Failed to read frame, stopping")
        break

    frame_idx += 1

    # frame_bgr is a NumPy array of pixels: (height, width, 3)
    h, w, c = frame_bgr.shape
    print(f"Frame {frame_idx}: shape={frame_bgr.shape}, dtype={frame_bgr.dtype}")

    # If you prefer RGB instead of BGR:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Example: get the pixel at row 100, col 200 (y=100, x=200)
    if h > 100 and w > 200:
        pixel_bgr = frame_bgr[100, 200]      # [B, G, R]
        pixel_rgb = frame_rgb[100, 200]      # [R, G, B]
        print(f"Pixel BGR at (100,200): {pixel_bgr}, RGB: {pixel_rgb}")

    # TODO: here you can pass frame_rgb to CLIP, etc.

    # Just to avoid infinite spam, break after some frames for testing:
    if frame_idx >= 20:
        break

cap.release()
print("Done")