import os
import cv2

# Note the query parameter: ?rtsp_transport=tcp
rtsp_url = os.environ.get("RTSP_URL", "")

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
