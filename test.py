import cv2

# Note the query parameter: ?rtsp_transport=tcp
rtsp_url = "rtsp://admin551:123456789@0.tcp.ngrok.io:12644/stream1"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
