import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
else:
    ret, frame = cap.read()
    print("Webcam OK, frame shape:", frame.shape)
cap.release()