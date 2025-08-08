import cv2

cap = cv2.VideoCapture("rtsp://admin:1234@192.168.0.167:8554/live")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("iPhone Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
