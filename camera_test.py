import cv2

print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("Could not open camera.")
    input("Press Enter to exit...")
    exit()

print("Camera is working.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()