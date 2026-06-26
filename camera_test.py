import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# спроба вимкнути auto exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)


prev_mean = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    print("Brightness:", mean_brightness)

    # перевірка стабільності
    if prev_mean is not None:
        diff = abs(mean_brightness - prev_mean)
        print("Δ brightness:", diff)

    prev_mean = mean_brightness

    cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()