import cv2
import time

for cam in [0, 1]:

    print(f"\n=== Camera {cam} ===")

    t0 = time.time()

    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)

    print("open =", round(time.time()-t0, 2), "s")

    t0 = time.time()

    ret, frame = cap.read()

    print("read =", round(time.time()-t0, 2), "s")

    if ret:
        print(frame.shape)

    cap.release()