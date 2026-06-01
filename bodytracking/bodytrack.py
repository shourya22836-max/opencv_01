import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('bodytracking/IMG_2385 (1).mov')
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()  # Fix 1: Pose() not pose() — it's a class, not a function

ptime = 0

while True:
    success, img = cap.read()
    if not success:  # Fix 2: handle end of video / read failure
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Fix 3: BGR2RGB not BAYER_BG2RGB
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS  # Fix 4: add connections so skeleton lines draw
        )
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)