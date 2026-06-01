import cv2
import mediapipe as mp
import time
import sqlite3
from datetime import datetime

# --- Database setup ---
conn = sqlite3.connect("hand_landmarks.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS landmarks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        hand_index  INTEGER,
        landmark_id INTEGER,
        cx          INTEGER,
        cy          INTEGER,
        x_norm      REAL,
        y_norm      REAL,
        z_norm      REAL
    )
""")
conn.commit()

def save_landmarks(timestamp, hand_index, landmark_id, cx, cy, lm):
    cursor.execute("""
        INSERT INTO landmarks (timestamp, hand_index, landmark_id, cx, cy, x_norm, y_norm, z_norm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, hand_index, landmark_id, cx, cy, lm.x, lm.y, lm.z))

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 5)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
SAVE_EVERY_N_FRAMES = 5   # write to DB every N frames to avoid hammering the disk
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    frame_count += 1
    should_save = (frame_count % SAVE_EVERY_N_FRAMES == 0)

    if results.multi_hand_landmarks:
        ts = datetime.now().isoformat()
        for hand_index, handLms in enumerate(results.multi_hand_landmarks):
            for landmark_id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if should_save:
                    save_landmarks(ts, hand_index, landmark_id, cx, cy, lm)

                if landmark_id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if should_save:
            conn.commit()   # one commit per batch of frames

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
cap.release()
cv2.destroyAllWindows()