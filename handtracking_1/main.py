import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image

# Path to the downloaded MediaPipe hand landmarker task file (placed next to
# this script). Use an absolute path so the script works regardless of CWD.
MODEL_PATH = str(Path(__file__).resolve().parent.joinpath("hand_landmarker.task"))

# Create the hand landmarker with the task-based API.
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Drawing utils from the tasks API
mp_hands = vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB and wrap in a MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)

        # Run detection (image mode). For better performance in a real app
        # consider using LIVE_STREAM mode with async callbacks.
        detection_result = detector.detect(mp_img)

        annotated_image = rgb.copy()

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Convert back to BGR for OpenCV display
        bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Close the detector to free native resources if available
    try:
        detector.close()
    except Exception:
        pass
    