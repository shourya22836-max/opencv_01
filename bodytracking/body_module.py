

# class bodytrack():

#     def __init__(self, mode=False, upBody=False, smooth=True,detectioncon=0.5, trackcon=0.5):

#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackConr= trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose()  # Fix 1: Pose() not pose() — it's a class, not a function

#     if not success:  # Fix 2: handle end of video / read failure
#         break
# d
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Fix 3: BGR2RGB not BAYER_BG2RGB
#     results = pose.process(imgRGB)

#     if results.pose_landmarks:
#         mpDraw.draw_landmarks(
#             img,
#             results.pose_landmarks,
#             mpPose.POSE_CONNECTIONS  # Fix 4: add connections so skeleton lines draw
#         )
#         for id, lm in enumerate(results.pose_landmarks.landmark):
#             h, w, c = img.shape
#             print(id, lm)
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

#     ctime = time.time()
#     fps = 1 / (ctime - ptime)
#     ptime = ctime

#     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
#     cv2.imshow("image", img)
#     cv2.waitKey(1)

# def main():
#     cap = cv2.VideoCapture('bodytracking/IMG_2385 (1).mov')
#     ptime = 0

#     while True:
#         success, img = cap.read()


#     ctime = time.time()
#     fps = 1 / (ctime - ptime)
#     ptime = ctime

#     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
#     cv2.imshow("image", img)
#     cv2.waitKey(1)


    



# if __name__ == "__main__":
#     main()
"""
bodytrackmodule.py
------------------
Reusable MediaPipe pose-detection module.
 
Usage (as a module):
    from bodytrackmodule import BodyTracker
    tracker = BodyTracker()
    landmarks = tracker.find_pose(img)
    lm_list    = tracker.get_position(img)
"""
 
import cv2
import mediapipe as mp
import time
 
 
class BodyTracker:
    """Wraps MediaPipe Pose for easy reuse across projects."""
 
    def __init__(
        self,
        static_image_mode: bool = False,
        upper_body_only: bool = False,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
 
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.results = None
 
    # ------------------------------------------------------------------
    def find_pose(self, img, draw: bool = True):
        """
        Detect pose landmarks in *img* (BGR numpy array).
 
        Parameters
        ----------
        img  : BGR image (numpy array)
        draw : if True, draws skeleton + landmark circles onto *img*
 
        Returns
        -------
        img  : annotated image (same array, modified in-place)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
 
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return img
 
    # ------------------------------------------------------------------
    def get_position(self, img, draw: bool = True):
        """
        Return a list of [id, cx, cy] pixel positions for every landmark.
 
        Parameters
        ----------
        img  : BGR image (used only for dimensions + optional circles)
        draw : if True, draws a filled circle at each landmark
 
        Returns
        -------
        lm_list : list of [landmark_id, cx_pixels, cy_pixels]
        """
        lm_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lm_list.append([lm_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
        return lm_list
 