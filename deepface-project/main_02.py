import json
from deepface import DeepFace

result = DeepFace.verify(
    img1_path="/Users/shourya/Desktop/opencv_pr_02/deepface-project/ai-generated-image-1764664835172.png",
    img2_path="/Users/shourya/Desktop/opencv_pr_02/deepface-project/Screenshot 2026-04-01 at 6.55.10 PM.png"
)

print(json.dumps(result, indent=2))