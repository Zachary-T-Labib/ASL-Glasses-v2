import mediapipe as mp
import numpy as np
import cv2
import image_utilities as imageUtil
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from PIL import Image

# Path to mediapipe hand landmarker model
model_path = './hand_landmarker.task'

# Test image
mp_image = mp.Image.create_from_file('./a16.jpg')

# Set up options/settomgs for hand landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    # # ...
    print("placeholder") 
    hand_landmarker_result = landmarker.detect(mp_image)

# Print raw data from landmarker result
print(hand_landmarker_result)

# Prcoessing the classigication result and displaying the annotated image
image_reformated = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
annotated_image = imageUtil.draw_landmarks_on_image(image_reformated, hand_landmarker_result)

annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
annotated_pil.show()