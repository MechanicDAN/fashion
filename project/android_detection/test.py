#@title Run object detection and show the detection results
import numpy as np
from IPython.core.display_functions import display
from PIL import Image

from project.android_detection.visualization_helper import visualize, ObjectDetectorOptions, ObjectDetector

DETECTION_THRESHOLD = 0.5
TFLITE_MODEL_PATH = "android.tflite"

image = Image.open(r"test_image.jpg") .convert('RGB')
image.thumbnail((512, 512), Image.ANTIALIAS)
image_np = np.asarray(image)

# Load the TFLite model
options = ObjectDetectorOptions(
      num_threads=4,
      score_threshold=DETECTION_THRESHOLD,
)
detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

# Run object detection estimation using the model.
detections = detector.detect(image_np)

# Draw keypoints and edges on input image
image_np = visualize(image_np, detections)

# Show the detection result
image = Image.fromarray(image_np)

image.save("result.jpg")