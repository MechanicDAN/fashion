#@title Run object detection and show the detection results
import numpy as np
from IPython.core.display_functions import display
from PIL import Image, ImageOps
import easyocr

from project.price_datection.visualization_helper import visualize, ObjectDetectorOptions, ObjectDetector

DETECTION_THRESHOLD = 0.1
TFLITE_MODEL_PATH = "price.tflite"

image = Image.open(r"lm_test.jpg").convert('RGB')
image = ImageOps.exif_transpose(image)
image.save("test.jpg")
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
visualize(image_np, detections)

# Show the detection result
#image = Image.fromarray(image_np)

reader = easyocr.Reader(["ru", "en"])
result = reader.readtext("name.jpg", detail=0)
print(result)

result = reader.readtext("price.jpg", detail=0)
print(result)