# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import cv2
from PIL import Image
# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image_path = "/home/aidankwok/Autonomous-Boat/model/chest.png"
frame = cv2.imread(image_path)
output = model(frame)
no_ultralytics = output[0]
print(f'no ultralytics: {no_ultralytics}')
results = Detections.from_ultralytics(output[0])
print(f'results: {results}')
for result in results:
    print(f'result [0]: {result[0]}')
    x1, y1, x2, y2 = result[0]
    print(f'x1: {x1}')
    print(f'y1: {y1}')
    print(f'x2: {x2}')
    print(f'y2: {y2}')
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), 2)
cv2.imshow("Frame", frame)
cv2.waitKey(0)

cv2.destroyAllWindows()