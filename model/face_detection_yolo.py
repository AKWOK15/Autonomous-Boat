# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from video_objects import create_objects
import cv2
import os
import datetime
# download model
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference


output_dir = f'/home/aidankwok/Autonomous-Boat/data/yolo'
cap, out = create_objects('/home/aidankwok/Autonomous-Boat/data/test_video_2025-12-20 16:22:05.047785.mp4', output_dir, 480, 640, 20.0)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()
if not out.isOpened():
    print(f'Failed to open video writer at {output_dir}!')
else:
    print(f'Recording to {output_dir}') 


frame_count = 0
try:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #verbose gets rid of model print statements
            #yolo internally scales to 640x480, then back down to 320x240 
            output = model(frame)
            no_ultralytics = output[0]
            # print(f'no ultralytics: {no_ultralytics}')
            results = Detections.from_ultralytics(output[0])
            # print(f'results: {results}')
            for result in results:
                # print(f'result [0]: {result[0]}')
                x1, y1, x2, y2 = result[0]
                print(f'width: {int(x2-x1)}')
                print(f'height: {int(y2-y1)}')
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), 2)
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(0)

            out.write(frame)
            frame_count+=1
            if frame_count%10 == 0:
                print(f'number of frames processed: {frame_count}')

        else:
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished applying computer vision to test video")
