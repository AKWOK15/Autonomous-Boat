# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import cv2
import os
import datetime
# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference


output_dir = f'/home/aidankwok/Autonomous-Boat/data/model'
os.makedirs(output_dir, exist_ok=True)
date = datetime.datetime.now()
video_name = f'face_detection_{date}.mp4'
video_path = os.path.join(output_dir, video_name)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height = 240
width = 320
out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
cap = cv2.VideoCapture('/home/aidankwok/Autonomous-Boat/data/test_video_2025-12-16 18:20:03.160070.mp4')
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
                # print(f'x1: {x1}')
                # print(f'y1: {y1}')
                # print(f'x2: {x2}')
                # print(f'y2: {y2}')
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

main()