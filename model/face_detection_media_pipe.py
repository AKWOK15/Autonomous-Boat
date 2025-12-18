import cv2
import mediapipe as mp
import datetime
import os

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0=short range (faster), 1=full range
    min_detection_confidence=0.5
)

output_dir = '/home/aidankwok/Autonomous-Boat/data/model'
os.makedirs(output_dir, exist_ok=True)
date = datetime.datetime.now()
video_path = os.path.join(output_dir, f'face_mediapipe_{date}.mp4')

cap = cv2.VideoCapture('/home/aidankwok/Autonomous-Boat/data/test_video_2025-12-16 18:20:03.160070.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break
    start = datetime.datetime.now()
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    out.write(frame)
    end = datetime.datetime.now()
    processing_time = end-start
    print(f'processing time: {processing_time.microseconds}')
    print(f'processing time: {processing_time}')
    frame_count += 1
    if frame_count % 10 == 0:
        print(f'Frames processed: {frame_count}')

cap.release()
out.release()
face_detection.close()
print(f'Done, wrote video to {video_path}!')