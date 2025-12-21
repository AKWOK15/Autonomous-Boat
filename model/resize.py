import cv2
import datetime 
import time
import video_objects

width = 640
height = 480
#Needed to tell camera what format to send otuput in 
#Default was YUYV which is raw uncompressed video format
date = datetime.datetime.now()
output_dir = f'/home/aidankwok/Autonomous-Boat/data/{width}x{height}'
cap, out = video_objects.create_objects('/home/aidankwok/Autonomous-Boat/data/1280x960/2025-12-20 20:36:39.602939.mp4', output_dir, height, width, 30.0)

frame_count = 0
# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
if not out.isOpened():
    print(f'Failed to open video writer at {output_dir}!')
else:
    print(f'Recording to {output_dir}')

try:
    print("made it to try block")
    while True:
        ret, frame = cap.read()
        frame_count+=1
        print(frame_count)
        resize = cv2.resize(frame, (width, height))
        out.write(resize)

            
        
finally:
# Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped")
