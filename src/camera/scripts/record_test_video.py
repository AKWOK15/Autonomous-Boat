import cv2
import datetime 
# Open camera using the same device path
# cap = cv2.VideoCapture('/dev/video0')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
date = datetime.datetime.now()
output_dir = f'/home/aidankwok/Autonomous-Boat/data/test_video_{date}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_dir, fourcc, 20.0, (320, 240))

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
if not out.isOpened():
    print(f'Failed to open video writer at {output_dir}!')
else:
    print(f'Recording to {output_dir}')

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        resize = cv2.resize(frame, (320, 240))
        out.write(resize)
        
finally:
# Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped")