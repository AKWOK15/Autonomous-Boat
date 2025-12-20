import cv2
import datetime 
import time
# Open camera using the same device path
# cap = cv2.VideoCapture('/dev/video0')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# Set resolution (optional)
width = 640
height = 480
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#Needed to tell camera what format to send otuput in 
#Default was YUYV which is raw uncompressed video format
date = datetime.datetime.now()
output_dir = f'/home/aidankwok/Autonomous-Boat/data/test_video_{date}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_dir, fourcc, 30.0, (width, height))

# 1920x1080 looks off, so need to correct
# Disable auto white balance and set manual
# cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)  # Neutral daylight

# # Reduce sharpness (if supported)
# cap.set(cv2.CAP_PROP_SHARPNESS, 0)  # Try 0-5 range

# # Reduce saturation to avoid over-saturated colors
# cap.set(cv2.CAP_PROP_SATURATION, 32)  # Default is usually 64, try 32-48

# # Set contrast manually
# cap.set(cv2.CAP_PROP_CONTRAST, 32)  # Default, not boosted

# # Disable auto-exposure for more consistent lighting
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
# cap.set(cv2.CAP_PROP_EXPOSURE, 100)  # Adjust this value (try 50-200)

# # Set gain manually to reduce noise
# cap.set(cv2.CAP_PROP_GAIN, 10)  # Lower = less noise, darker image

# # Give camera time to adjust
# time.sleep(2)

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
        print(cap.read())
        ret, frame = cap.read()
        print(frame)
        if not ret or frame is None:
            print("Failed to grab frame")
            break
        print(f'frame width: {frame.shape[1]}')
        print(f'frame height: {frame.shape[0]}')
        # resize = cv2.resize(frame, (320, 240))
        # cv2.imshow('Frame', resize)
        # cv2.waitKey(1)
        out.write(frame)

            
        
finally:
# Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped")
