import cv2
import numpy as np
import os
import datetime 
MOG2 = cv2.createBackgroundSubtractorMOG2()

def model(frame, k_5, k_3):
    MOG2_mask = MOG2.apply(frame)
    #First output is threshold that was used, second is thresholded image
    _, threshold = cv2.threshold(MOG2_mask, 180, 255, cv2.THRESH_BINARY)
    
    dilate = cv2.dilate(threshold, k_5, iterations=1)
    morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, k_3)
    return morph
def find_biggest_contour(mask):
    """Find the biggest contour in the mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        return biggest_contour, contours
    return None, []

def draw_rectangle_on_image(image, contour):
    """Draw rectangles around the biggest contour and all contours"""
    result_image = image.copy()
    
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        area = cv2.contourArea(contour)
        cv2.putText(result_image, f'Area: {int(area)}', (x, y-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_image, f'Size: {w}x{h}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result_image

def main():
    output_dir = f'/home/aidankwok/Autonomous-Boat/data/model'
    os.makedirs(output_dir, exist_ok=True)
    date = datetime.datetime.now()
    video_name = f'cv_video_{date}.mp4'
    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (320, 240))
    cap = cv2.VideoCapture('/home/aidankwok/Autonomous-Boat/data/test_video_2025-12-16 14:22:44.312634.mp4')
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    if not out.isOpened():
        print(f'Failed to open video writer at {output_dir}!')
    else:
        print(f'Recording to {output_dir}')

    k_5 = np.ones((5, 5), np.uint8)
    k_3 = np.ones((3, 3), np.uint8)
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            processed_frame = model(frame, k_5, k_3)
            biggest_contour, contours = find_biggest_contour(processed_frame)
            final_frame = draw_rectangle_on_image(frame, biggest_contour)
            out.write(final_frame)
            frame_count+=1
            if frame_count%10 == 0:
                print(f'number of frames processed: {frame_count}')

        else:
            break


    cap.release()
    print("Finished applying computer vision to test video")

main()