# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import cv2
import os
import datetime
import argparse
from imutils.video import FPS

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create,
    # "tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create
}
# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
bounding_box_coords = None
fps = None
# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference


output_dir = f'/home/aidankwok/Autonomous-Boat/data/model'
os.makedirs(output_dir, exist_ok=True)
date = datetime.datetime.now()
video_name = f'face_single_object_{date}.mp4'
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
            key = cv2.waitKey(1) & 0xFF
            if bounding_box_coords is None:
                # select the bounding box of the object we want to track (make
                # sure you press ENTER or SPACE after selecting the ROI)

                #selectROI gives corners of bouding box to track 
                #could use a different model to initialy find the bouding box and thenhave single object detection track it afterwards
                #base bouding box off of my skin color
                # start OpenCV object tracker using the supplied bounding box
                # coordinates, then start the FPS throughput estimator as well
                output = model(frame)
                no_ultralytics = output[0]
                # print(f'no ultralytics: {no_ultralytics}')
                results = Detections.from_ultralytics(output[0])
                if len(results) > 0:  # This checks if any faces were detected
                    # Get the first detection's bounding box
                    x1, y1, x2, y2 = results.xyxy[0]  # Access directly with results.xyxy[0]
                    print(f'results.xyxy[0]: {results.xyxy[0]}')
                    print(f'results[0].xyxy: {results[0].xyxy[0]}')
                    width = int(x2-x1)
                    height = int(y2-y1)
                    print(f'area: {width*height}')
                    if (width*height > 450):
                    # Convert to CSRT format: [x, y, width, height]
                        bounding_box_coords = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        
                        print(f'Initializing tracker with box: {bounding_box_coords}')
                        tracker.init(frame, bounding_box_coords)
                        fps = FPS().start()
            if bounding_box_coords is not None:
                # grab the new bounding box coordinates of the object
                (success, box) = tracker.update(frame)
                # check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)
                # update the FPS counter
                fps.update()
                fps.stop()
                # initialize the set of information we'll be displaying on
                # the frame
                info = [
                    ("Tracker", args["tracker"]),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps())),
                ]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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

