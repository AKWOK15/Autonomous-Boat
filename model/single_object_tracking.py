from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import os
import argparse
import imutils
import time
import datetime

def main():
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
    initBB = None
    fps = None
    #Settings for video recording
    output_dir = f'/home/aidankwok/Autonomous-Boat/data/model'
    os.makedirs(output_dir, exist_ok=True)
    date = datetime.datetime.now()
    video_name = f'cv_video_{date}.mp4'
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
                # check to see if we are currently tracking an object
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s") or frame_count == 90:
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                    initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                        showCrosshair=True)
                    print(f'initBB: {initBB}')
                    #selectROI gives corners of bouding box to track 
                    #could use a different model to initialy find the bouding box and thenhave single object detection track it afterwards
                    #base bouding box off of my skin color
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    tracker.init(frame, initBB)
                    fps = FPS().start()
                if initBB is not None:
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
                # cv2.imshow("Frame", frame)
                # if the 's' key is selected, we are going to "select" a bounding
                # box to track
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
        print(f'Finished applying computer vision to test video, new video is { video_name}')

main()