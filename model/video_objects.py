import datetime 
import os
import cv2
import utils
import time
def create_objects(input_path, output_dir, height, width, fps):
    os.makedirs(output_dir, exist_ok=True)
    date = datetime.datetime.now()
    video_name = f'{date}.mp4'
    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    cap = cv2.VideoCapture(input_path)
    return cap, out


def process_frames(model, cap, out):
    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if ret == False:
                print("End of the file or error to read the next frame.")
                break

            start_time = time.perf_counter()
            bboxes, scores = model.inference(frame)
            end_time = time.perf_counter()

            n_frames += 1
            fps = 1.0 / (end_time - start_time)
            fps_cum += fps
            fps_avg = fps_cum / n_frames

            frame = utils.draw_boxes_with_scores(frame, bboxes, scores)
            frame = utils.put_text_on_image(frame, text='FPS: {:.2f}'.format( fps_avg ))
            out.write(frame)
            if n_frames % 10:
                print(f'number of frames: {n_frames}')
    finally:
        cap.release()
        out.release()
