import cv2
import mediapipe as mp
import datetime
import os
from rclpy.node import Node
from sensor_msgs.msg import Node
from cv_bridge import CvBridge
import time
from geometry_msgs import Twist

#1. need node class
#2. get raw image message
#3. turn it into opencv and send it into algo
#4. find bounding box of face and turn that into angle measurment 
#5 send angle measurment to arudino 

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            Twist,
            '/camera/cmd_vel',
            10
        )
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.width = 320
        self.height = 240
        self.center_x = self.width/2
        self.turn_threshold = self.width * 0.05
        self.servo_angle = 55

        #Record frames with bounding boxes
        self.output_dir = '/home/aidankwok/Autonomous-Boat/data/model'
        self.date = datetime.datetime.now()
        self.video_path = os.path.join(output_dir, f'face_detection_node_{date}.mp4')
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(video_path, fself.ourcc, 20, (width, height))
        self.frame_count = 0


    def image_callback(self, msg):
        msg = Twist()
        frame= self.bridge.imgsmg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * self.width)
                y = int(bbox.ymin * self.height)
                w = int(bbox.width * self.width)
                h = int(bbox.height * self.height)
                movement = x - center_x
                if abs(movement > turn_threshold):
                    self.servo_angle = x / 2.91

                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        msg.angular.z = self.servo_angle
        self.publisher.publish(msg)
        self.out.write(frame)
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            print(f'Frames processed: {self.frame_count}')
    
    def shutdown():
        self.face_detection.close()
        self.out.release()
        print('Properly shutdown ')


def main():
    rclpy.init(args=args)
    face_detection_node = FaceDetectionNode()
    try:
        rclpy.spin(face_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        face_detection_node.shutdown()
        face_detection_node.destroy_node()
        rclpy,shutdown()

if __name__ == '__main__':
    main()
print(f'Done, wrote video to {video_path}!')