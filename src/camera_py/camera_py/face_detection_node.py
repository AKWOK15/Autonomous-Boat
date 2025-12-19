import cv2
import mediapipe as mp
import datetime
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # Fixed import
from geometry_msgs.msg import Twist  # Fixed import
from cv_bridge import CvBridge

class FaceDetectionNode(Node):
    
    def __init__(self):
        #Creates logger, create subscription, create publisher
        super().__init__('face_detection_node')
        self.get_logger().info('Face detection node started!')
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
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )
        
        self.width = 320
        self.height = 240
        self.center_x = self.width / 2
        self.turn_threshold = self.width * 0.05
        self.servo_angle = 55.0
        
        # Record frames with bounding boxes
        self.output_dir = '/home/aidankwok/Autonomous-Boat/data/model'
        os.makedirs(self.output_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        self.date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f'face_detection_node_{self.date}.mp4')
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 20, (self.width, self.height))
        self.frame_count = 0
    
    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')  # Fixed typo
        frame = cv2.resize(frame, (self.width, self.height))
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)  # Fixed: added self.
        
        twist_msg = Twist()  # Create message here
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * self.width)
                y = int(bbox.ymin * self.height)
                w = int(bbox.width * self.width)
                h = int(bbox.height * self.height)
                
                face_center_x = x + w / 2
                movement = face_center_x - self.center_x
                
                if abs(movement) > self.turn_threshold:  # Fixed: added parentheses
                    self.servo_angle = face_center_x / 2.91
                    self.get_logger().info(f'servo_angle: {self.servo_angle}')
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        twist_msg.angular.z = self.servo_angle
        self.publisher.publish(twist_msg)
        
        self.out.write(frame)
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            print(f'Frames processed: {self.frame_count}')
    
    def shutdown(self):
        self.face_detection.close()
        self.out.release()
        print(f'Properly shutdown. Video saved to: {self.video_path}')

def main(args=None):
    print('Running')
    rclpy.init(args=args)
    face_detection_node = FaceDetectionNode()
    
    try:
        rclpy.spin(face_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        face_detection_node.shutdown()
        face_detection_node.destroy_node()
        rclpy.shutdown()  # Fixed typo

if __name__ == '__main__':
    main()