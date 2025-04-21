# camera_streamer_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ximea import xiapi

class CameraStreamer(Node):
    def __init__(self):
        super().__init__('camera_streamer')
        self.bridge = CvBridge()
        self.timer_period = 0.1  # 10 FPS

        # Try Ximea first
        try:
            self.get_logger().info('Trying to open Ximea camera...')
            self.ximea = xiapi.Camera()
            self.ximea.open_device()
            self.ximea.set_exposure(50000)
            self.ximea.set_param("imgdataformat", "XI_RGB24")
            self.ximea.set_param("auto_wb", 1)
            self.ximea_img = xiapi.Image()
            self.ximea.start_acquisition()
            self.ximea_pub = self.create_publisher(Image, '/rgb_stream/ximea', 10)
            self.create_timer(self.timer_period, self.publish_ximea)
            self.get_logger().info('Ximea camera started.')
        except Exception as e:
            self.get_logger().warn(f'Ximea unavailable: {e}')
            self.ximea = None

        # Try default camera
        self.default_cam = cv2.VideoCapture(0)
        if self.default_cam.isOpened():
            self.default_pub = self.create_publisher(Image, '/rgb_stream/default', 10)
            self.create_timer(self.timer_period, self.publish_default)
            self.get_logger().info('Default camera started.')
        else:
            self.get_logger().warn('Default camera unavailable.')
            self.default_cam = None

    def publish_ximea(self):
        if self.ximea:
            self.ximea.get_image(self.ximea_img)
            frame = self.ximea_img.get_image_data_numpy()
            frame = frame[:, :, [2, 1, 0]]  # RGB to BGR
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            self.ximea_pub.publish(msg)

    def publish_default(self):
        if self.default_cam:
            ret, frame = self.default_cam.read()
            if ret:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.default_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamer()
    rclpy.spin(node)
    if node.default_cam:
        node.default_cam.release()
    if node.ximea:
        node.ximea.stop_acquisition()
        node.ximea.close_device()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
