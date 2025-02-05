import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import tensorflow as tf

class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction_node')
        self.publisher_ = self.create_publisher(String, 'predictions', 10)
        self.bridge = CvBridge()

        # Load your .h5 model
        self.model = tf.keras.models.load_model('model.h5')

        # Start the camera loop
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        # Capture frame from camera
        cap = cv2.VideoCapture(0)  # Adjust camera index if needed
        ret, frame = cap.read()
        if not ret:
            self.get_logger().warning("Camera frame not captured")
            return
        
        # Preprocess the frame for the model
        input_image = cv2.resize(frame, (224, 224))  # Adjust to model input size
        input_image = input_image / 255.0  # Normalize if needed
        input_image = input_image.reshape((1, 224, 224, 3))  # Add batch dimension

        # Get the prediction
        prediction = self.model.predict(input_image)
        predicted_label = f'Label: {prediction.argmax()}'

        # Publish the prediction
        self.publisher_.publish(String(data=predicted_label))

        # Release the camera (if needed for single frames; keep it open for continuous capture)
        cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
