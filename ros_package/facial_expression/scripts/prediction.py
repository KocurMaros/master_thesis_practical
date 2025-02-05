import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from approach.ResEmoteNet import ResEmoteNet

class EmotionRecognitionNode(Node):
    def __init__(self):
        super().__init__('emotion_recognition_node')
        self.publisher_ = self.create_publisher(String, 'emotion_prediction', 10)
        
        # Initialize the model
        self.model = ResEmoteNet()
        self.model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Define class labels for the 7 classes
        self.class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Load the DNN face detector (SSD)
        self.face_net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel'
        )

        # Open the OpenCV camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Cannot open webcam")
            exit()

        # Transformation for PyTorch model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3 channels
        ])

        # Start the timer to process frames continuously
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Cannot read frame from webcam")
            return

        # Resize for faster processing
        frame = cv2.resize(frame, (480, 480))

        # Prepare the blob for face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        (h, w) = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                # Extract bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Ensure bounding box is within frame bounds
                x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)

                # Crop and preprocess the face
                face = frame[y:y1, x:x1]
                if face.size == 0:
                    continue

                face_tensor = self.transform(face).unsqueeze(0)  # Add batch dimension

                # Make prediction
                with torch.no_grad():
                    predictions = self.model(face_tensor)  # Forward pass
                    probabilities = torch.softmax(predictions, dim=1)[0]  # Apply softmax to get probabilities
                    class_idx = torch.argmax(probabilities).item()  # Get class index
                    class_label = self.class_labels[class_idx]  # Map to class label

                # Publish the detected emotion
                msg = String()
                msg.data = f'{class_label}: {probabilities[class_idx].item() * 100:.2f}%'
                self.publisher_.publish(msg)

                # Display results
                cv2.putText(frame, msg.data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Facial Expression Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = EmotionRecognitionNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
