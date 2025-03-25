import cv2
import cv2.data
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from approach.ResEmoteNet import ResEmoteNet
from ximea import xiapi
# Add ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from flask import Flask, Response
import threading
import signal
import sys
class EmotionRecognitionNode(Node):
    def __init__(self):
        super().__init__('emotion_recohgnition_node')
        self.publisher_ = self.create_publisher(String, 'emotion_prediction', 10)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Emotions labels
        self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        
        self.model = ResEmoteNet().to(self.device)
        checkpoint = torch.load('scripts/rafdb_model.pth', weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Load the DNN face detector (SSD)
        self.face_net = cv2.dnn.readNetFromCaffe(
            'scripts/deploy.prototxt',
            'scripts/res10_300x300_ssd_iter_140000.caffemodel'
        )
        # Access the webcam
        self.cap = xiapi.Camera()
        self.cap.open_device()
        self.cap.set_exposure(50000)
        self.cap.set_param("imgdataformat", "XI_RGB24")
        self.cap.set_param("auto_wb", 1)
        self.img = xiapi.Image()
        print('Starting data acquisition...')
        self.cap.start_acquisition()
        
        # Settings for text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.2
        self.font_color = (0, 255, 0)  # This is BGR color
        self.thickness = 3
        self.line_type = cv2.LINE_AA
        
        self.max_emotion = ''
        self.counter = 0
        self.evaluation_frequency = 5
        
        # Create timer for processing frames
        self.timer = self.create_timer(0.1, self.process_frame)

        #create a thread to run the Flask app
        self.thread = threading.Thread(target=self.run_flask)
        self.thread.start()
    def process_frame_for_web(self):
        self.cap.get_image(self.img)
        image = self.img.get_image_data_numpy()
        # image = image[:, :, [2, 1, 0]]  # RGB to BGR
        frame = cv2.resize(image, (800, 800))
        
        # Convert to JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def gen_frames(self):
        while True:
            frame_bytes = self.process_frame_for_web()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    def gen(self):
        """Generate Server-Sent Events (SSE) for emotion updates"""
        import time
        while True:
            try:
                # Format as Server-Sent Event
                if self.max_emotion:
                    yield f"data: {self.max_emotion}\n\n"
                else:
                    yield f"data: waiting...\n\n"
                # Sleep to avoid flooding the client
                time.sleep(0.5)
            except Exception as e:
                self.get_logger().error(f"Error in gen(): {e}")
                yield f"data: error\n\n"
                time.sleep(1)
    def run_flask(self):
        app = Flask(__name__)
        
        @app.route('/video_feed')
        def video_feed():
            return Response(self.gen_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/emotion')
        def emotion():
            return Response(self.gen(), mimetype='text/event-stream')

        @app.route('/')
        def index():
            return '''
            <html>
            <head>
                <title>Emotion Recognition Stream</title>
                <style>
                #emotion { font-size: 24px; margin-top: 20px; font-weight: bold; }
                body { font-family: Arial, sans-serif; }
                </style>
            </head>
            <body>
                <h1>Live Emotion Detection</h1>
                <img src="/video_feed" width="480" height="480" />
                <div id="emotion">Detected emotion: waiting...</div>
                <div id="status">Connection status: connecting...</div>
                <script>
                const eventSource = new EventSource('/emotion');
                eventSource.onmessage = function(e) {
                    console.log('Received emotion:', e.data);
                    document.getElementById('emotion').innerHTML = 'Detected emotion: ' + e.data;
                };
                eventSource.onopen = function() {
                    document.getElementById('status').innerHTML = 'Connection status: connected';
                };
                eventSource.onerror = function(e) {
                    document.getElementById('status').innerHTML = 'Connection status: error/reconnecting';
                    console.error('EventSource error:', e);
                };
                </script>
            </body>
            </html>
            '''
        
        app.run(host='0.0.0.0', port=5000, threaded=True)

        
    
    def detect_emotion(self, video_frame):
        vid_fr_tensor = self.transform(video_frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(vid_fr_tensor)
            probabilities = F.softmax(outputs, dim=1)
        scores = probabilities.cpu().numpy().flatten()
        rounded_scores = [round(score, 2) for score in scores]
        return rounded_scores
    
    def get_max_emotion(self, x, y, w, h, video_frame):
        crop_img = video_frame[y : y + h, x : x + w]
        pil_crop_img = Image.fromarray(crop_img)
        rounded_scores = self.detect_emotion(pil_crop_img)    
        max_index = np.argmax(rounded_scores)
        max_emotion = self.emotions[max_index]
        return max_emotion, rounded_scores[max_index]
    
    def print_max_emotion(self, x, y, video_frame, max_emotion):
        org = (x, y - 15)
        cv2.putText(video_frame, max_emotion, org, self.font, self.font_scale, 
                   self.font_color, self.thickness, self.line_type)
        
    def print_all_emotion(self, x, y, w, h, video_frame):
        crop_img = video_frame[y : y + h, x : x + w]
        pil_crop_img = Image.fromarray(crop_img)
        rounded_scores = self.detect_emotion(pil_crop_img)
        org = (x + w + 10, y - 20)
        for index, value in enumerate(self.emotions):
            emotion_str = (f'{value}: {rounded_scores[index]:.2f}')
            y = org[1] + 40
            org = (org[0], y)
            cv2.putText(video_frame, emotion_str, org, self.font, self.font_scale, 
                       self.font_color, self.thickness, self.line_type)
        
    def detect_bounding_box(self, video_frame):
        gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            # Draw bounding box on face

            # cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop bounding box
            if self.counter == 0:
                self.max_emotion, max_probability = self.get_max_emotion(x, y, w, h, video_frame)
                
                max_probability *= 100

                # Publish emotion to ROS2 topic
                msg = String()
                # Include probability in the message
                msg.data = f'{self.max_emotion} ({max_probability:.2f}%)'
                self.publisher_.publish(msg)
                # self.get_logger().info(f'Publishing: {msg.data}')
            
            # self.print_max_emotion(x, y, video_frame, self.max_emotion) 
            # self.print_all_emotion(x, y, w, h, video_frame)
    
        return faces
    def detect_bounding_box_sgg(self, video_frame):
        h, w = video_frame.shape[:2]
        # Create a 300x300 blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(video_frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network and get detections
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Process detections
        faces_detected = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.5:
                faces_detected += 1
                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within the frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Calculate width and height
                face_w, face_h = x2 - x1, y2 - y1
                
                # Skip if dimensions are too small
                if face_w < 20 or face_h < 20:
                    continue
                    
                # Draw bounding box
                # cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Process for emotion detection if needed
                if self.counter == 0:
                    self.max_emotion, max_probability = self.get_max_emotion(
                        x1, y1, face_w, face_h, video_frame)
                    
                    max_probability *= 100
                    
                    # Publish emotion to ROS2 topic
                    msg = String()
                    msg.data = f'{self.max_emotion} ({max_probability:.2f}%)'
                    self.publisher_.publish(msg)
                    # self.get_logger().info(f'Publishing: {msg.data}')
                
                # Optional: Display emotion on the frame
                # self.print_max_emotion(x1, y1, video_frame, self.max_emotion)
        
        return faces_detected

    def process_frame(self):
        self.cap.get_image(self.img)
        image = self.img.get_image_data_numpy()
        image = image[:, :, [2, 1, 0]]  # RGB to BGR
        frame = cv2.resize(image, (480, 480))
        
        # self.detect_bounding_box(frame)
        self.detect_bounding_box_sgg(frame)
        # cv2.imshow("ResEmoteNet", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cap.stop_acquisition()
            self.cap.close_device()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()
            return
        
        self.counter += 1
        if self.counter == self.evaluation_frequency:
            self.counter = 0
            
def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    print('\nReceived termination signal. Shutting down...')
    if 'node' in globals():
        node.cleanup()
    rclpy.shutdown()
    sys.exit(0)
def main(args=None):
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Termination request
    
    rclpy.init(args=args)
    
    global node
    node = EmotionRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Exception in main loop: {e}")
    finally:
        # Ensure cleanup happens
        node.cleanup()
        node.cap.stop_acquisition()
        node.cap.close_device()
        rclpy.shutdown()


if __name__ == '__main__':
    main()