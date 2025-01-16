import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ximea import xiapi
from approach.ResEmoteNet import ResEmoteNet


# Initialize the model
model = ResEmoteNet()
model.load_state_dict(torch.load('best_mode_RAF-DB.pth', map_location=torch.device('cpu')))
model.eval()

# Define class labels for the 7 classes
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # for ResEmoteNet

# Initialize Haar Cascade for face detection
# Load the DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

# Open the webcam
cam = xiapi.Camera()
print('Opening first camera...')
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

print('Exposure was set to %i us' % cam.get_exposure())
img = xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

# Transformation for PyTorch model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3 channels
])

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    # Convert RGBA to BGR
    if image.shape[2] == 4:  # If the image has 4 channels (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    print(image.shape)
    frame = cv2.resize(image, (480, 480))

    # Convert to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Prepare the blob for face detection (keep it in BGR format)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    # Process detections
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
            face_tensor = transform(face).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                predictions = model(face_tensor)  # Forward pass
                # percentages = (predictions * 100).round(2)

                probabilities = torch.softmax(predictions, dim=1)[0]  # Apply softmax to get probabilities
                class_idx = torch.argmax(probabilities).item()  # Get class index
                class_label = class_labels[class_idx]  # Map to class label

            # Display results
            text = f'{class_label}: {probabilities[class_idx].item() * 100:.2f}%'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

            # Print percentages for all classes
            # for j, (label, percentage) in enumerate(zip(class_labels, percentages)):
            #     class_text = f'{label}: {percentage}%'
            #     cv2.putText(frame, class_text, (10, 30 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Display the frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
