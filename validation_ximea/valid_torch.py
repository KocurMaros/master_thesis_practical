import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ximea import xiapi

# Load the pre-trained PyTorch model
class CustomEmotionModel(torch.nn.Module):
    def __init__(self):
        super(CustomEmotionModel, self).__init__()
        # Define your model architecture here if not already done
        pass

    def forward(self, x):
        # Define the forward pass here
        pass

# Load the model architecture and weights
model = CustomEmotionModel()
model.load_state_dict(torch.load('best_mode_RAF-DB.pth', map_location=torch.device('cpu')))
model.eval()

# Define class labels for the 7 classes
class_labels = ['surprise', 'scared', 'disgust', 'happy', 'sad', 'angry', 'neutral']  # for resEmoteNet

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    frame = cv2.resize(image, (480, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = gray[y:y + h, x:x + w]

        # Preprocess the face
        face_resized = cv2.resize(face, (64, 64))
        cv2.imshow('Input Image to NN', face_resized)
        face_tensor = transform(face_resized).unsqueeze(0)  # Add batch dimension

        # Predict the probabilities for each class
        with torch.no_grad():
            predictions = F.softmax(model(face_tensor), dim=1).numpy()[0]

        percentages = (predictions * 100).round(2)

        # Get the class with the highest probability
        max_index = np.argmax(predictions)
        predicted_class = class_labels[max_index]

        # Display the results on the frame
        text = f'{predicted_class}: {percentages[max_index]}%'
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Print percentages for all classes
        for i, (label, percentage) in enumerate(zip(class_labels, percentages)):
            class_text = f'{label}: {percentage}%'
            cv2.putText(frame, class_text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Display the frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
