import cv2
import numpy as np
from keras.models import load_model
from ximea import xiapi

# sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0
# Load the pre-trained model
model = load_model('seven_expresions.h5')

# Define class labels for the 7 classes
class_labels = ['angry', 'happy', 'surprise', 'disgust', 'neutral', 'fear', 'sad']
# class_labels = ['surprise', 'scared', 'disgust', 'happy', 'sad', 'angry', 'neutral'] #for resEmoteNet

# Initialize Haar Cascade for face detection (you can use MTCNN if preferred)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cam = xiapi.Camera()
print('Opening first camera...')
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())
img = xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    frame = cv2.resize(image,(480,480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = gray[y:y+h, x:x+w]

        # Resize to 48x48
        face_resized = cv2.resize(face, (48, 48))
        cv2.imshow('Input Image to NN', face_resized)

        # Normalize pixel values
        face_normalized = face_resized / 255.0

        # Reshape to match the input shape of the model
        face_input = np.expand_dims(face_resized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)  # Add channel dimension (48x48x1)

        # Predict the probabilities for each class
        predictions = model.predict(face_input)[0]
        percentages = (predictions * 100).round(2)

        # Get the class with the highest probability
        max_index = np.argmax(predictions)
        predicted_class = class_labels[max_index]

        # Display the results on the frame
        text = f'{predicted_class}: {percentages[max_index]}%'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

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