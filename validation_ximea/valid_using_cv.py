import cv2
import numpy as np
from keras.models import load_model
from ximea import xiapi

# Load the pre-trained model
model = load_model('seven_expresions.h5')

# Define class labels for the 7 classes
class_labels = ['angry', 'happy', 'surprise', 'disgust', 'neutral', 'fear', 'sad']

# Load the DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

# Open the Ximea camera
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

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    frame = cv2.resize(image, (480, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Prepare the blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Process detections
    (h, w) = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            # Get the bounding box for the detected face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            
            # Ensure the coordinates are within frame bounds
            x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)

            # Crop the face from the frame
            face = frame[y:y1, x:x1]

            # Convert to grayscale
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Resize to 48x48
            face_resized = cv2.resize(gray_face, (48, 48))

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
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

            # Show the input to the neural network in a separate window
            face_display = cv2.resize(face_normalized * 255, (200, 200))  # Scale back to 0-255 for display
            face_display = face_display.astype(np.uint8)  # Convert to uint8 for OpenCV
            cv2.imshow('Input Image to NN', face_display)

            # Print percentages for all classes
            for j, (label, percentage) in enumerate(zip(class_labels, percentages)):
                class_text = f'{label}: {percentage}%'
                cv2.putText(frame, class_text, (10, 30 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Display the main frame with predictions
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
