import os
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "keras_model.h5")

# Load the model
model = load_model(model_path, compile=False)

# Load the labels
labels_path = os.path.join(script_dir, "labels.txt")
with open(labels_path, "r") as f:
    class_names = f.readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera 0, trying camera 1...")
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try camera index 1

if not camera.isOpened():
    print("Error: Could not open any camera")
    exit()

print("Camera opened successfully")

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    
    if not ret or image is None:
        print("Failed to grab frame from camera")
        break

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image_np = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_np = (image_np / 127.5) - 1

    # Predicts the model
    try:
        prediction = model.predict(image_np)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])
    except Exception as e:
        print("Prediction error:", e)
        continue

    # Print prediction and confidence score
    print(f"Class: {class_name} | Confidence Score: {confidence_score * 100:.2f}%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()