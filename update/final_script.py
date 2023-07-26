import time
import cv2
import numpy as np
import os
import serial  
from pycoral.utils.dataset import read_label_file
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

MODEL_PATH = './model.tflite'
LABELS_PATH = './labels.txt'
COUNT = 5
TARGET_SIZE = (300, 350)
TETRAPACK_CLASS = 1
PLASTIC_CLASS = 0

def load_model():
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def load_labels():
    return read_label_file(LABELS_PATH) if LABELS_PATH else {}


def take_photo(timeout=10):
    # Take photo using OpenCV with timeout
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open video capture device")
        return None

    # Set a timeout to capture the image
    start_time = time.time()
    while time.time() - start_time < timeout:
        ret, frame = cap.read()

        # Check the value of ret
        if not ret:
            print("Failed to read frame from the video capture device")
            return None

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return image

    # If the timeout is reached
    print("Camera capture timeout")
    cap.release()
    return None


def trim_image(image):
    # Calculate the trim amounts
    height, width, _ = image.shape
    left_trim = int(width * 0.25)
    right_trim = int(width * 0.25)
    bottom_trim_pixels = int(height * (1 - 0.148))  # Calculate pixels to be trimmed from the bottom

    # Perform the trimming
    trimmed_image = image[:bottom_trim_pixels, left_trim:width - right_trim]

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(trimmed_image, cv2.COLOR_BGR2RGB)

    return rgb_image

def load_and_preprocess_image(image, target_size):
    #to prevent any future errors
    image = np.array(image)

    trimmed_image = trim_image(image)

    # Resize the image
    resized_image = cv2.resize(trimmed_image, target_size)

    # Add batch dimension to the image
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)

    return input_data

def classify_image(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = take_photo(10) # 10 seconds timeout
    if image is None:
        return

    processed_image = load_and_preprocess_image(image, TARGET_SIZE)
    
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    classes = []

    # Perform multiple inferences
    for _ in range(COUNT):
        interpreter.invoke()
        # Get the output tensor and post-process the predictions
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = 1 if output_data[0][0] >= 0.5 else 0
        classes.append(predicted_class)


    # Count occurrences of each class
    tetrapack_count = classes.count(TETRAPACK_CLASS)
    plastic_count = classes.count(PLASTIC_CLASS)

    # Print the final result
    print('-------RESULTS--------')
    print('Tetrapack Count:', tetrapack_count)
    print('Plastic Count:', plastic_count)

    # Classify the image based on the majority of inferences
    if tetrapack_count > plastic_count:
        return TETRAPACK_CLASS
    elif plastic_count > tetrapack_count:
        return PLASTIC_CLASS
    

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        exit()
    interpreter = load_model()
    # connect to raspberry
    ser = serial.Serial("/dev/ttymxc2", 9600)
    time.sleep(10) # 10 seconds to load model (first time only)

    try:
        while True:
            activity = ser.readline().decode()
            print(activity)
            # Receives command to classify object
            if activity == "d\n":
                print("Received command to classify object.")
                prediction = classify_image(interpreter)
                if prediction == PLASTIC_CLASS:
                    print("Classified as Plastic")
                    ser.write(("plastic" + '\n').encode())  # plastic or can
                elif prediction == TETRAPACK_CLASS:
                    print("Classified as Tetrapack")
                    ser.write(("tetrapack" + '\n').encode())
                else:
                    print("No object classified.")
                    ser.write("0\n".encode())

    except KeyboardInterrupt:
        # Handle Ctrl+C 
        print("KeyboardInterrupt: Stopping the script...")
        ser.close()
        exit()

if __name__ == "__main__":
    main()

