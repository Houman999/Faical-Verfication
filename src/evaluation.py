# Library
import os
import numpy as np
import tensorflow as tf
import cv2
from model_train import create_siamese_model

# Load the Siamese model and its weights
input_shape = (100, 100, 3)
siamese_model = create_siamese_model(input_shape)
siamese_model.load_weights('siamese_model.h5')

# Define preprocessing function
def preprocess(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (100, 100)) / 255.0
    return img

# Define verification function
def verify(model, detection_threshold=0.8, verification_threshold=0.8):
    verification_dir = 'application_data/verification_images'
    input_image_path = 'application_data/input_image/input_image.jpg'

    input_img = preprocess(input_image_path)
    results = []

    for img_name in os.listdir(verification_dir):
        verification_img = preprocess(os.path.join(verification_dir, img_name))
        prediction = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(verification_img, axis=0)])
        results.append(prediction[0][0])

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    
    return results, verification > verification_threshold

# Main loop for webcam verification
def run_verification():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[120:120+250, 200:200+250]
        cv2.imshow('Verification', cropped_frame)

        # Press 'v' to verify
        if cv2.waitKey(10) & 0xFF == ord('v'):
            cv2.imwrite('application_data/input_image/input_image.jpg', cropped_frame)
            results, verified = verify(siamese_model)
            print("Verification Results:", results)
            print("Verification Status:", "Verified" if verified else "Not Verified")

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


run_verification()