# FaceVerification

**FaceVerification** is a Python-based face verification sys developed using **Siamese Neural Networks**. The sys allows users to authenticate their identity by comparing their face with a set of pre-registered face images. It integrates real-time webcam feed and uses deep learning techniques for accurate verification.

## Features
- Real-time webcam feed for capturing face images.
- Preprocessing of captured images for compatibility with the model.
- Siamese Neural Network architecture for calculating similarity scores.
- Adjustable detection and verification thresholds for customizable verification criteria.
- Integration with TensorFlow and OpenCV libraries for deep learning and image processing.

## How it Works
1. The sys captures an image of the user's face using the webcam.
2. The captured image is preprocessed to match the input requirements of the Siamese Neural Network model.
3. The model compares the input image with a set of pre-registered face images.
4. Similarity scores are calculated for each comparison.
5. The scores are compared against detection and verification thresholds.
6. The sys displays the verification status as **"Verified"** or **"Un-verified"** based on the outcome.

## Implementation Details
This project implements a **Siamese Neural Network** architecture, commonly used for one-shot image recognition. The model is trained on a dataset containing images of the user's face along with images from the **"Labelled Faces in the Wild"** dataset.

## Dependencies
- **TensorFlow**: Deep learning library for building and training neural networks.
- **OpenCV**: Computer vision library for image and video processing.
- **NumPy**: Library for numerical computing and array operations.


## Conclusion
**FaceVerification** demonstrates the potential of deep learning and computer vision in building practical applications. By leveraging the **Siamese Neural Network architecture**, this project provides a reliable face verification solution. Feel free to explore, contribute, and utilize the project for your own applications.
