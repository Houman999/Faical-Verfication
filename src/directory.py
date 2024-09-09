# Library
import cv2
import os
import uuid

# Define paths and directory
POS_PATH = os.path.join('data' , 'positive')
NEG_PATH = os.path.join('data' , 'negative')
ANC_PATH = os.path.join('data' , 'anchor')
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

# Data Collection
!tar -xf lfw.tgz

base_dir = 'lfw'
# Iterate over each subdirectory in the base directory
for directory in os.listdir(base_dir):
    sub_dir_path = os.path.join(base_dir, directory)
    if os.path.isdir(sub_dir_path):
        for file in os.listdir(sub_dir_path):
            EX_PATH = os.path.join(sub_dir_path, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)


# Open camera and capture images
cap = cv2.VideoCapture(0) # Maybe the number is different for you?
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Crop the frame to the desired region
    frame = frame[120:120+250, 200:200+250, :]

    # Save the frame as an image in ANC_PATH when 'a' is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    # Save the frame as an image in POS_PATH when 'p' is pressed
    if cv2.waitKey(1) & 0xFF == ord('p'):
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    # Display the frame in a window
    cv2.imshow('Image Collector', frame)

    # Break the loop and exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()