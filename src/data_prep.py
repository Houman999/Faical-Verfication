# Library
import tensorflow as tf
import numpy as np


# Setup paths
POS_PATH = "YOUR PATH"
NEG_PATH = "YOUR PATH"
ANC_PATH = "YOUR PATH"

# Load datasets
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH + '/*').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*').take(3000)

# Preprocess function
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# Create labeled datasets
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(3000))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(3000))))

# Combine datasets
data = positives.concatenate(negatives)

# Preprocess and map function
def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img)), label

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
data = data.batch(16)
data = data.prefetch(tf.data.AUTOTUNE)

# Calculate the size of the dataset
data_size = tf.data.experimental.cardinality(data).numpy()

# Training partition (70% of the data)
train_data = data.take(round(data_size * 0.7))

# Testing partition (30% of the data)
test_data = data.skip(round(data_size * 0.7))
test_data = test_data.take(round(data_size * 0.3))