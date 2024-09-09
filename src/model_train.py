# Library
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the embedding model
def make_embedding():
    inp = layers.Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = layers.Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = layers.MaxPooling2D((2, 2), padding='same')(c1)

    # Second block
    c2 = layers.Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = layers.MaxPooling2D((2, 2), padding='same')(c2)

    # Third block
    c3 = layers.Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = layers.MaxPooling2D((2, 2), padding='same')(c3)

    # Final embedding block
    c4 = layers.Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = layers.Flatten()(c4)
    d1 = layers.Dense(4090, activation='sigmoid')(f1)

    return models.Model(inputs=inp, outputs=d1)

# Define the Siamese model
def create_siamese_model(input_shape):
    embedding_model = make_embedding()

    input_1 = layers.Input(shape=input_shape)
    input_2 = layers.Input(shape=input_shape)

    # Generate embeddings for both inputs
    encoded_1 = embedding_model(input_1)
    encoded_2 = embedding_model(input_2)

    # Compute the Euclidean distance
    distance = layers.Lambda(lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))([encoded_1, encoded_2])

    # Use a Dense layer to produce the final similarity score
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = models.Model(inputs=[input_1, input_2], outputs=output)
    return model

input_shape = (100, 100, 3)
siamese_model = create_siamese_model(input_shape)
siamese_model.summary()

# Compile the model
siamese_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


siamese_model.save('siamese_model.h5')
