from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

# Load and display an image
img = image.load_img("dataset/training/not_happy/Image_1.jpg")
plt.imshow(img)

# Check the shape of the image using OpenCV
cv2.imread("dataset/training/not_happy/Image_1.jpg").shape

# Data augmentation and scaling for the training and validation datasets
train = ImageDataGenerator(rescale=1/248)
validation = ImageDataGenerator(rescale=1/248)

# Load the training and validation datasets from directories
train_dataset = train.flow_from_directory("dataset/training", target_size=(200, 200), batch_size=3, class_mode="binary")
validation_dataset = train.flow_from_directory("dataset/validation", target_size=(200, 200), batch_size=3, class_mode="binary")

# Print the class indices of the training dataset
train_dataset.class_indices

# Print the classes of the training dataset
train_dataset.classes

# Define the neural network model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    ##
    tf.keras.layers.Flatten(),
    ##
    tf.keras.layers.Dense(512, activation="relu"),
    ##
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Compile the model with loss, optimizer, and metrics
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

# Train the model on the training dataset
model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=30, validation_data=validation_dataset)

# Perform predictions on test images
dir_path = "dataset/testing"
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + "//" + i, target_size=(200, 200))
    # Display the image using plt.imshow()
    plt.imshow(img)
    plt.axis('off')  # Optional: Turn off axis ticks and labels
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print("You are happy")
    else:
        print("You are not happy")
