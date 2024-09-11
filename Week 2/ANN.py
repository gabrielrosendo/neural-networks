import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to values between 0 and 1
# Pixel values are 0 to 255 - divide by 255 to normalize
# 0 is black, 255 is white
# After normalization, 0 is black, 1 is white
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the neural network model
# Sequential model allows us to build a model layer by layer
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D array
    # Each Dense command adds a fully connected layer with the specified number of neurons
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])

# Compile the model
# Specify the optimizer, loss function, and metrics to track during training
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# Epochs is the number of times the model will cycle through the data
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Save the model if needed
model.save("mnist_model.h5")

# Predictions

import matplotlib.pyplot as plt

# Get predictions on the test set
predictions = model.predict(test_images)

# Convert predictions from one-hot encoded to labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Function to plot images and their predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel(f"Predicted: {predicted_label} (True: {true_label})", color=color)

# Plot some test images along with their predicted and true labels
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, true_labels, test_images)
plt.tight_layout()
plt.show()

