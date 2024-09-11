import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os

# Load the trained model
model = load_model("mnist_model.h5")

# Compile the model with dummy metrics to avoid the warning
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was successfully loaded
    if img is None:
        raise ValueError(f"Error: Could not open or read the image file: {image_path}")
    
    # Invert the image (assuming black digit on white background)
    # Model was trained on white digits on black background
    img = 255 - img
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Normalize
    img = img / 255.0
    
    # Expand dimensions to represent a single sample
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    
    return img

# Path to your handwritten image
image_folder = "test/"

for image in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image)
    print(image_path)
    
    img = preprocess_image(image_path)

    # Predict using the model
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)

    # Debug: Print prediction probabilities
    print("Prediction Probabilities:", predictions)

    # Visualize the result
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()
    
    temp = input("Do you want to continue? (y/n) ")
    if temp.lower() == 'n':
        break
