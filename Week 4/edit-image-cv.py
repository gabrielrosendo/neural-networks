import cv2
import pandas as pd
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Read the original image
image = cv2.imread('wolf.jpg')

# Convert BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the kernel for inverting the image
kernel = np.array([[-1, 0, -1], [-1, 9, -1], [-1, -1, -1]])

# Apply the kernel to the image
inverted_image = cv2.filter2D(image, -1, kernel)

# Convert the inverted image to RGB for displaying with matplotlib
inverted_image_rgb = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB)

# Plot the original and inverted images side by side
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Inverted image
plt.subplot(1, 2, 2)
plt.imshow(inverted_image_rgb)
plt.title('Image with filter')
plt.axis('off')

# Show the plot
plt.show()