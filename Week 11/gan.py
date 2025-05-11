import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import time  # Import the time module
import os
from PIL import Image  # Import the Image module from PIL

GENERATE_RES = 3 # Generation resolution factor 
# (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image 
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
EPOCHS = 50
BATCH_SIZE = 16  # Reduced batch size to reduce memory usage
BUFFER_SIZE = 10000  # Reduced buffer size

print(f"Will generate {GENERATE_SQUARE}px square images.")

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# Load CIFAR-10 dataset
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# Use only 10,000 images
train_images = train_images[:10000]

# Preprocess the data
train_images = train_images.astype(np.float32)
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Resize images if necessary
if GENERATE_SQUARE != 32:
    train_images = tf.image.resize(train_images, [GENERATE_SQUARE, GENERATE_SQUARE])

# Shuffle and batch the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Input(shape=(seed_size,)))
    model.add(Dense(4*4*256, activation="relu"))
    model.add(Reshape((4, 4, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    # Output resolution, additional upsampling
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if GENERATE_RES > 1:
        model.add(UpSampling2D(size=(GENERATE_RES, GENERATE_RES)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model

def build_discriminator(image_shape):
    model = Sequential()

    model.add(Input(shape=image_shape))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# Example usage of the generator and discriminator
generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)
discriminator = build_discriminator((GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

generator.summary()
discriminator.summary()

# Compile the models
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Combined model
z = Input(shape=(SEED_SIZE,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Save generated images
def save_images(cnt, noise):
    image_array = np.full((PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE + PREVIEW_MARGIN)), 
                           PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE + PREVIEW_MARGIN)), 3), 
                          255, dtype=np.uint8)

    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (GENERATE_SQUARE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (GENERATE_SQUARE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + GENERATE_SQUARE, c:c + GENERATE_SQUARE] = generated_images[image_count] * 255
            image_count += 1

    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

# Training function
def train(dataset, epochs):
    fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            # Train Discriminator
            noise = np.random.normal(0, 1, (BATCH_SIZE, SEED_SIZE))
            generated_images = generator.predict(noise)

            real_labels = np.ones((BATCH_SIZE, 1))
            fake_labels = np.zeros((BATCH_SIZE, 1))

            d_loss_real = discriminator.train_on_batch(image_batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (BATCH_SIZE, SEED_SIZE))
            g_loss = combined.train_on_batch(noise, real_labels)

        save_images(epoch, fixed_seed)
        print(f'{epoch + 1}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]')
        print(f'Time for epoch {epoch + 1} is {hms_string(time.time() - start)}')

# Train the model
train(train_dataset, EPOCHS)