import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding, Reshape, TimeDistributed, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import requests

# Define constants
VOCAB_SIZE = 5000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
SEQUENCE_LENGTH = 30
LATENT_DIM = 100
NUM_CLASSES = 5  # Number of condition classes
BATCH_SIZE = 64

# Download dataset (same as original script)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
try:
    response = requests.get(url, timeout=(5, 10))
    response.raise_for_status()
    with open("train.txt", "wb") as file:
        file.write(response.content)
except requests.exceptions.RequestException as e:
    print(f"Error downloading dataset: {e}")
    exit(1)

# Load and tokenize text data
def load_text_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = text.splitlines()
    return sentences

# Tokenize and prepare sequences
sentences = load_text_data("train.txt")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding='post')

# Modified data loader with conditions
def data_loader(batch_size, num_classes):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    # Generate random condition labels
    condition_labels = np.random.randint(0, num_classes, batch_size)
    one_hot_labels = tf.keras.utils.to_categorical(condition_labels, num_classes=num_classes)
    
    # Sample sequences
    idx = np.random.randint(0, len(padded_sequences), batch_size)
    real_sequences = np.array([padded_sequences[i] for i in idx])
    
    return real_sequences, real_labels, fake_labels, one_hot_labels
# Modify generator to accept conditional input
def build_conditional_generator(vocab_size, embedding_dim, hidden_dim, sequence_length, num_classes):
    # Latent noise input
    noise_input = Input(shape=(LATENT_DIM,))
    
    # Condition input
    condition_input = Input(shape=(num_classes,))
    
    # Concatenate noise and condition
    merged_input = Concatenate()([noise_input, condition_input])
    
    # Generator layers
    x = Dense(hidden_dim * sequence_length, activation='relu')(merged_input)
    x = Reshape((sequence_length, hidden_dim))(x)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    
    # Create model
    model = Model([noise_input, condition_input], output)
    return model

# Modify discriminator to accept conditional input
def build_conditional_discriminator(vocab_size, embedding_dim, hidden_dim, sequence_length, num_classes):
    # Sequence input
    sequence_input = Input(shape=(sequence_length,))
    
    # Condition input
    condition_input = Input(shape=(num_classes,))
    
    # Embedding layer for sequence
    x = Embedding(vocab_size, embedding_dim, input_length=sequence_length)(sequence_input)
    x = LSTM(hidden_dim)(x)
    
    # Concatenate sequence features with condition
    merged = Concatenate()([x, condition_input])
    
    # Output layer
    output = Dense(1, activation='sigmoid')(merged)
    
    # Create and compile model
    model = Model([sequence_input, condition_input], output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Modify GAN to incorporate conditions
def build_conditional_gan(generator, discriminator):
    discriminator.trainable = False
    
    # Generator inputs
    noise_input = Input(shape=(LATENT_DIM,))
    condition_input = Input(shape=(NUM_CLASSES,))
    
    # Generate sequences
    generated_sequence_probs = generator([noise_input, condition_input])
    
    # Create an ArgmaxLayer as a separate layer
    class CustomArgmaxLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.argmax(inputs, axis=-1)
    
    # Apply argmax and pass to discriminator
    argmax_layer = CustomArgmaxLayer()
    generated_sequences = argmax_layer(generated_sequence_probs)
    
    # Discriminator output
    gan_output = discriminator([generated_sequences, condition_input])
    
    # Create and compile the GAN model
    gan = Model([noise_input, condition_input], gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    return gan

# Modified training function
def train_conditional_seqgan(generator, discriminator, gan, data_loader, epochs):
    for epoch in range(epochs):
        # Sample batches with conditions
        real_sequences, valid, fake, conditions = data_loader(BATCH_SIZE, NUM_CLASSES)
        
        # Generate noise
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        
        # Generate sequences with conditions
        generated_sequences_probs = generator.predict([noise, conditions])
        generated_sequences = np.argmax(generated_sequences_probs, axis=-1)
        
        # Train discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_sequences, conditions], valid)
        d_loss_fake = discriminator.train_on_batch([generated_sequences, conditions], fake)
        discriminator_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        valid_y = np.ones((BATCH_SIZE, 1))
        
        generator_loss = gan.train_on_batch([noise, conditions], valid_y)
        
        # Log progress
        print(f"Epoch: {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

# Modify text generation to include conditions
def generate_conditional_text(generator, tokenizer, conditions, num_sequences=5, temperature=1.0, max_length=30):
    noise = np.random.normal(0, 1, (num_sequences, LATENT_DIM))
    generated_sequences = []

    for _ in range(num_sequences):
        current_sequence = []
        current_noise = noise[_].reshape(1, -1)
        current_condition = conditions[_].reshape(1, -1)

        for step in range(max_length):
            generated_sequences_probs = generator.predict([current_noise, current_condition])
            generated_sequences_probs = generated_sequences_probs[0, -1, :]  # Get probabilities for last step

            # Apply temperature
            generated_sequences_probs = np.log(generated_sequences_probs) / temperature
            generated_sequences_probs = np.exp(generated_sequences_probs)
            generated_sequences_probs /= np.sum(generated_sequences_probs)

            # Prevent repetition by reducing probability of already used words
            if current_sequence:
                for word_idx in current_sequence:
                    generated_sequences_probs[word_idx] *= 0.1  # Heavily reduce probability of repeated words
                generated_sequences_probs /= np.sum(generated_sequences_probs)

            # Sample next word
            next_word_idx = np.random.choice(len(generated_sequences_probs), p=generated_sequences_probs)
            current_sequence.append(next_word_idx)

            # Optional: Add stopping condition if needed

        generated_sequences.append(current_sequence)
    
    # Convert indices to words
    generated_texts = []
    for sequence in generated_sequences:
        text = tokenizer.sequences_to_texts([sequence])[0]
        text = text.replace("<OOV>", "").strip()
        generated_texts.append(text)
    
    return generated_texts
def apply_repetition_penalty(generated_sequence, seq_probs, penalty_factor=1.5):
    unique_words = set()
    
    # Loop through the sequence and apply penalty
    for i in range(len(generated_sequence)):
        word_idx = generated_sequence[i]  # Word index in the sequence
        if word_idx in unique_words:  # If the word has been generated before
            seq_probs[i] /= penalty_factor  # Penalize by dividing the probability
        else:
            unique_words.add(word_idx)
    
    # Normalize the probabilities to ensure they sum to 1
    seq_probs = seq_probs / np.sum(seq_probs)
    
    return seq_probs


# Main execution
# Create condition samples (one-hot encoded)
sample_conditions = tf.keras.utils.to_categorical(
    np.random.randint(0, NUM_CLASSES, 5), 
    num_classes=NUM_CLASSES
)

# Initialize models with conditional inputs
generator = build_conditional_generator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, SEQUENCE_LENGTH, NUM_CLASSES)
discriminator = build_conditional_discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, SEQUENCE_LENGTH, NUM_CLASSES)
gan = build_conditional_gan(generator, discriminator)

# Train the Conditional SeqGAN
train_conditional_seqgan(generator, discriminator, gan, data_loader, epochs=60)

# Generate conditional text
generated_texts = generate_conditional_text(generator, tokenizer, sample_conditions)
print("Generated Texts:")
for i, text in enumerate(generated_texts, 1):
    print(f"{i}. {text}")