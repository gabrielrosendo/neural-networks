import numpy as np
import re
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.corpus import gutenberg

# Download required NLTK data
nltk.download('gutenberg')

def load_gutenberg_texts():
    """Load and combine selected Gutenberg texts"""
    selected_texts = [
        'austen-pride.txt',  # Pride and Prejudice
        'carroll-alice.txt', # Alice in Wonderland
        'shelley-frankenstein.txt', # Frankenstein
        'doyle-case.txt'  # Sherlock Holmes
    ]
    
    combined_text = ""
    for text_id in selected_texts:
        try:
            # Get raw text
            text = gutenberg.raw(text_id)
            # Basic preprocessing
            text = text.replace('\n', ' ')
            combined_text += text + " "
            print(f"Loaded {text_id}: {len(text)} characters")
        except Exception as e:
            print(f"Error loading {text_id}: {e}")
    
    return combined_text

def preprocess_text(text):
    """Clean and preprocess the text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove Project Gutenberg headers and footers (approximate)
    text = re.sub(r'\*\*\* START OF .* \*\*\*', '', text)
    text = re.sub(r'\*\*\* END OF .* \*\*\*', '', text)
    
    # Remove chapter headings (common in Gutenberg texts)
    text = re.sub(r'chapter [IVXLC]+', '', text, flags=re.IGNORECASE)
    
    # Remove non-alphabet characters but keep basic punctuation
    text = re.sub(r'[^a-z\s\.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load and preprocess the text
source_text = load_gutenberg_texts()
source_text = preprocess_text(source_text)
print(f"\nProcessed text length: {len(source_text)} characters")
print("Sample of processed text:")
print(source_text[:200])

# Tokenize the text
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([source_text])
encoded_text = tokenizer.texts_to_sequences([source_text])[0]

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Create sequences
sequence_length = 100
sequences = []
for i in range(sequence_length, len(encoded_text)):
    seq = encoded_text[i-sequence_length:i+1]
    sequences.append(seq)

# Convert to numpy arrays
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_file = 'glove.6B.50d.txt'  # Path to your GloVe file
embeddings_index = load_glove_embeddings(glove_file)
print(f'Loaded {len(embeddings_index)} word vectors.')

# Create the embedding matrix
def create_embedding_matrix(tokenizer, embeddings_index, embedding_dim):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_dim = 50  # Dimension of GloVe embeddings
embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index, embedding_dim)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length, weights=[embedding_matrix], trainable=False))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=3)

# Train the model
model.fit(X, y, epochs=10, batch_size=256, callbacks=[early_stopping])

# Generate text
def generate_text(model, tokenizer, sequence_length, seed_text, num_chars):
    result = []
    in_text = seed_text
    for _ in range(num_chars):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded[-sequence_length:])
        encoded = np.pad(encoded, (sequence_length - len(encoded), 0), 'constant')
        yhat = model.predict(encoded.reshape(1, sequence_length), verbose=0)
        out_char = tokenizer.index_word[np.argmax(yhat)]
        in_text += out_char
        result.append(out_char)
    return ''.join(result)

"""# Example usage
seed_text = "once upon a time"
generated_text = generate_text(model, tokenizer, sequence_length, seed_text, 200)
print(generated_text)"""
# Generate sample text
print("\nGenerating sample text...")
seed_texts = [
    "once upon a time",
    "it was a dark and stormy night",
    "dear mr holmes",
    "she was certain that"
]

for seed in seed_texts:
    generated = generate_text(model, tokenizer, sequence_length, seed, 200)
    print(f"\nSeed: {seed}")
    print(f"Generated text: {generated}")
