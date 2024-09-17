# Required Libraries
# https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.layers import Dense
import re

# Load Dataset
data = fetch_20newsgroups(subset='all')
X, y = data.data, data.target
target_names = data.target_names  # Get the category names

# TODO: Preprocess text data: Convert to lowercase, remove punctuation, tokenize
def preprocess_text(text_data):
    text = text_data.lower()
    clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
    return clean_text

preprocessed_X = [preprocess_text(text) for text in X]

# Convert Text Data to Numerical Format
vectorizer = TfidfVectorizer(max_features=5000) # Limit to 5000 most frequent words
X_vectorized = vectorizer.fit_transform(preprocessed_X)
X_vectorized = X_vectorized.toarray()

# Split data
X_train, X_test, y_train, y_test, original_X_train, original_X_test = train_test_split(X_vectorized, y, X, test_size=0.2, random_state=42)

# Design Neural Network Architecture
model = Sequential()
# Input layer
model.add(Dense(units = 128, activation = 'relu'))
# TODO: Add First Hidden Layer
model.add(Dense(units = 64, activation = 'relu'))
# TODO: Add Second Hidden Layer
model.add(Dense(units = 32, activation = 'relu'))   
# TODO: Add Output Layer
model.add(Dense(units = 20, activation = 'softmax'))
# Compile the Model
# TODO: Compile model specifying optimizer, loss and metrics

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
)
# Train the Model
# TODO: Fit the model using training data
# Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the Model
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Display text and label
for i in range(len(X_test)):
    print(f"Text: {original_X_test[i]}")
    print(f"True Label: {target_names[y_test[i]]}")
    print(f"Predicted Label: {target_names[y_pred[i]]}")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model if needed
model.save("newsgroups.keras")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMetrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")