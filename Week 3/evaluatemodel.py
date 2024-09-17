import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re

# Load Dataset
data = fetch_20newsgroups(subset='all')
X, y = data.data, data.target
target_names = data.target_names  # Get the category names

# Preprocess text data: Convert to lowercase, remove punctuation, tokenize
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

def evaluate_model(model_path):
    # Load the saved model
    model = load_model(model_path)

    # Make predictions on the test data
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

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

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

evaluate_model("newsgroups.h5")