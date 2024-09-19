
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import random
import os

# 1. Feature Extraction Function
def extract_features(transaction, known_users):
    amount = transaction['amount']
    user_id = transaction['userID']
    item_id = transaction['itemID']

    features = {}
    features['amount'] = amount
    features['is_high_amount'] = 1 if amount > 500 else 0
    features['is_new_user'] = 0 if user_id in known_users else 1

    # Update known users
    known_users.add(user_id)

    return features

# 2. Simulate Training Data
def simulate_training_data(num_samples=1000):
    X = []
    y = []
    known_users = set()

    for _ in range(num_samples):
        transaction = {
            'transactionID': _,
            'userID': random.randint(1, 50),
            'amount': random.uniform(10.0, 1000.0),
            'itemID': random.randint(1, 100)
        }
        features = extract_features(transaction, known_users)
        X.append(list(features.values()))
        # Simulate labels: high amount and new user increases fraud risk
        label = 1 if ((features['is_high_amount'] and features['is_new_user'])) \
            else 0
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


# 3. Create the Model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    return model

# 4. Main Training Function
def main():
    # Prepare training data
    X_train, y_train = simulate_training_data()
    input_dim = X_train.shape[1]

    # Create and train the model
    model = create_model(input_dim)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the model
    model_filename = 'model.keras'
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
